import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    poss_det = heatmap - (cls > heatmap).float() * 1e5
    if max_det > poss_det.numel():
        max_det = poss_det.numel()
    s, l = torch.topk(poss_det.view(-1), max_det)
    return [(float(i), int(j) % heatmap.size(1), int(j) // heatmap.size(1))
            for i, j in zip(s.cpu(), l.cpu()) if s > min_score]




class Detector(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size//2, stride=stride,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, padding=1, stride=kernel_size // 2)
            )



            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity

    class UpConv(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.net = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.net(x))
    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=3, kernel_size=3, skip=True):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        c = 3
        self.skip = skip
        self.convs = len(layers)
        skip_layers = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpConv(c, l, kernel_size, 2))
            c = l
            if self.skip:
                c += skip_layers[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)
        self.size = torch.nn.Conv2d(c, 2, 1)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """

        self.mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.std = torch.Tensor([0.2064, 0.1944, 0.2252])
        x_norm = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device)

        up_skip = []
        for i in range(self.convs):
            # skip connections
            up_skip.append(x_norm)
            x_norm = self._modules['conv%d'%i](x_norm)


        for i in reversed(range(self.convs)):
            x_norm = self._modules['upconv%d'%i](x_norm)
            # crop the output of up-convolution.
            x_norm = x_norm[:, :, :up_skip[i].size(2), :up_skip[i].size(3)]
            # cat the skip connection with up-convolution
            if self.skip:
                x_norm = torch.cat([x_norm, up_skip[i]], dim=1)
        return self.classifier(x_norm), self.size(x_norm)




    def detect(self, image, **kwargs):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        cls, size = self.forward(image[None])
        size = size.cpu()
        return [[(s, x, y, float(size[0, 0, y, x]), float(size[0, 1, y, x]))
                 for s, x, y in extract_peak(c, max_det=30, **kwargs)] for c in cls[0]]



def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
