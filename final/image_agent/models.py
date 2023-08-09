import torch
import torch.nn.functional as F
import numpy as np


def extract_peak(heatmap, max_pool_ks: int = 7, min_score: float = -5, max_det: int = 100):
    """
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    m_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    prob_det = heatmap - (m_cls > heatmap).float() * 1e5
    if max_det > prob_det.numel():
        max_det = prob_det.numel()
    score, position = torch.topk(prob_det.view(-1), max_det)
    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), position.cpu()) if s > min_score]

class Detector(torch.nn.Module):
    class conv(torch.nn.Module):
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
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
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


    class upconv(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.net = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.net(x))

    def __init__(self, layers=[32, 64, 128], n_input_channels=3, n_output_channels=2, kernel_size=3, skip=True):
        super().__init__()

        self.norm = torch.nn.BatchNorm2d(n_input_channels)

        self.min_size = np.power(2, len(layers) + 1)

        c = n_input_channels
        self.skip = skip
        self.convs = len(layers)
        skip_layers = [3] + layers[:-1]

        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.conv(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.upconv(c, l, kernel_size, 2))
            c = l
            if self.skip:
                c += skip_layers[i]
        self.size = torch.nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):
        x = self.norm(x)
        h = x.size(2)
        w = x.size(3)



        # Calculate
        up_skip = []

        '''
    

        # skip connections
        up_skip.append(x)
        x = self.conv0(x)
        up_skip.append(x)
        x = self.conv1(x)
        up_skip.append(x)
        x = self.conv2(x)
        up_skip.append(x)
        x = self.conv3(x)
        x = self.upconv3(x)
        x = x[:, :, :up_skip[3].size(2), :up_skip[3].size(3)]
        x = torch.cat([x, up_skip[3]], dim=1)
        x = self.upconv2(x)
        x = x[:, :, :up_skip[2].size(2), :up_skip[2].size(3)]
        x = torch.cat([x, up_skip[2]], dim=1)
        x = self.upconv1(x)
        x = x[:, :, :up_skip[1].size(2), :up_skip[1].size(3)]
        x = torch.cat([x, up_skip[1]], dim=1)
        x = self.upconv0(x)
        x = x[:, :, :up_skip[0].size(2), :up_skip[0].size(3)]
        x = torch.cat([x, up_skip[0]], dim=1)
        '''


        for i in range(self.convs):
            # skip connections
            up_skip.append(x)
            x= self._modules['conv%d'%i](x)


        for i in reversed(range(self.convs)):
            x = self._modules['upconv%d'%i](x)
            # crop the output of up-convolution.
            x = x[:, :, :up_skip[i].size(2), :up_skip[i].size(3)]
            # cat the skip connection with up-convolution
            if self.skip:
                x = torch.cat([x, up_skip[i]], dim=1)

        x = self.size(x)
        pred = x[:, 0, :h, :w]
        width = x[:, 1, :h, :w]

        return pred, width

    def detect(self, image, max_pool_ks=7, min_score=0.2, max_det=1):
        """
           Implement object detection here.
           @image: 3 x H x W image
           @min_socre: minimum score for a detection to be returned (sigmoid from 0 to 1)
           @return: One list of detections [(score, cx, cy, w, h), ...]
           Return Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        cls, sizes = self(image[None])
        cls = torch.sigmoid(cls.squeeze(0).squeeze(0))
        width = sizes.squeeze(0)
        return [(peak[0], peak[1], peak[2], (width[peak[2], peak[1]]).item())
                for peak in extract_peak(cls, max_pool_ks, min_score, max_det)]







def save_model(model, name: str = 'det.th'):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))


def load_model(name: str = 'det.th'):
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    return r