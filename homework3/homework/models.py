import torch
import torch.nn.functional as F
import torchvision.transforms


class CNNClassifier(torch.nn.Module):
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
                torch.nn.MaxPool2d(3, padding = 1, stride=kernel_size//2)
            )
            torch.nn.init.xavier_normal_(self.net[0].weight)


            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity
    def __init__(self, layers=[32, 64, 128], n_output_channels=6, kernel_size=3):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)





    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        self.mean = torch.Tensor([0.3235, 0.3310, 0.3445])
        self.std = torch.Tensor([0.2533, 0.2224, 0.2483])
        x_norm = torchvision.transforms.Normalize(self.mean.to(x.device), self.std.to(x.device))(x)
        z = self.network(x_norm)

        return self.classifier(z.mean(dim=[2, 3]))






class FCN(torch.nn.Module):
    class UpConv(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                    stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[32, 64, 128], n_output_channels=5, kernel_size=3, skip=True):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        c = 3
        self.skip = skip
        self.convs = len(layers)
        skip_layers = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, CNNClassifier.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpConv(c, l, kernel_size, 2))
            c = l
            if self.skip:
                c += skip_layers[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)



    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        self.mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.std = torch.Tensor([0.2064, 0.1944, 0.2252])
        norm = torchvision.transforms.Normalize(self.mean.to(x.device), self.std.to(x.device))
        x_norm = norm(x)

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
        return self.classifier(x_norm)




model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
