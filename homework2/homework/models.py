import torch


class CNNClassifier(torch.nn.Module):
    """
    Your code here` ASDCVB N  M,.
    """
    def __init__(self, layers=[32, 64, 128], n_input_channels=3, n_output_channels=6, kernel_size=3):
        super().__init__()
        L = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride=2, padding=(kernel_size-1)//2))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(3, padding=1, stride=2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)





    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """

        classifier = self.classifier(self.network(x).mean(dim=[2, 3]))
        return classifier



def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
