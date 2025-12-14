from torch import nn


class AlwaysDropout(nn.Dropout):
    """An extension of the PyTorch Dropout class that applies dropout during inference"""

    def __init__(self, p):
        super().__init__(p)

    def forward(self, input):
        self.training = True
        return super().forward(input)
