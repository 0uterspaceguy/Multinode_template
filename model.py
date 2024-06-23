from torch import nn
from efficientnet_pytorch import EfficientNet

class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=101)
    
    def forward(self, x):
        return self.model(x)