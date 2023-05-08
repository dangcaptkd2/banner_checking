import torch.nn as nn
import torch 
import numpy as np
import torchvision.models as models

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(resnet50.children())[:-1])
        
    def forward(self, x):
        x = self.model(x)
        return x.squeeze()
    
if __name__ == '__main__':
    model = Network()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Numer of params:", params)
    img = torch.rand(1,3,224,224)
    r = model(img)
    print("Output shape of model for single image:", r.shape)