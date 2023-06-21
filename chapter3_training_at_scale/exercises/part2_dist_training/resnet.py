# %%

import torch
import torchvision.models as models
from torchinfo import summary

resnet34 = models.resnet34(pretrained=True)

class PipeNet34(torch.nn.Module):
    def __init__(self):
        super(PipeNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        
    def forward(self, x):
        # Run each layer individually and return intermediate outputs
      
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        
        x = self.resnet.layer2(x)
        
        x = self.resnet.layer3(x)
        
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        return x

pipenet34 = PipeNet34()


# %%
