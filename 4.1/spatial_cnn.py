import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SpatialCNN(nn.Module):
    def __init__(self, num_classes=101, dropout_p=0.7, freeze_until_layer=2):
        super().__init__()

        # Load pretrained ResNet-18
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        for name, param in self.resnet.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

        # Replace classifier head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)