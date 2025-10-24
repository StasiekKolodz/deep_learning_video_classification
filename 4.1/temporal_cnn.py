import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def modify_resnet_for_flow(model, num_in_channels=18):
    # Get the pretrained conv1 weights
    old_weights = model.conv1.weight.data  # shape [64, 3, 7, 7]
    # Average across RGB channels
    mean_weights = old_weights.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
    # Replicate across flow channels (x,y for T frames)
    new_weights = mean_weights.repeat(1, num_in_channels, 1, 1)  # [64, num_in_channels, 7, 7]

    # Replace conv1 with new layer
    model.conv1 = nn.Conv2d(num_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = new_weights

    return model

class TemporalCNN(nn.Module):
    def __init__(self, num_classes=101, num_in_channels=18):
        super().__init__()
        # Use weights instead of pretrained=True
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = modify_resnet_for_flow(self.resnet, num_in_channels)
        
        # Replace classifier head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    