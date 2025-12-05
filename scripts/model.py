import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def get_resnet_model(
    backbone: str="resnet18",
    num_classes=2, 
    pretrained: bool=True,
    freeze_backbone: bool=False,
    in_channels: int=3,
) -> torch.nn.Module:
    """
    
    Builds a ResNet model for binary classification.

    Args:
        backbone: "resnet18" or "resnet34".
        num_classes: Number of output classes (2 for binary).
        pretrained: Whether to load ImageNet pretrained weights.
        freeze_backbone: If True, freeze all layers except the final FC.
        in_channels: Number of input channels (1 for X-rays, 3 for RGB).

    Returns:
        model (torch.nn.Module): ResNet model with modified final layer.

    """
    # Select backbone architecture
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights) # Load a pre-trained ResNet18 model
    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights) # Load a pre-trained ResNet34 model
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Adapt first conv layer if using 1-channel input
    if in_channels != 3:
        old_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels, 
            old_conv1.out_channels, 
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride, 
            padding=old_conv1.padding, 
            bias=old_conv1.bias is not None,
        )
        if pretrained:
            # Initialize new conv1 weights by averaging old conv weights across over RGB channels
            with torch.no_grad():
                if in_channels == 1:
                    model.conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)
                else:
                    # For weird in_channels>3, just repeat / truncate as needed
                    repeat = int((in_channels + 2) // 3)
                    w = old_conv1.weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
                    model.conv1.weight[:] = w
                            
    # Freeze backbone layers if specified
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes) # Binary classification, so num_classes = 2
    
    return model

def get_model(num_classes=2):
    """
    Backwards-compatible helper that returns a ResNet18 by default.
    
    Args:
        :param num_classes: number of output classes
    
    Returns: 
        ResNet18 model for binary classification
    """
    return get_resnet_model(
        backbone="resnet18",
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False,
        in_channels=3, # assuming RGB images
    )
    
# ----- LEGACY CODE: PREVIOUS MODEL DEFINITIONS -----
    
# # Freeze the early layers since we are fine-tuning and not retraining the entire model
# for param in model.parameters():
#     param.requires_grad = False
    
# # Modify the final fully connected layer to match the number of classes
# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, num_classes) # Binary classification, so num_classes = 2

# return model

# Simple CNN model for image classification from scratch

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.fc1 = nn.Linear(64 * 6 * 6, 128)
#         self.fc2 = nn.Linear(128, num_classes)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
    
# def get_model(num_classes=2):
#     """
#     Returns a pre-trained ResNet model with a modified final layer for binary classification.
#     Args:
#         num_classes (int): Number of output classes. Default is 2 for binary classification.
#     Returns:
#         model (torch.nn.Module): The modified ResNet model.
#     """
#     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Load a pre-trained ResNet model
    
#     # Freeze the early layers since we are fine-tuning and not retraining the entire model
#     for param in model.parameters():
#         param.requires_grad = False
        
#     # Modify the final fully connected layer to match the number of classes
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes) # Binary classification, so num_classes = 2
    
#     return model

