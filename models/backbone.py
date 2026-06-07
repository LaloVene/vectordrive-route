import torch
import torch.nn as nn
import torchvision.models as models

class ImageBackbone(nn.Module):
    """
    2D Feature Extractor (ResNet-18) backbone.
    Processes the stitched cylindrical image and outputs stride-8 features.
    """
    def __init__(self, C_feat=256):
        super().__init__()
        # Use standard resnet18 architecture
        resnet = models.resnet18(weights=None)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1 # Stride 4, Output channels: 64
        self.layer2 = resnet.layer2 # Stride 8, Output channels: 128
        
        # Projection layer to map 128 channels to C_feat (256) channels
        self.proj = nn.Sequential(
            nn.Conv2d(128, C_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C_feat),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Input: (B, 3, H_out, W_out)
        Output: (B, C_feat, H_feat, W_feat)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.proj(x)
        return x

class LSSLiftStage(nn.Module):
    """
    Lift stage of Lift-Splat-Shoot.
    Predicts a categorical depth distribution and multiplies it with context features.
    """
    def __init__(self, C_feat=256, C_context=80, D_bins=40):
        super().__init__()
        self.C_context = C_context
        self.D_bins = D_bins
        
        # Context head (ConvBlock -> 1x1 Conv)
        self.context_conv = nn.Sequential(
            nn.Conv2d(C_feat, C_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_feat, C_context, kernel_size=1)
        )
        
        # Depth head (ConvBlock -> 1x1 Conv)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(C_feat, C_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_feat, D_bins, kernel_size=1)
        )
        
    def forward(self, x):
        """
        Input: (B, C_feat, H_feat, W_feat)
        Returns:
            frustum_tensor: (B, C_context, D_bins, H_feat, W_feat)
            depth_probs: (B, D_bins, H_feat, W_feat) (before/after softmax? After softmax for losses/supervision)
        """
        # Context features: shape (B, C_context, H_feat, W_feat)
        context = self.context_conv(x)
        
        # Depth logits: shape (B, D_bins, H_feat, W_feat)
        depth_logits = self.depth_conv(x)
        depth_probs = torch.softmax(depth_logits, dim=1)
        
        # Element-wise outer product:
        # (B, C_context, 1, H_feat, W_feat) * (B, 1, D_bins, H_feat, W_feat)
        # -> (B, C_context, D_bins, H_feat, W_feat)
        frustum_tensor = context.unsqueeze(2) * depth_probs.unsqueeze(1)
        
        return frustum_tensor, depth_probs
