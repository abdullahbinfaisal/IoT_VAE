import torch
import torch.nn as nn
import pytorch_msssim
import torch.nn.functional as F
from torchvision.models import vgg11, VGG11_Weights

class VGGPerceptual(nn.Module):
    def __init__(self, perc_layers=None):
        super().__init__()
        weights = VGG11_Weights.IMAGENET1K_V1
        self.vgg = vgg11(weights=weights).features.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        # Updated perceptual layers for VGG11
        # VGG11 has fewer conv layers than VGG16, so valid layer indices are different
        self.perc_layers = perc_layers or [3, 6, 11]  # These are ReLU layers after conv2, conv4, conv6

        self.loss_fn = nn.MSELoss()

        # ImageNet normalization values (same as before)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)

        loss = 0
        with torch.no_grad():
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                y = layer(y)
                if i in self.perc_layers:
                    loss += self.loss_fn(x, y)
            return loss



class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        alpha: weight for cosine similarity loss
        beta: weight for MSE loss
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        # Detach teacher so gradients don't flow through it
        teacher_feat = teacher_feat.detach()

        # Global Average Pooling: [B, C, H, W] -> [B, C]
        s_pooled = F.adaptive_avg_pool2d(student_feat, (1, 1)).squeeze(-1).squeeze(-1)
        t_pooled = F.adaptive_avg_pool2d(teacher_feat, (1, 1)).squeeze(-1).squeeze(-1)

        # Cosine Similarity Loss: want vectors pointing in same direction => 1 - cos_sim
        cosine_loss = 1 - F.cosine_similarity(s_pooled, t_pooled, dim=1).mean()

        # Standard MSE Loss over full feature maps
        mse_loss = self.mse(student_feat, teacher_feat)

        # Combined Hybrid Loss
        loss = self.alpha * cosine_loss + self.beta * mse_loss
        return loss
    



def get_loss_functions():
    msssim_loss = lambda x, y: 1 - pytorch_msssim.ms_ssim(x, y, data_range=1.0, size_average=True)
    vgg_perceptual = VGGPerceptual()
    distillation_loss = DistillationLoss()
    return msssim_loss, vgg_perceptual, distillation_loss
