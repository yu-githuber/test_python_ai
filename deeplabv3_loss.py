import torch
import torch.nn.functional as F

class DeepLabV3Loss(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Loss, self).__init__()
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs, targets, reduction='mean')
        return loss
