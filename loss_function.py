import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights)
        return ce_loss
    
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, pov_weight=None):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.pos_weight = pov_weight

    def forward(self, inputs, targets):
        loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(inputs, targets)
        return loss    
