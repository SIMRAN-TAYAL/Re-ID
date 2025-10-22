import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, embeddings, labels):
        n = embeddings.size(0)

        # Compute pairwise distance
        dist = torch.cdist(embeddings, embeddings, p=2)

        # Create mask for positives and negatives
        mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        mask_neg = ~mask_pos

        # For each anchor, select hardest positive and negative
        dist_ap = torch.max(dist * mask_pos.float(), dim=1)[0]
        dist_an = torch.min(dist + 1e5 * mask_pos.float(), dim=1)[0]

        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss.mean()
