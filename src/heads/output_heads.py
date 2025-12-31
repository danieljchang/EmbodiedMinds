import torch.nn as nn
import torch


class ActionHead:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def forward(self, features):
        # Implement the forward pass for action classification
        pass

    def predict(self, features):
        # Implement the prediction logic
        pass


class ValueHead:
    def __init__(self):
        pass

    def forward(self, features):
        # Implement the forward pass for value estimation
        pass

    def predict(self, features):
        # Implement the prediction logic for value estimation
        pass


class OutputHeads(nn.Module):
    def __init__(self, in_dim, bins):
        """
        bins: list[int] length 7 giving number of classes per action dim
        """
        super().__init__()
        assert len(bins) == 7
        self.bins = bins
        self.heads = nn.ModuleList([nn.Linear(in_dim, b) for b in bins])

    def forward(self, x):
        # x: (B, in_dim)
        logits = [h(x) for h in self.heads]  # list of (B, bins_i)
        return logits

    def loss(self, logits, targets):
        # logits: list of (B, bins_i), targets: (B,7) longs
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        total = 0.0
        for i, log in enumerate(logits):
            target_dim = targets[:, i].clone()
            # Clamp targets to valid range [0, bins[i]-1] or keep -1 for invalid
            # Invalid targets (-1) are ignored by ignore_index=-1
            valid_mask = (target_dim >= 0) & (target_dim < self.bins[i])
            target_dim[~valid_mask] = -1  # Mark out-of-range targets as invalid
            total = total + loss_fn(log, target_dim)
        return total / len(logits)

    def predict(self, logits):
        # returns (B,7) ints
        preds = [log.argmax(dim=-1) for log in logits]
        return torch.stack(preds, dim=1)