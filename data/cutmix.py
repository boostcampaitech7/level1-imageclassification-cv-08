import torch
import numpy as np

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def apply(self, data, targets):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, _, _ = data.size()
        rand_index = torch.randperm(batch_size).to(data.device)

        targets_a = targets
        targets_b = targets[rand_index]

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        return data, targets_a, targets_b, lam

    def rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

        cx, cy = np.random.randint(W), np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
