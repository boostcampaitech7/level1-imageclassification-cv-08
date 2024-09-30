import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate cross entropy loss
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Get softmax probabilities
        pt = torch.exp(-BCE_loss)  # pt is the predicted probability for the true class
        # Compute Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps: float = 0.1, num_classes: int = 500, reduction: str = 'mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.eps = eps
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        # 원 핫 인코딩 생성
        target = target.unsqueeze(1)  # (N, 1)
        one_hot = torch.zeros_like(x).scatter_(1, target, 1)  # (N, C)
        # Label smoothing 적용
        smooth_label = one_hot * (1 - self.eps) + self.eps / self.num_classes
        # Cross entropy 손실 계산
        loss = -torch.sum(smooth_label * F.log_softmax(x, dim=-1), dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
                
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        """
        :param gamma_neg: negative class에 대한 focusing parameter
        :param gamma_pos: positive class에 대한 focusing parameter
        :param clip: prediction 확률을 clip하는 값 (값이 너무 작아지지 않도록 방지)
        :param eps: log의 계산 시 발생하는 zero log를 방지하기 위한 작은 값
        :param reduction: 'mean' 또는 'sum'으로 결과를 축약하는 방식
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Sigmoid로 예측 값을 확률로 변환
        probas = torch.sigmoid(inputs)
        
        # Clip 확률 (0과 1 사이의 작은 값을 보정)
        if self.clip is not None:
            probas = torch.clamp(probas, min=self.clip, max=1.0 - self.clip)

        # Positive와 Negative에 대한 loss 계산
        pos_loss = targets * torch.log(probas + self.eps) * ((1 - probas) ** self.gamma_pos)
        neg_loss = (1 - targets) * torch.log(1 - probas + self.eps) * (probas ** self.gamma_neg)
        
        # Loss 합산
        loss = -pos_loss - neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss