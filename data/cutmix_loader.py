from .cutmix import CutMix

class CutMixLoader:
    def __init__(self, loader, alpha=1.0):
        """
        Args:
            loader (DataLoader): 원본 데이터 로더
            alpha (float): CutMix에서 사용할 Beta 분포의 하이퍼파라미터
        """
        self.loader = loader
        self.cutmix = CutMix(alpha)

    def __iter__(self):
        for data, targets in self.loader:
            data, targets_a, targets_b, lam = self.cutmix.apply(data, targets)
            yield data, targets_a, targets_b, lam

    def __len__(self):
        return len(self.loader)
