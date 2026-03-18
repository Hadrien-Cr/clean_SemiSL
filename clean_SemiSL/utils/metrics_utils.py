import torch
from torchmetrics import Metric

class ClassificationTopKAccuracy(Metric):
    def __init__(self, topk: int = 1,):
        super().__init__()
        self.topk = topk
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: torch.Tensor, target: torch.Tensor):
        n = len(target)
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        self.correct += correct[:self.topk].view(-1).float().sum()
        self.total += n

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.)
        return self.correct.float() / self.total

def classification_top_1_accuracy():
    return ClassificationTopKAccuracy(topk=1)

def classification_top_5_accuracy(**kwargs):
    return ClassificationTopKAccuracy(topk=5)

