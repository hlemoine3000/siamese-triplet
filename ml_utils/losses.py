import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class QuadrupletLoss_bad(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin1, margin2, lamda=0.1):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.lamda = lamda

    def forward(self, anchor, positive, negative, target, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        distance_target = (anchor - target).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative - self.lamda * distance_target + self.margin1)
        return losses.mean() if size_average else losses.sum()


class QuadrupletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin1, margin2, lamda=0.1):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.lamda = lamda

    def forward(self, anchor, positive, negative, target, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        distance_target = (anchor - target).pow(2).sum(1)  # .pow(.5)

        triplet_losses = F.relu(distance_positive - distance_negative + self.margin1)
        target_losses = F.relu(distance_positive - distance_target + self.margin2)

        if size_average:
            quadruplet_loss = triplet_losses.mean() + self.lamda * target_losses.mean()
        else:
            quadruplet_loss = triplet_losses.sum() + self.lamda * target_losses.sum()

        return quadruplet_loss


class QuadrupletLoss2(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin1, margin2, lamda=0.1):
        super(QuadrupletLoss2, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.lamda = lamda
        self.loss_keys = ['cls_losses', 'sep_losses', 'adv_losses']

    def forward(self, anchor, positive, negative, target):

        losses_dict = {}

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        distance_target = (anchor - target).pow(2).sum(1)  # .pow(.5)

        cls_losses = F.relu(distance_positive - distance_negative + self.margin1)
        sep_losses = F.relu(distance_positive - distance_target + self.margin2)
        adv_losses = distance_target - distance_negative

        quadruplet_loss = cls_losses.mean() + sep_losses.mean() + self.lamda * adv_losses.mean()

        losses_dict['cls_losses'] = cls_losses.mean()
        losses_dict['sep_losses'] = sep_losses.mean()
        losses_dict['adv_losses'] = self.lamda * adv_losses.mean()

        return quadruplet_loss, losses_dict
