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
        super(QuadrupletLoss_bad, self).__init__()
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
        adv_losses = F.relu(distance_target - distance_negative)

        quadruplet_loss = cls_losses.mean() + sep_losses.mean() + self.lamda * adv_losses.mean()

        losses_dict['cls_losses'] = cls_losses.mean()
        losses_dict['sep_losses'] = sep_losses.mean()
        losses_dict['adv_losses'] = self.lamda * adv_losses.mean()

        return quadruplet_loss, losses_dict


class QuadrupletLoss3(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, lamda=0.1):
        super(QuadrupletLoss3, self).__init__()
        self.margin = margin
        self.lamda = lamda
        self.loss_keys = ['loss1', 'loss2', 'loss3', 'loss4']

    def forward(self, anchor, positive, negative, target):

        losses_dict = {}

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        distance_target = (anchor - target).pow(2).sum(1)  # .pow(.5)

        loss1 = F.relu(distance_positive - distance_negative + self.margin)
        loss2 = F.relu(distance_positive - distance_target + self.margin)
        loss3 = F.relu(distance_positive - distance_negative + 3 * self.margin)
        loss4 = F.relu(distance_positive - distance_target + 3 * self.margin)

        quadruplet_loss = loss1.mean() + loss2.mean() - self.lamda * (loss3.mean() + loss4.mean())

        losses_dict['loss1'] = loss1.mean()
        losses_dict['loss2'] = loss2.mean()
        losses_dict['loss3'] = - self.lamda * loss3.mean()
        losses_dict['loss4'] = - self.lamda * loss4.mean()

        return quadruplet_loss, losses_dict


class QuadrupletLoss4(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, lamda, margin_factor=2):
        super(QuadrupletLoss4, self).__init__()
        self.margin = margin
        self.lamda = lamda
        self.loss_keys = ['loss1', 'loss2', 'loss3', 'loss4', 'loss5', 'loss6']
        self.lamda = lamda
        self.margin_factor = margin_factor

    def forward(self, anchor, positive, negative, target):

        losses_dict = {}

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        distance_target = (anchor - target).pow(2).sum(1)  # .pow(.5)

        distance_negtotgt = (negative - target).pow(2).sum(1)  # .pow(.5)
        distance_tgttoneg = (target - negative).pow(2).sum(1)  # .pow(.5)

        loss1 = F.relu(distance_positive - distance_negative + self.margin)
        loss2 = F.relu(distance_positive - distance_target + self.margin)
        loss3 = F.relu(distance_negative - distance_positive - self.margin_factor * self.margin)
        loss4 = F.relu(distance_target - distance_positive - self.margin_factor * self.margin)
        loss5 = F.relu(-distance_negtotgt + self.margin)
        loss6 = F.relu(distance_tgttoneg - self.margin_factor * self.margin)


        quadruplet_loss = self.lamda[0] * loss1.mean() + \
                          self.lamda[1] * loss2.mean() + \
                          self.lamda[2] * loss3.mean() + \
                          self.lamda[3] * loss4.mean() + \
                          self.lamda[4] * loss5.mean() + \
                          self.lamda[5] * loss6.mean()

        losses_dict['loss1'] = self.lamda[0] * loss1.mean()
        losses_dict['loss2'] = self.lamda[1] * loss2.mean()
        losses_dict['loss3'] = self.lamda[2] * loss3.mean()
        losses_dict['loss4'] = self.lamda[3] * loss4.mean()
        losses_dict['loss5'] = self.lamda[4] * loss5.mean()
        losses_dict['loss6'] = self.lamda[5] * loss6.mean()

        return quadruplet_loss, losses_dict


class KMeanLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin1, margin2, lamda=0.1):
        super(KMeanLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.lamda = lamda
        self.loss_keys = ['loss1', 'loss2', 'loss3', 'loss4']

    def forward(self, anchor, positive, negative, target):

        losses_dict = {}

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        distance_target = (anchor - target).pow(2).sum(1)  # .pow(.5)

        loss1 = F.relu(distance_positive - distance_negative + self.margin1)
        loss2 = F.relu(distance_positive - distance_target + 2 * self.margin2)
        loss3 = F.relu(distance_negative - distance_positive - 2 * self.margin2)
        loss4 = F.relu(distance_target - distance_positive - 3 * self.margin2)

        quadruplet_loss = loss1.mean() + loss2.mean() + self.lamda * (loss3.mean() * loss4.mean())

        losses_dict['loss1'] = loss1.mean()
        losses_dict['loss2'] = loss2.mean()
        losses_dict['loss3'] = self.lamda * loss3.mean()
        losses_dict['loss4'] = self.lamda * loss4.mean()

        return quadruplet_loss, losses_dict
