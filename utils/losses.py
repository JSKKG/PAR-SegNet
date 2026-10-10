"""Loss functions used for PAR-SegNet training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyDC(nn.Module):
    """Foreground Dice loss for PS and FH segmentation."""

    def __init__(self):
        super(MyDC, self).__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        # Exclude the background channel.
        y_pred_fg = y_pred[:, 1:, :, :]
        y_truth_fg = y_truth[:, 1:, :, :]

        intersection = (y_pred_fg * y_truth_fg).sum()
        union = y_pred_fg.sum() + y_truth_fg.sum()

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1.0 - dice_score


class DCloss(nn.Module):
    """Differentiable foreground Dice loss for three-class segmentation."""

    def __init__(self):
        super(DCloss, self).__init__()
        self.dc = MyDC()

    def forward(self, net_output, target):
        """
        Args:
            net_output: Logits with shape [B, 3, H, W].
            target: Class-index mask with shape [B, H, W].
        """
        net_output = F.softmax(net_output, dim=1)

        target = F.one_hot(
            target.long(),
            num_classes=3
        ).permute(0, 3, 1, 2).float()

        return self.dc(net_output, target)


class DiceLoss(nn.Module):
    """Multi-class Dice loss."""

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []

        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))

        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        dice_score = (
            2.0 * intersect + smooth
        ) / (
            z_sum + y_sum + smooth
        )

        return 1.0 - dice_score

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), (
            f"Prediction shape {inputs.size()} and "
            f"target shape {target.size()} do not match."
        )

        loss = 0.0

        for i in range(self.n_classes):
            loss += self._dice_loss(
                inputs[:, i],
                target[:, i]
            ) * weight[i]

        return loss / self.n_classes
