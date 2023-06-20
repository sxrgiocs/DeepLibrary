import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCoefficient(nn.Module):
    """
    Dice coefficient for either binary or multiclass segmentation tasks.
    It uses the original formulation D = 2*|X, Y| / (|X|+|Y|) with X being the
    prediction probabilities and Y being the ground truth/target probabilities.

    Parameters
    ----------
    num_classes : int
        Number of classes of the problem.
    epsilon : float, optional
        Constant used to prevent division by zero and to improve stability.
        Defaults to 1e-5
    classwise : bool, optional
        Boolean parameter used to choose whether or not to return the DICE loss
        per class instead of the averaged loss, which can be useful to analyze
        per-class performance. Defaults to False.
    """

    def __init__(self, num_classes, epsilon=1e-5, classwise=False):
        super(DiceCoefficient, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.classwise = classwise

    def forward(self, inputs, targets, logits=False):
        """
        Compute the Dice coefficient.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted inputs, which can be logits or probabilities.
        targets : torch.Tensor
            Target labels.
        logits : bool, optional
            Determine whether or not the inputs are logits of probabilities. If
            true, the inputs are converted to probabilities. Defaults to False.

        Returns
        -------
        dice : torch.Tensor
            Dice coefficient.
        """

        # Convert the inputs to probabilities if they are logits
        if logits:
            # Use sigmoid if the problem is binary
            if self.num_classes == 2:
                probs = F.sigmoid(inputs)
            # Use softmax if the problem is multiclass
            else:
                probs = F.softmax(inputs, dim=1)

        else:
            probs = inputs

        # Reshape probs and targets to [batch_size, num_classes, -1]
        probs = probs.view(probs.size(0), self.num_classes, -1)
        targets = targets.view(targets.size(0), self.num_classes, -1)

        # Compute the intersection between samples
        intersection = torch.sum(probs * targets, dim=2)

        # Compute numerator and denominator for Dice loss
        numerator = 2 * intersection
        denominator = torch.sum(probs, dim=2) + torch.sum(targets, dim=2)

        # Compute Dice loss per class
        dice_per_class = (numerator + self.epsilon) / (denominator + self.epsilon)

        # Choose whether or not to return the loss per class
        if self.classwise:
            dice_coef = torch.mean(dice_per_class, dim=0)
        else:
            dice_coef = torch.mean(dice_per_class)

        return dice_coef
