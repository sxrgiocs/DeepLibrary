import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for either binary or multiclass segmentation tasks.

    It uses the original formulation 1 - D with D = 2*|X, Y| / (|X|+|Y|) with X
    being the prediction probabilities and Y being the ground truth/target
    probabilities.

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

    Attributes
    ----------
    Same as the Parameters.

    Methods
    -------
    forward(inputs, targets, logits=False)
        Compute the Dice loss.
    """

    def __init__(self, num_classes, epsilon=1e-5, classwise=False):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.classwise = classwise

    def forward(self, inputs, targets, logits=False):
        """
        The forward method computes the Dice loss given the predicted inputs
        and target labels. If the inputs are logits, they are converted to
        probabilities using sigmoid for binary problems and softmax for
        multiclass problems. The loss is then computed per class and averaged
        across all classes and batches.

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
        dice_loss : torch.Tensor
            Dice loss.
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

        # Average Dice loss across all classes and batches
        dice_loss_per_class = 1 - dice_per_class

        # Choose whether or not to return the loss per class
        dice_loss = torch.mean(dice_loss_per_class,
                               dim=0) if self.classwise else torch.mean(dice_loss_per_class)

        return dice_loss
