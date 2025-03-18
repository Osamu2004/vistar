
import torch
from apps.utils.dist import sync_tensor



class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, is_distributed=False):
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: torch.Tensor or int or float) -> torch.Tensor or int or float:
        return sync_tensor(val, reduce="sum") if self.is_distributed else val

    def update(self, val: torch.Tensor or int or float, delta_n=1):
        self.count += self._sync(delta_n)
        self.sum += self._sync(val * delta_n)

    def get_count(self) -> torch.Tensor or int or float:
        return self.count.item() if isinstance(self.count, torch.Tensor) and self.count.numel() == 1 else self.count

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg

import torch
import torch.nn.functional as F

# Helper functions
def _take_channels(*xs, ignore_channels=None):
    """
    Select specific channels, ignoring others, and support negative indexing.

    Args:
        xs: Input tensors to process.
        ignore_channels (list[int], optional): List of channels to ignore. Negative indices are allowed.

    Returns:
        Processed tensors with ignored channels removed.
    """
    if ignore_channels is None:
        return xs
    else:
        num_channels = xs[0].shape[1]  # Number of channels in the input tensors
        # Convert negative indices to positive
        ignore_channels = [(ch + num_channels) if ch < 0 else ch for ch in ignore_channels]
        # Generate the list of channels to keep
        channels = [
            channel
            for channel in range(num_channels)
            if channel not in ignore_channels
        ]
        # Select the desired channels
        xs = [
            torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device))
            for x in xs
        ]
        return xs


def _threshold(x, threshold=None):
    """
    Apply a threshold to binarize predictions.

    Args:
        x (torch.Tensor): Input tensor.
        threshold (float, optional): Threshold for binarization.

    Returns:
        Binarized tensor.
    """
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


# Prepare inputs for multiclass metrics
def prepare_multiclass_inputs(predictions, labels, ignore_channels=None, apply_argmax=True):
    """
    Prepare predictions and labels for multiclass metrics calculation.

    Args:
        predictions (torch.Tensor): Predicted logits or probabilities, shape (N, C, H, W).
        labels (torch.Tensor): Ground truth labels, shape (N, H, W).
        ignore_channels (list[int], optional): Channels to ignore during computation.
        apply_argmax (bool, optional): Whether to apply argmax to predictions. Defaults to True.

    Returns:
        predictions, labels: Processed predictions and labels, both with shape (N, H*W) or (N, C, H*W).
    """
    bs, num_classes, h, w = predictions.shape

    # Apply argmax to predictions if needed
    if apply_argmax:
        predictions = predictions.argmax(dim=1)  # Shape: (N, H, W)

    # Reshape labels and predictions
    labels = labels.view(bs, -1)  # Shape: (N, H*W)
    if apply_argmax:
        predictions = predictions.view(bs, -1)  # Shape: (N, H*W)
    else:
        predictions = predictions.view(bs, num_classes, -1)  # Shape: (N, C, H*W)

    # If not using argmax, we may still need to one-hot encode labels
    if apply_argmax:
        predictions = F.one_hot(predictions, num_classes=num_classes)
        predictions = predictions.permute(0, 2, 1)
        labels = F.one_hot(labels, num_classes=num_classes)  # Shape: (N, H*W, C)
        labels = labels.permute(0, 2, 1)  # Shape: (N, C, H*W)

    # Apply ignore_channels if specified
    if ignore_channels is not None:
        predictions, labels = _take_channels(predictions, labels, ignore_channels=ignore_channels)

    return predictions, labels


# Metric functions
def iou(pr, gt, eps=1e-7,threshold=0.5,ignore_channels=[-1]):
    """Calculate Intersection over Union."""
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    intersection = torch.sum(gt * pr, dim=(0, 2))
    union = torch.sum(gt, dim=(0, 2)) + torch.sum(pr, dim=(0, 2)) - intersection
    return (intersection + eps) / (union + eps)


def f_score(pr, gt, beta=1, eps=1e-7,threshold=0.5,ignore_channels=[-1]):
    """Calculate F1-Score."""
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    tp = torch.sum(gt * pr, dim=(0, 2))
    fp = torch.sum(pr, dim=(0, 2)) - tp
    fn = torch.sum(gt, dim=(0, 2)) - tp
    return ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)


def precision(pr, gt, eps=1e-7,threshold=0.5,ignore_channels=[-1]):
    """Calculate Precision."""
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    tp = torch.sum(gt * pr, dim=(0, 2))
    fp = torch.sum(pr, dim=(0, 2)) - tp
    return (tp + eps) / (tp + fp + eps)


def recall(pr, gt, eps=1e-7,threshold=0.5,ignore_channels=[-1]):
    """Calculate Recall."""
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    tp = torch.sum(gt * pr, dim=(0, 2))
    fn = torch.sum(gt, dim=(0, 2)) - tp
    return (tp + eps) / (tp + fn + eps)


# Main metric calculation functions
def calculate_iou(predictions: torch.Tensor, labels: torch.Tensor, return_type="avg", ignore_channels=[0]) -> torch.Tensor:
    """
    Calculate IoU for multiclass predictions and labels.

    Args:
        predictions (torch.Tensor): Predicted logits or probabilities, shape (N, C, H, W).
        labels (torch.Tensor): Ground truth labels, shape (N, H, W).
        return_type (str, optional): If "avg", return the average IoU. If None, return IoU for each class.
        ignore_channels (list[int], optional): Channels to ignore during computation.

    Returns:
        torch.Tensor: IoU values.
    """
    predictions = F.softmax(predictions, dim=1)
    predictions, labels = prepare_multiclass_inputs(predictions, labels)
    iou_values = iou(predictions, labels,ignore_channels=ignore_channels)
    return iou_values.mean() if return_type == "avg" else iou_values


def calculate_f1_score(predictions: torch.Tensor, labels: torch.Tensor, beta: float = 1, return_type="avg", ignore_channels=[0]) -> torch.Tensor:
    """
    Calculate F1-Score for multiclass predictions and labels.

    Args:
        predictions (torch.Tensor): Predicted logits or probabilities, shape (N, C, H, W).
        labels (torch.Tensor): Ground truth labels, shape (N, H, W).
        beta (float): Weighting factor for precision and recall.
        return_type (str, optional): If "avg", return the average F1-Score. If None, return F1-Score for each class.
        ignore_channels (list[int], optional): Channels to ignore during computation.

    Returns:
        torch.Tensor: F1-Score values.
    """
    predictions = F.softmax(predictions, dim=1)
    predictions, labels = prepare_multiclass_inputs(predictions, labels)
    f1_values = f_score(predictions, labels, beta=beta,ignore_channels=ignore_channels)
    return f1_values.mean() if return_type == "avg" else f1_values


def calculate_precision(predictions: torch.Tensor, labels: torch.Tensor, return_type="avg", ignore_channels=[0]) -> torch.Tensor:
    """
    Calculate Precision for multiclass predictions and labels.
    """
    predictions = F.softmax(predictions, dim=1)
    predictions, labels = prepare_multiclass_inputs(predictions, labels)
    precision_values = precision(predictions, labels,ignore_channels=ignore_channels)
    return precision_values.mean() if return_type == "avg" else precision_values


def calculate_recall(predictions: torch.Tensor, labels: torch.Tensor, return_type="avg", ignore_channels=[0]) -> torch.Tensor:
    """
    Calculate Recall for multiclass predictions and labels.
    """
    predictions = F.softmax(predictions, dim=1)
    predictions, labels = prepare_multiclass_inputs(predictions, labels)
    recall_values = recall(predictions, labels,ignore_channels=ignore_channels)
    return recall_values.mean() if return_type == "avg" else recall_values



