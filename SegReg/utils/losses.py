import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss


from .utils import expand_as_one_hot


class ContrastiveLoss(nn.Module):
    """
    Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function'
    """

    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001):
        super(ContrastiveLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_cluster_means(self, input, target):
        embedding_dims = input.size()[1]
        # expand target: NxCxDxHxW -> NxCxExDxHxW
        # NxCx1xDxHxW
        target = target.unsqueeze(2)
        # save target's copy in order to compute the average embeddings later
        target_copy = target.clone()
        shape = list(target.size())
        shape[2] = embedding_dims
        target = target.expand(shape)

        # expand input: NxExDxHxW -> Nx1xExDxHxW
        input = input.unsqueeze(1)

        # sum embeddings in each instance (multiply first via broadcasting) output: NxCxEx1x1x1
        embeddings_per_instance = input * target
        num = torch.sum(embeddings_per_instance, dim=(3, 4, 5), keepdim=True)
        # get number of voxels in each cluster output: NxCx1x1x1x1
        num_voxels_per_instance = torch.sum(
            target_copy, dim=(3, 4, 5), keepdim=True)
        # compute mean embeddings NxCxEx1x1x1
        mean_embeddings = num / num_voxels_per_instance
        # return mean embeddings and additional tensors needed for further computations
        return mean_embeddings, embeddings_per_instance

    def _compute_variance_term(self, cluster_means, embeddings_per_instance, target):
        # compute the distance to cluster means, result:(NxCxDxHxW)
        embedding_norms = torch.norm(
            embeddings_per_instance - cluster_means, self.norm, dim=2)
        # get per instance distances (apply instance mask)
        embedding_norms = embedding_norms * target
        # zero out distances less than delta_var and sum to get the variance (NxC)
        embedding_variance = torch.clamp(
            embedding_norms - self.delta_var, min=0) ** 2
        embedding_variance = torch.sum(embedding_variance, dim=(2, 3, 4))
        # get number of voxels per instance (NxC)
        num_voxels_per_instance = torch.sum(target, dim=(2, 3, 4))
        # normalize the variance term
        C = target.size()[1]
        variance_term = torch.sum(
            embedding_variance / num_voxels_per_instance, dim=1) / C
        return variance_term

    def _compute_distance_term(self, cluster_means, C):
        if C == 1:
            # just one cluster in the batch, so distance term does not contribute to the loss
            return 0.
        # squeeze space dims
        for _ in range(3):
            cluster_means = cluster_means.squeeze(-1)
        # expand cluster_means tensor in order to compute the pair-wise distance between cluster means
        cluster_means = cluster_means.unsqueeze(1)
        shape = list(cluster_means.size())
        shape[1] = C
        # NxCxCxEx1x1x1
        cm_matrix1 = cluster_means.expand(shape)
        # transpose the cluster_means matrix in order to compute pair-wise distances
        cm_matrix2 = cm_matrix1.permute(0, 2, 1, 3)
        # compute pair-wise distances (NxCxC)
        dist_matrix = torch.norm(cm_matrix1 - cm_matrix2, p=self.norm, dim=3)
        # create matrix for the repulsion distance (i.e. cluster centers further apart than 2 * delta_dist
        # are not longer repulsed)
        repulsion_dist = 2 * self.delta_dist * (1 - torch.eye(C))
        # 1xCxC
        repulsion_dist = repulsion_dist.unsqueeze(0).to(cluster_means.device)
        # zero out distances grater than 2*delta_dist (NxCxC)
        hinged_dist = torch.clamp(repulsion_dist - dist_matrix, min=0) ** 2
        # sum all of the hinged pair-wise distances
        hinged_dist = torch.sum(hinged_dist, dim=(1, 2))
        # normalized by the number of paris and return
        return hinged_dist / (C * (C - 1))

    def _compute_regularizer_term(self, cluster_means, C):
        # squeeze space dims
        for _ in range(3):
            cluster_means = cluster_means.squeeze(-1)
        norms = torch.norm(cluster_means, p=self.norm, dim=2)
        assert norms.size()[1] == C
        # return the average norm per batch
        return torch.sum(norms, dim=1).div(C)

    def forward(self, input, target):
        """
        Args:
             input (torch.tensor): embeddings predicted by the network (NxExDxHxW) (E - embedding dims)
             target (torch.tensor): ground truth instance segmentation (NxDxHxW)

        Returns:
            Combined loss defined as: alpha * variance_term + beta * distance_term + gamma * regularization_term
        """
        # get number of instances in the batch
        C = torch.unique(target).size()[0]
        # expand each label as a one-hot vector: N x D x H x W -> N x C x D x H x W
        target = expand_as_one_hot(target, C)
        # compare spatial dimensions
        assert input.dim() == target.dim() == 5
        assert input.size()[2:] == target.size()[2:]

        # compute mean embeddings and assign embeddings to instances
        cluster_means, embeddings_per_instance = self._compute_cluster_means(
            input, target)
        variance_term = self._compute_variance_term(
            cluster_means, embeddings_per_instance, target)
        distance_term = self._compute_distance_term(cluster_means, C)
        regularization_term = self._compute_regularizer_term(cluster_means, C)
        # total loss
        loss = self.alpha * variance_term + self.beta * \
            distance_term + self.gamma * regularization_term
        # reduce batch dimension
        return torch.mean(loss)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    input = input.float()
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(
            1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        #target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super().__init__(weight, sigmoid_normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, sigmoid_normalization=True, epsilon=1e-6):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class New_WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(New_WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        loss = []
        for i in range(input.shape[1]):
            #weight = self._class_weights(input[:,i:i+1,:,:])
            loss.append(F.binary_cross_entropy(
                input[:, i:i+1, :, :], target[:, i:i+1, :, :]))
        #weight = self._class_weights(input)
        #target = expand_as_one_hot(target.long(), C=input.size()[1], ignore_index=self.ignore_index)
        return torch.stack(loss).mean()

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        #
        vinput = torch.zeros_like(input)
        # input=(input>0.5).long()
        input = torch.cat([vinput, input], 1)
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target.long(), C=input.size()[
                                   1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(
                input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class TagsAngularLoss(nn.Module):
    def __init__(self, tags_coefficients):
        super(TagsAngularLoss, self).__init__()
        self.tags_coefficients = tags_coefficients

    def forward(self, inputs, targets, weight):
        assert isinstance(inputs, list)
        # if there is just one output head the 'inputs' is going to be a singleton list [tensor]
        # and 'targets' is just going to be a tensor (that's how the HDF5Dataloader works)
        # so wrap targets in a list in this case
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets) == len(self.tags_coefficients)
        loss = 0
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients):
            loss += alpha * square_angular_loss(input, target, weight)

        return loss


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


def square_angular_loss(input, target, weights=None):
    """
    Computes square angular loss between input and target directions.
    Makes sure that the input and target directions are normalized so that torch.acos would not produce NaNs.

    :param input: 5D input tensor (NCDHW)
    :param target: 5D target tensor (NCDHW)
    :param weights: 3D weight tensor in order to balance different instance sizes
    :return: per pixel weighted sum of squared angular losses
    """
    assert input.size() == target.size()
    # normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
    stability_coeff = 0.999999
    input = input / \
        torch.norm(input, p=2, dim=1).detach().clamp(
            min=1e-8) * stability_coeff
    target = target / \
        torch.norm(target, p=2, dim=1).detach().clamp(
            min=1e-8) * stability_coeff
    # compute cosine map
    cosines = (input * target).sum(dim=1)
    error_radians = torch.acos(cosines)
    if weights is not None:
        return (error_radians * error_radians * weights).sum()
    else:
        return (error_radians * error_radians).sum()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config['device'])

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        # convert to cuda tensor if necessary
        pos_weight = torch.tensor(pos_weight).to(config['device'])

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(
            loss, loss_config.get('squeeze_channel', False))

    return loss


SUPPORTED_LOSSES = ['BCEWithLogitsLoss', 'BCEDiceLoss', 'CrossEntropyLoss', 'WeightedCrossEntropyLoss',
                    'PixelWiseCrossEntropyLoss', 'GeneralizedDiceLoss', 'DiceLoss', 'TagsAngularLoss', 'MSELoss',
                    'SmoothL1Loss', 'L1Loss', 'WeightedSmoothL1Loss']


def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alphs', 1.)
        beta = loss_config.get('beta', 1.)
        return BCEDiceLoss(alpha, beta)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        return GeneralizedDiceLoss(sigmoid_normalization=sigmoid_normalization)
    elif name == 'DiceLoss':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        return DiceLoss(weight=weight, sigmoid_normalization=sigmoid_normalization)
    elif name == 'TagsAngularLoss':
        tags_coefficients = loss_config['tags_coefficients']
        return TagsAngularLoss(tags_coefficients)
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'ContrastiveLoss':
        return ContrastiveLoss(loss_config['delta_var'], loss_config['delta_dist'], loss_config['norm'],
                               loss_config['alpha'], loss_config['beta'], loss_config['gamma'])
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss(threshold=loss_config['threshold'], initial_weight=loss_config['initial_weight'],
                                    apply_below_threshold=loss_config.get('apply_below_threshold', True))
    else:
        raise RuntimeError(
            f"Unsupported loss function: '{name}'. Supported losses: {SUPPORTED_LOSSES}")


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def ncc_loss(I, J, win=None):

    ndims = len(list(I.size())) - 2
    assert ndims in [
        1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).cuda()
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(
        I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return (1-torch.mean(cc))/2


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    #
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) *
                              (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] *
                          (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] -
                          (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(fild):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(fild)) - Get_Ja(fild))
    return torch.sum(Neg_Jac)


def vox_morph_loss(x, xnew, y, ytrue, field, lamda=0.01, usex=True):
    cc = ncc_loss(y, ytrue) * 0.1 + F.smooth_l1_loss(y, ytrue) * 10
    sm = gradient_loss(field) + NJ_loss(field)
    loss = cc + lamda * sm
    if usex:
        cc2 = ncc_loss(x, xnew)
        loss += cc2
    return loss
