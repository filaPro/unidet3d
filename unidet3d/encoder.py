import torch
import torch.nn as nn
import itertools

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for i in range(len(x)):
            z, _ = self.attn(x[i], x[i], x[i])
            z = self.dropout(z) + x[i]
            z = self.norm(z)
            out.append(z)
        return out

class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out

class PredBBox(nn.Module):
    """Prediction module for bounding boxes.

    Args:
        d_model (int): Number of channels for model layers.
        n_bbox_outs (int): The number of outputs for 
            bounding box parameters.
        bbox_init_normal (bool, optional): If True, 
            initializes the linear layer weights using 
            a normal distribution. Defaults to False.
    """
    def __init__(self, d_model, n_bbox_outs, bbox_init_normal=False):
        super(PredBBox, self).__init__()
        self.linear = nn.Linear(d_model, n_bbox_outs)
        if bbox_init_normal:
            nn.init.normal_(self.linear.weight, std=.01)

    def forward(self, x):
        """Forward pass to predict bounding boxes.

        Args:
            x (Tensor): Input tensor of shape (n_bboxes, d_model).

        Returns:
            Tensor: A tensor of shape (n_bboxes, n_bbox_outs) 
                containing the predicted bounding boxes.
        """
        x = self.linear(x)
        return torch.hstack((torch.exp(x[:, :6]), 
                             x[:, 6:]))

@MODELS.register_module()
class UniDet3DEncoder(BaseModule):
    """Encoder for the UniDet3D model.

    Args:
        num_layers (int): Number of transformer layers.
        datasets_classes (List[List[str]]): List of classes for 
            each dataset.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        datasets (List[str]): List of dataset names.
        angles (List[Bool]): Whether to use angle prediction.
    """

    def __init__(self, num_layers, datasets_classes, in_channels, 
                 d_model, num_heads, hidden_dim, dropout, activation_fn,
                 datasets, angles, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.datasets = datasets
        self.angles = angles 
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.outs_cls = nn.ModuleList([])

        unique_cls = sorted(list(set(itertools.chain.from_iterable(
                            datasets_classes)))) + ['no_obj']
        self.outs_cls = nn.Sequential(
                                nn.Linear(d_model, d_model), nn.ReLU(),
                                nn.Linear(d_model, len(unique_cls)))
        self.datasets_cls_idxs = []
        for dataset_classes in datasets_classes:
            dataset_cls_idxs = []
            for cls in dataset_classes:
                dataset_cls_idxs.append(unique_cls.index(cls))
            self.datasets_cls_idxs.append(dataset_cls_idxs + [-1])

        self.out_bboxes = PredBBox(d_model, 8)

    def _forward_head(self, feats, sp_centers, datasets_names):
        """Prediction head forward.

        Args:
            feats (List[Tensor]): of len batch_size,
                each of shape (n_bboxes_i, d_model).
            sp_centers (List[Tensor]): of len batch_size,
                each of shape (n_bboxes_i, 3) representing 
                the spatial centers for the predicted 
                bounding boxes.
            datasets_names (List[str]): A list of dataset names 
                corresponding to each input feature tensor.

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                    - List[Tensor]: Classification predictions 
                        of length `batch_size`, with each tensor 
                        having a shape of (n_bboxes_i, n_classes + 1).
                    - List[Tensor]: Bounding box predictions of 
                        length `batch_size`, with each tensor having 
                        a shape of (n_bboxes_i, 7) (or 6 if `angles` 
                        are not applicable).
        """
        cls_preds, pred_bboxes = [], []
        for i in range(len(feats)):
            norm_query = self.out_norm(feats[i])
            idx = self.datasets.index(datasets_names[i])
            class_idxs = norm_query.new_tensor(
                            self.datasets_cls_idxs[idx]).long()
            cls_preds.append(self.outs_cls(norm_query)[:, class_idxs])
            pred_bbox = self.out_bboxes(norm_query)
            if not self.angles[idx]:
                pred_bbox = pred_bbox[:, :6]
            pred_bbox = _bbox_pred_to_bbox(sp_centers[i], pred_bbox)
            pred_bboxes.append(pred_bbox)

        return cls_preds, pred_bboxes

    def forward(self, x, sp_centers, datasets_names):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_bboxes_i, in_channels).
            sp_centers (List[Tensor]): A list of spatial centers 
                for the superpoints, with a length of `batch_size`. 
                Each tensor has a shape of (n_bboxes_i, 3).
            datasets_names (List[str]): A list of dataset names 
                corresponding to each input feature tensor.
        Returns:
            Dict: with cls_preds, pred_bboxes and aux_outputs.
        """
        cls_preds, pred_bboxes = [], []
        feats = [self.input_proj(y) for y in x]
        cls_pred, pred_bbox = \
            self._forward_head(feats, sp_centers, datasets_names)
        cls_preds.append(cls_pred)
        pred_bboxes.append(pred_bbox)
        for i in range(self.num_layers):
            feats = self.self_attn_layers[i](feats)
            feats = self.ffn_layers[i](feats)
            cls_pred, pred_bbox = \
                self._forward_head(feats, sp_centers, datasets_names)
            cls_preds.append(cls_pred)
            pred_bboxes.append(pred_bbox)
        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                bboxes=bboxes)
            for cls_pred, bboxes in zip(
                cls_preds[:-1], pred_bboxes[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            bboxes=pred_bboxes[-1],
            aux_outputs=aux_outputs)

def _bbox_pred_to_bbox(points, bbox_pred):
    """Transform predicted bbox parameters to bbox.

    Args:
        points (Tensor): Final locations of shape (N, 3)
        bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
            or (N, 8).

    Returns:
        Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
    """
    if bbox_pred.shape[0] == 0:
        return bbox_pred

    x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
    y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
    z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

    # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
    base_bbox = torch.stack([
        x_center,
        y_center,
        z_center,
        bbox_pred[:, 0] + bbox_pred[:, 1],
        bbox_pred[:, 2] + bbox_pred[:, 3],
        bbox_pred[:, 4] + bbox_pred[:, 5],
    ], -1)

    # axis-aligned case
    if bbox_pred.shape[1] == 6:
        return base_bbox

    # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
    scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
        bbox_pred[:, 2] + bbox_pred[:, 3]
    q = torch.exp(
        torch.sqrt(
            torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
    alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
    return torch.stack(
        (x_center, y_center, z_center, scale / (1 + q), scale /
            (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
        dim=-1)
