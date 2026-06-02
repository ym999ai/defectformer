"""
Training losses for DefectFormer.

  HungarianMatcher    - bipartite matching between predictions and GT
  FocalLoss           - for classification (down-weights easy negatives)
  DefectFormerLoss    - full compound loss with auxiliary outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Pairwise cost helpers (no gradient needed — used inside matcher)
# ---------------------------------------------------------------------------

def _batch_dice_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred [N, L] sigmoid, gt [M, L] binary → [N, M] Dice-loss matrix."""
    numerator = 2.0 * torch.einsum("nl,ml->nm", pred, gt)
    denominator = pred.sum(-1)[:, None] + gt.sum(-1)[None, :]
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def _batch_bce_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred_logits [N, L], gt [M, L] binary → [N, M] mean-BCE matrix."""
    L = pred_logits.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        pred_logits, torch.ones_like(pred_logits), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        pred_logits, torch.zeros_like(pred_logits), reduction="none"
    )
    cost = torch.einsum("nl,ml->nm", pos, gt) + torch.einsum("nl,ml->nm", neg, 1.0 - gt)
    return cost / L


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Finds bipartite assignment minimising classification + mask costs."""

    def __init__(self, cost_cls: float = 1.0, cost_mask: float = 5.0,
                 cost_dice: float = 5.0):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, pred_logits: torch.Tensor, pred_mask_logits: torch.Tensor,
                targets: list) -> list:
        """
        pred_logits       : [B, Nq, C+1]
        pred_mask_logits  : [B, Nq, H*W]   (logits, before sigmoid)
        targets           : list of dicts with 'labels' [N_i] and 'masks' [N_i, H*W]

        Returns list of (pred_idx, tgt_idx) LongTensors, one per image.
        """
        B, Nq = pred_logits.shape[:2]
        device = pred_logits.device

        # Flatten for batch cost computation
        flat_logits = pred_logits.flatten(0, 1).float()         # [B*Nq, C+1]
        flat_mlogs  = pred_mask_logits.flatten(0, 1).float()    # [B*Nq, H*W]

        tgt_labels = torch.cat([t["labels"] for t in targets]).to(device)
        tgt_masks  = torch.cat([t["masks"]  for t in targets]).float().to(device)

        # Class cost: −P(c_j | pred_i)
        cost_cls = -flat_logits.softmax(-1)[:, tgt_labels]      # [B*Nq, N_gt_total]

        # Mask costs
        cost_bce  = _batch_bce_loss(flat_mlogs, tgt_masks)
        cost_dice = _batch_dice_loss(torch.sigmoid(flat_mlogs), tgt_masks)

        C = (self.cost_cls  * cost_cls  +
             self.cost_mask * cost_bce  +
             self.cost_dice * cost_dice)
        C = C.view(B, Nq, -1).cpu()

        indices = []
        offset = 0
        for b, tgt in enumerate(targets):
            n = len(tgt["labels"])
            if n == 0:
                indices.append((torch.zeros(0, dtype=torch.long),
                                 torch.zeros(0, dtype=torch.long)))
                continue
            c_b = C[b, :, offset: offset + n]
            ri, ci = linear_sum_assignment(c_b.numpy())
            indices.append((torch.as_tensor(ri, dtype=torch.long),
                             torch.as_tensor(ci, dtype=torch.long)))
            offset += n

        return indices


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, reduction="none")
        p  = torch.exp(-ce)
        return (self.alpha * (1.0 - p) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Full DefectFormer Loss
# ---------------------------------------------------------------------------

class DefectFormerLoss(nn.Module):
    """
    L_total = Σ_{matched} [λ_cls·L_cls + λ_mask·L_BCE + λ_dice·L_Dice]

    Auxiliary losses at each of the L decoder layers (unit weight).
    Matching is performed on the *last* layer's outputs.
    """

    NO_OBJ_WEIGHT = 0.1   # down-weight background class

    def __init__(self, num_classes: int,
                 lambda_cls:  float = 2.0,
                 lambda_mask: float = 5.0,
                 lambda_dice: float = 5.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls  = lambda_cls
        self.lambda_mask = lambda_mask
        self.lambda_dice = lambda_dice

        self.matcher   = HungarianMatcher()
        self.focal_cls = FocalLoss()

    # ------------------------------------------------------------------

    def _layer_loss(self, pred_logits: torch.Tensor,
                    pred_mask_logits: torch.Tensor,
                    targets: list,
                    indices: list) -> torch.Tensor:
        """Compute combined loss for one decoder layer given fixed matching."""
        device = pred_logits.device
        B, Nq, _ = pred_logits.shape

        total = torch.zeros(1, device=device)
        num_inst = max(sum(len(t["labels"]) for t in targets), 1)

        for b, (pi, ti) in enumerate(indices):
            pi, ti = pi.to(device), ti.to(device)
            tgt_labels = targets[b]["labels"].to(device)
            tgt_masks  = targets[b]["masks"].float().to(device)

            # ---- classification ----
            if len(pi) > 0:
                total = total + self.lambda_cls * self.focal_cls(
                    pred_logits[b, pi], tgt_labels[ti]
                )

            # no-object penalty for unmatched queries
            all_q = torch.arange(Nq, device=device)
            unmatched = all_q[~torch.isin(all_q, pi)]
            if len(unmatched) > 0:
                no_obj = torch.full((len(unmatched),), self.num_classes,
                                    dtype=torch.long, device=device)
                total = total + self.NO_OBJ_WEIGHT * self.lambda_cls * \
                    F.cross_entropy(pred_logits[b, unmatched], no_obj)

            # ---- mask losses (only for matched pairs) ----
            if len(pi) == 0:
                continue

            p_mask = pred_mask_logits[b, pi]       # [N, H*W]
            t_mask = tgt_masks[ti]                 # [N, H*W]

            bce = F.binary_cross_entropy_with_logits(p_mask, t_mask)
            total = total + self.lambda_mask * bce

            p_sig = torch.sigmoid(p_mask)
            num = 2.0 * (p_sig * t_mask).sum(-1)
            den = p_sig.sum(-1) + t_mask.sum(-1)
            dice = (1.0 - (num + 1.0) / (den + 1.0)).mean()
            total = total + self.lambda_dice * dice

        return total / num_inst

    # ------------------------------------------------------------------

    def forward(self, all_cls_logits: list, all_mask_logits: list,
                targets: list) -> torch.Tensor:
        """
        all_cls_logits  : list[L] of [B, Nq, C+1]
        all_mask_logits : list[L] of [B, Nq, H*W]
        targets         : list of dicts with 'labels' [N_i], 'masks' [N_i, H*W]
        """
        # Matching on last layer
        indices = self.matcher(all_cls_logits[-1], all_mask_logits[-1], targets)

        loss = torch.zeros(1, device=all_cls_logits[0].device)
        for cls_l, mask_l in zip(all_cls_logits, all_mask_logits):
            loss = loss + self._layer_loss(cls_l, mask_l, targets, indices)

        return loss.squeeze()
