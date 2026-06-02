"""
Query-Driven Defect Instance Decoder (QDID).

Components:
  DefectPriorInitializer  - saliency-based Top-K query seeding
  DMCA                    - Defect-Masked Cross-Attention
  DCSA                    - Defect Context Self-Attention (IoU repulsion)
  DecoderLayer            - DMCA → DCSA → FFN
  QDID                    - L-layer decoder with per-layer auxiliary outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Defect-Prior Query Initializer
# ---------------------------------------------------------------------------

class DefectPriorInitializer(nn.Module):
    """
    1. Saliency map  S = σ(‖E‖₂)
    2. Select Top-Nq positions
    3. MLP-project feature vectors at those positions → Q₀
    Initial attention mask M₀[i] is 1 only at position p_i (expanded to a
    small neighbourhood via dilated saliency threshold).
    """

    def __init__(self, d: int, nq: int):
        super().__init__()
        self.nq = nq
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )

    def forward(self, E: torch.Tensor):
        """
        E : [B, d, Hf, Wf]  (pixel-decoder output)
        Returns
          Q0  [B, Nq, d]
          M0  [B, Nq, L]   binary initial attention masks  (L = Hf*Wf)
        """
        B, d, Hf, Wf = E.shape
        L = Hf * Wf

        # Saliency
        S = torch.sigmoid(E.norm(p=2, dim=1))          # [B, Hf, Wf]
        S_flat = S.view(B, L)                            # [B, L]

        # Top-K positions
        _, topk = torch.topk(S_flat, self.nq, dim=1)    # [B, Nq]

        # Feature extraction
        E_flat = E.view(B, d, L).permute(0, 2, 1)       # [B, L, d]
        idx_exp = topk.unsqueeze(-1).expand(-1, -1, d)   # [B, Nq, d]
        Q0 = self.mlp(torch.gather(E_flat, 1, idx_exp)) # [B, Nq, d]

        # Initial mask: 1 at each query's seed position
        M0 = torch.zeros(B, self.nq, L, device=E.device)
        batch_idx = torch.arange(B, device=E.device).unsqueeze(1)  # [B, 1]
        query_idx = torch.arange(self.nq, device=E.device).unsqueeze(0)  # [1, Nq]
        M0[batch_idx, query_idx, topk] = 1.0

        return Q0, M0


# ---------------------------------------------------------------------------
# DMCA – Defect-Masked Cross-Attention
# ---------------------------------------------------------------------------

class DMCA(nn.Module):
    """
    Cross-attention where positions outside the predicted mask are blocked.

    q_i^(l) = softmax( q_i^(l-1) K^T/√d + M_i^(l-1) ) V
    M_i[j] = 0 if j inside predicted mask, −∞ otherwise
    """

    def __init__(self, d: int, num_heads: int = 8):
        super().__init__()
        assert d % num_heads == 0
        self.nh = num_heads
        self.hd = d // num_heads
        self.scale = self.hd ** -0.5

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self,
                queries: torch.Tensor,   # [B, Nq, d]
                E_flat: torch.Tensor,    # [B, L, d]
                mask: torch.Tensor,      # [B, Nq, L]  1=allow, 0=block
                ) -> torch.Tensor:
        B, Nq, d = queries.shape
        L = E_flat.shape[1]

        def reshape(t, seq):
            return t.view(B, seq, self.nh, self.hd).transpose(1, 2)  # [B, nh, seq, hd]

        Q = reshape(self.q_proj(queries), Nq)
        K = reshape(self.k_proj(E_flat), L)
        V = reshape(self.v_proj(E_flat), L)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # [B, nh, Nq, L]

        # Mask: add −∞ to blocked positions  (mask=1 → allow)
        attn_bias = (1.0 - mask.unsqueeze(1)) * (-1e9)              # [B, 1, Nq, L]
        attn = attn + attn_bias

        # Safety: if every position is masked for a query, allow all
        all_masked = (mask.sum(-1) == 0)                            # [B, Nq]
        if all_masked.any():
            attn_bias_safe = torch.zeros_like(attn)
            attn_bias_safe[all_masked.unsqueeze(1).expand_as(attn_bias_safe)] = 0
            attn[all_masked.unsqueeze(1).unsqueeze(-1).expand_as(attn)] = \
                (torch.matmul(Q, K.transpose(-2, -1)) * self.scale
                 )[all_masked.unsqueeze(1).unsqueeze(-1).expand_as(attn)]

        out = torch.matmul(F.softmax(attn, dim=-1), V)              # [B, nh, Nq, hd]
        out = out.transpose(1, 2).contiguous().view(B, Nq, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# DCSA – Defect Context Self-Attention
# ---------------------------------------------------------------------------

class DCSA(nn.Module):
    """
    Self-attention with IoU-based spatial repulsion:

    Repulsion(i, j) = −γ · IoU(m̂_i^(l-1), m̂_j^(l-1))

    This prevents two queries from attending the same defect region.
    """

    def __init__(self, d: int, num_heads: int = 8, gamma: float = 0.5):
        super().__init__()
        assert d % num_heads == 0
        self.nh = num_heads
        self.hd = d // num_heads
        self.scale = self.hd ** -0.5
        self.gamma = gamma

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    @staticmethod
    def _iou(masks: torch.Tensor) -> torch.Tensor:
        """masks [B, Nq, L] soft → pairwise IoU [B, Nq, Nq]."""
        m = (masks > 0.5).float()
        inter = torch.bmm(m, m.transpose(1, 2))
        areas = m.sum(-1)                                            # [B, Nq]
        union = areas.unsqueeze(2) + areas.unsqueeze(1) - inter
        return inter / (union + 1e-6)

    def forward(self,
                queries: torch.Tensor,      # [B, Nq, d]
                prev_masks: torch.Tensor,   # [B, Nq, L] sigmoid values
                ) -> torch.Tensor:
        B, Nq, d = queries.shape

        def reshape(t):
            return t.view(B, Nq, self.nh, self.hd).transpose(1, 2)

        Q = reshape(self.q_proj(queries))
        K = reshape(self.k_proj(queries))
        V = reshape(self.v_proj(queries))

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale    # [B, nh, Nq, Nq]

        iou = self._iou(prev_masks)                                  # [B, Nq, Nq]
        repulsion = (-self.gamma * iou).unsqueeze(1)                 # [B, 1, Nq, Nq]
        attn = attn + repulsion

        out = torch.matmul(F.softmax(attn, dim=-1), V)               # [B, nh, Nq, hd]
        out = out.transpose(1, 2).contiguous().view(B, Nq, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Single decoder layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """DMCA → DCSA → FFN, each with pre-LN residual."""

    def __init__(self, d: int, num_heads: int = 8, ffn_ratio: int = 4):
        super().__init__()
        self.dmca = DMCA(d, num_heads)
        self.dcsa = DCSA(d, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(d * ffn_ratio, d),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self,
                queries: torch.Tensor,      # [B, Nq, d]
                E_flat: torch.Tensor,       # [B, L, d]
                attn_mask: torch.Tensor,    # [B, Nq, L]  binary
                prev_masks: torch.Tensor,   # [B, Nq, L]  sigmoid
                ) -> torch.Tensor:
        q = queries + self.dmca(self.norm1(queries), E_flat, attn_mask)
        q = q + self.dcsa(self.norm2(q), prev_masks)
        q = q + self.ffn(self.norm3(q))
        return q


# ---------------------------------------------------------------------------
# QDID
# ---------------------------------------------------------------------------

class QDID(nn.Module):
    """
    Query-Driven Defect Instance Decoder.

    Returns per-layer (class logits, mask logits) for auxiliary loss.
    Final-layer outputs are used for inference.
    """

    def __init__(self, d: int = 256, nq: int = 100,
                 num_layers: int = 6, num_classes: int = 15,
                 num_heads: int = 8):
        super().__init__()
        self.nq = nq
        self.d = d

        self.initializer = DefectPriorInitializer(d, nq)
        self.layers = nn.ModuleList(
            [DecoderLayer(d, num_heads) for _ in range(num_layers)]
        )

        # Prediction heads (shared across layers)
        self.class_head = nn.Linear(d, num_classes + 1)   # +1 = no-object
        self.mask_embed = nn.Linear(d, d)                  # projects query for mask dot-product

    def _predict_mask_logits(self, queries: torch.Tensor,
                             E_flat: torch.Tensor) -> torch.Tensor:
        """
        m̂_i = σ( q_i^(L) · E^T / √d )   — returned as logits (before σ)
        queries : [B, Nq, d]
        E_flat  : [B, L, d]
        returns : [B, Nq, L]  logits
        """
        q = self.mask_embed(queries)                               # [B, Nq, d]
        logits = torch.bmm(q, E_flat.transpose(1, 2)) * (self.d ** -0.5)
        return logits                                              # [B, Nq, L]

    def forward(self, E: torch.Tensor):
        """
        E : [B, d, Hf, Wf]
        Returns
          all_cls_logits : list[L] of [B, Nq, C+1]
          all_mask_logits: list[L] of [B, Nq, Hf*Wf]
        """
        B, d, Hf, Wf = E.shape
        L = Hf * Wf
        E_flat = E.view(B, d, L).permute(0, 2, 1)                 # [B, L, d]

        # Defect-prior initialisation
        queries, M0 = self.initializer(E)                          # Q0 [B,Nq,d], M0 [B,Nq,L]
        current_masks_sigmoid = M0                                 # initial soft mask

        all_cls_logits = []
        all_mask_logits = []

        for layer in self.layers:
            attn_mask = (current_masks_sigmoid > 0.5).float()     # binarise
            queries = layer(queries, E_flat, attn_mask, current_masks_sigmoid)

            # Predict mask logits for this layer
            mask_logits = self._predict_mask_logits(queries, E_flat)  # [B, Nq, L]
            current_masks_sigmoid = torch.sigmoid(mask_logits)

            all_mask_logits.append(mask_logits)
            all_cls_logits.append(self.class_head(queries))

        return all_cls_logits, all_mask_logits
