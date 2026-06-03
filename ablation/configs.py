"""
Ablation experiment configurations — mirrors Table III of the paper.

Each ExperimentConfig maps directly to one row of the ablation table.
Configs are ordered to match the cumulative ablation sequence.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    name:               str             # machine-readable key (used as dir name)
    label:              str             # display label for the results table
    backbone:           str  = "swin_tiny"  # "swin_tiny" | "resnet50"
    msfe_variant:       str  = "full"       # "full" | "csag_only" | "additive" | "none"
    use_prior_init:     bool = True
    use_dmca_mask:      bool = True
    use_dcsa_repulsion: bool = True
    use_synthetic_aug:  bool = True
    # Expected values from Table III (paper) for quick sanity-checking
    paper_ap:           Optional[float] = None
    paper_ap_s:         Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# All 10 ablation rows  (Table III)
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_EXPERIMENTS: list[ExperimentConfig] = [

    ExperimentConfig(
        name="baseline",
        label="Baseline (Mask2Former + ResNet-50)",
        backbone="resnet50",
        msfe_variant="none",
        use_prior_init=False,
        use_dmca_mask=False,
        use_dcsa_repulsion=False,
        use_synthetic_aug=False,
        paper_ap=38.1, paper_ap_s=14.2,
    ),

    ExperimentConfig(
        name="swin_backbone",
        label="+ Swin-T backbone",
        backbone="swin_tiny",
        msfe_variant="none",
        use_prior_init=False,
        use_dmca_mask=False,
        use_dcsa_repulsion=False,
        use_synthetic_aug=False,
        paper_ap=41.3, paper_ap_s=16.8,
    ),

    ExperimentConfig(
        name="msfe_full",
        label="+ MSFE module",
        backbone="swin_tiny",
        msfe_variant="full",
        use_prior_init=False,
        use_dmca_mask=False,
        use_dcsa_repulsion=False,
        use_synthetic_aug=False,
        paper_ap=45.2, paper_ap_s=22.6,
    ),

    ExperimentConfig(
        name="msfe_additive",
        label="  ↳ CSAG → additive (FPN)",
        backbone="swin_tiny",
        msfe_variant="additive",
        use_prior_init=False,
        use_dmca_mask=False,
        use_dcsa_repulsion=False,
        use_synthetic_aug=False,
        paper_ap=42.8, paper_ap_s=17.9,
    ),

    ExperimentConfig(
        name="msfe_no_ltec",
        label="  ↳ LTEC removed",
        backbone="swin_tiny",
        msfe_variant="csag_only",
        use_prior_init=False,
        use_dmca_mask=False,
        use_dcsa_repulsion=False,
        use_synthetic_aug=False,
        paper_ap=44.0, paper_ap_s=20.1,
    ),

    ExperimentConfig(
        name="qdid_full",
        label="+ QDID (DMCA + prior init)",
        backbone="swin_tiny",
        msfe_variant="full",
        use_prior_init=True,
        use_dmca_mask=True,
        use_dcsa_repulsion=False,
        use_synthetic_aug=False,
        paper_ap=47.4, paper_ap_s=24.3,
    ),

    ExperimentConfig(
        name="qdid_random_init",
        label="  ↳ random init",
        backbone="swin_tiny",
        msfe_variant="full",
        use_prior_init=False,
        use_dmca_mask=True,
        use_dcsa_repulsion=False,
        use_synthetic_aug=False,
        paper_ap=45.8, paper_ap_s=22.9,
    ),

    ExperimentConfig(
        name="qdid_full_attn",
        label="  ↳ full cross-attn (no DMCA mask)",
        backbone="swin_tiny",
        msfe_variant="full",
        use_prior_init=True,
        use_dmca_mask=False,
        use_dcsa_repulsion=False,
        use_synthetic_aug=False,
        paper_ap=44.1, paper_ap_s=21.4,
    ),

    ExperimentConfig(
        name="dcsa",
        label="+ DCSA spatial exclusivity",
        backbone="swin_tiny",
        msfe_variant="full",
        use_prior_init=True,
        use_dmca_mask=True,
        use_dcsa_repulsion=True,
        use_synthetic_aug=False,
        paper_ap=48.0, paper_ap_s=24.7,
    ),

    ExperimentConfig(
        name="full",
        label="+ Synthetic augmentation (DefectFormer)",
        backbone="swin_tiny",
        msfe_variant="full",
        use_prior_init=True,
        use_dmca_mask=True,
        use_dcsa_repulsion=True,
        use_synthetic_aug=True,
        paper_ap=48.6, paper_ap_s=25.1,
    ),
]

# Convenience dict for lookup by name
EXPERIMENT_MAP: dict[str, ExperimentConfig] = {e.name: e for e in ABLATION_EXPERIMENTS}
