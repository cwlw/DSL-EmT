import torch
import torch.nn as nn


class AdaptiveGateFusion(nn.Module):
    """
    Lightweight learnable rPSD + DE gate.
    Only trainable parameter:
        alpha: [32, 6]
    Fusion rule:
        F = alpha * F_rPSD + (1 - alpha) * F_DE
    No convolution, linear layer, concatenation, dropout, normalization,
    or additional regularization is introduced here.
    """
    def __init__(self, num_channels=32, num_bands=6, alpha_init=0.7):
        super().__init__()
        # [MOD] The sole learnable gate parameter is fixed to [32, 6] by design.
        self.alpha = nn.Parameter(
            torch.full((num_channels, num_bands), float(alpha_init), dtype=torch.float32)
        )

    def forward(self, rpsd, de):
        # [MOD] Keep the original 32‑channel x 6‑band shape and apply only
        # element‑wise Hadamard weighting.
        rpsd = rpsd.float()
        de = de.float()

        if rpsd.shape != de.shape:
            raise ValueError(
                "rPSD and DE shapes must match for gated fusion, "
                f"got {tuple(rpsd.shape)} vs {tuple(de.shape)}."
            )
        if rpsd.ndim < 2 or tuple(rpsd.shape[-2:]) != tuple(self.alpha.shape):
            raise ValueError(
                "Expected trailing EEG shape [32, 6], "
                f"got {tuple(rpsd.shape[-2:])}."
            )

        out = self.alpha * rpsd + (1.0 - self.alpha) * de
        return out.float()