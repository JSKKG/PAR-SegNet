"""PARM: pairwise anatomical relation modeling module."""

import torch
import torch.nn as nn


class PARM(nn.Module):
    def __init__(self, channels, reduction=4):
        super(PARM, self).__init__()

        reduced_channels = channels // reduction

        # PS-oriented representation
        self.ps_proj = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )

        # FH-oriented representation
        self.fh_proj = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )

        # Encode pairwise differences, correlations, and joint features
        self.relation = nn.Sequential(
            nn.Conv2d(
                reduced_channels * 3,
                reduced_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                reduced_channels,
                reduced_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )

        # PS-directed and FH-directed relation gates
        self.ps_gate = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.fh_gate = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Reconstruct refined PS and FH features
        self.ps_refine = nn.Sequential(
            nn.Conv2d(
                reduced_channels * 2,
                reduced_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )
        self.fh_refine = nn.Sequential(
            nn.Conv2d(
                reduced_channels * 2,
                reduced_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )

        # Fuse refined PS/FH features and their shared relation
        self.out_proj = nn.Sequential(
            nn.Conv2d(
                reduced_channels * 3,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual scaling (initialized to zero, as in the original code)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Step 1: PS- and FH-oriented representations
        f_ps = self.ps_proj(x)
        f_fh = self.fh_proj(x)

        # Step 2: pairwise relation construction
        difference = torch.abs(f_ps - f_fh)
        correlation = f_ps * f_fh
        joint = f_ps + f_fh
        relation_input = torch.cat([difference, correlation, joint], dim=1)

        # Step 3: explicit pairwise relation representation
        relation = self.relation(relation_input)

        # Step 4: asymmetric relation gates
        gate_ps = self.ps_gate(relation)
        gate_fh = self.fh_gate(relation)

        # Step 5: PS refinement
        ps_relation = gate_ps * relation
        f_ps_refined = self.ps_refine(torch.cat([f_ps, ps_relation], dim=1))
        f_ps_refined = f_ps + f_ps_refined

        # Step 6: FH refinement
        fh_relation = gate_fh * relation
        f_fh_refined = self.fh_refine(torch.cat([f_fh, fh_relation], dim=1))
        f_fh_refined = f_fh + f_fh_refined

        # Step 7: joint reconstruction
        fused = self.out_proj(
            torch.cat([f_ps_refined, f_fh_refined, relation], dim=1)
        )

        # Step 8: residual output
        return x + self.gamma * fused
