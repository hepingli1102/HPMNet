from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        "HPMNet requires the Mamba implementation provided by mamba_ssm."
    ) from exc


class ChannelCorrelationAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_depthwise = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            padding=1,
            groups=dim * 3,
            bias=False,
        )
        self.projection = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, token_count, channels = tokens.shape
        if token_count != height * width:
            raise ValueError("token count does not match the spatial dimensions")

        feature_map = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        query, key, value = self.qkv_depthwise(self.qkv(feature_map)).chunk(3, dim=1)
        query = query.reshape(batch, self.num_heads, self.head_dim, token_count)
        key = key.reshape(batch, self.num_heads, self.head_dim, token_count)
        value = value.reshape(batch, self.num_heads, self.head_dim, token_count)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = attention @ value
        output = output.reshape(batch, channels, height, width)
        output = self.projection(output)
        return output.flatten(2).transpose(1, 2)


class STME(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(dim)
        self.spatial_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.phenological_correlation = ChannelCorrelationAttention(
            dim=dim,
            num_heads=num_heads,
        )
        self.dynamic_gate = nn.Linear(dim * 2, dim)
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature_map.shape
        shortcut = feature_map.flatten(2).transpose(1, 2)
        normalized = self.normalization(shortcut)

        spatial = self.spatial_mamba(normalized)
        phenological = self.phenological_correlation(normalized, height, width)
        gate = torch.sigmoid(self.dynamic_gate(torch.cat([spatial, phenological], dim=-1)))
        mixed = spatial * gate + phenological * (1.0 - gate)
        output = shortcut + self.output_projection(mixed)
        return output.transpose(1, 2).reshape(batch, channels, height, width)


class HPE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_prototypes: int = 64,
        pooling_factor: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.pooling_factor = pooling_factor
        self.key_projection = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_projection = nn.Conv2d(dim, dim, kernel_size=1)
        self.prototypes = nn.Parameter(torch.empty(1, num_prototypes, dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.hyperedge_transformer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, optical_feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = optical_feature.shape
        pooled_height = height // self.pooling_factor
        pooled_width = width // self.pooling_factor
        if pooled_height < 1 or pooled_width < 1:
            raise ValueError("input is smaller than the configured HPE pooling factor")

        pooled = F.adaptive_avg_pool2d(
            optical_feature,
            output_size=(pooled_height, pooled_width),
        )
        keys = self.key_projection(pooled).flatten(2).transpose(1, 2)
        values = self.value_projection(pooled).flatten(2).transpose(1, 2)
        prototypes = self.prototypes.expand(batch, -1, -1)

        similarity = torch.einsum("bnc,bkc->bnk", keys, prototypes)
        incidence = F.softmax(similarity / math.sqrt(channels), dim=-1)
        hyperedges = torch.einsum("bnk,bnc->bkc", incidence, values)
        evolved_hyperedges = hyperedges + self.hyperedge_transformer(hyperedges)
        reconstructed = torch.einsum("bnk,bkc->bnc", incidence, evolved_hyperedges)
        reconstructed = reconstructed.transpose(1, 2).reshape(
            batch,
            channels,
            pooled_height,
            pooled_width,
        )
        reconstructed = F.interpolate(
            reconstructed,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
        return optical_feature + self.output_projection(reconstructed)


class SGIG(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.spatial_guidance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.channel_adaptation = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )
        self.fusion_projection = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(
        self,
        dynamic_sar_feature: torch.Tensor,
        contextual_optical_feature: torch.Tensor,
    ) -> torch.Tensor:
        spatial_mask = self.spatial_guidance(contextual_optical_feature)
        guided_sar = dynamic_sar_feature * spatial_mask + dynamic_sar_feature

        batch, channels, _, _ = guided_sar.shape
        channel_mask = self.channel_adaptation(
            self.global_pooling(guided_sar).reshape(batch, channels)
        ).reshape(batch, channels, 1, 1)
        adapted_sar = guided_sar * channel_mask + guided_sar
        return self.fusion_projection(
            torch.cat([adapted_sar, contextual_optical_feature], dim=1)
        )


class HPMNet(nn.Module):
    def __init__(
        self,
        time_steps: int = 12,
        sar_channels: int = 3,
        optical_channels: int = 4,
        num_classes: int = 2,
        feature_dim: int = 96,
        num_stages: int = 3,
        num_prototypes: int = 64,
        hpe_pooling_factor: int = 4,
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.sar_stem = nn.Sequential(
            nn.Conv2d(
                sar_channels * time_steps,
                feature_dim,
                kernel_size=3,
                padding=1,
                groups=sar_channels,
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )
        self.optical_stem = nn.Sequential(
            nn.Conv2d(optical_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )
        self.stages = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "hpe": HPE(
                            dim=feature_dim,
                            num_prototypes=num_prototypes,
                            pooling_factor=hpe_pooling_factor,
                        ),
                        "stme": STME(dim=feature_dim),
                        "sgig": SGIG(dim=feature_dim),
                    }
                )
                for _ in range(num_stages)
            ]
        )
        self.output_normalization = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        sar_sequence: torch.Tensor,
        optical_static: torch.Tensor,
    ) -> torch.Tensor:
        if sar_sequence.ndim != 5:
            raise ValueError("sar_sequence must have shape [B, T, Cs, H, W]")
        if optical_static.ndim != 4:
            raise ValueError("optical_static must have shape [B, Co, H, W]")

        batch, time_steps, sar_channels, height, width = sar_sequence.shape
        if time_steps != self.time_steps or sar_channels != self.sar_channels:
            raise ValueError("SAR time/channel dimensions do not match the model configuration")
        if optical_static.shape != (batch, self.optical_channels, height, width):
            raise ValueError("optical input dimensions do not match the SAR input or model configuration")

        sar = sar_sequence.permute(0, 2, 1, 3, 4).reshape(
            batch,
            sar_channels * time_steps,
            height,
            width,
        )
        sar = self.sar_stem(sar)
        optical = self.optical_stem(optical_static)
        for stage in self.stages:
            optical = stage["hpe"](optical)
            sar = stage["stme"](sar)
            sar = stage["sgig"](sar, optical)

        tokens = sar.flatten(2).transpose(1, 2)
        logits = self.classifier(self.output_normalization(tokens))
        return logits.transpose(1, 2).reshape(batch, -1, height, width)


__all__ = ["HPE", "STME", "SGIG", "HPMNet"]
