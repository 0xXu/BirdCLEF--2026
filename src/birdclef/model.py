from pathlib import Path

import torch
import torch.nn as nn

from birdclef.config import CFG
from birdclef.deps import require_dependencies, timm
from birdclef.utils import load_species_ids


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=[-2, -1]).pow(1.0 / self.p)


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg: CFG, load_backbone_weights: bool = True):
        super().__init__()
        require_dependencies(("timm", timm))
        self.cfg = cfg
        self.num_classes = len(load_species_ids(cfg))

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            in_chans=3,
            num_classes=0,
            global_pool="",
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        if load_backbone_weights:
            self._load_local_backbone_weights(cfg.pretrained_path)
        self._convert_first_conv_to_single_channel()

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.in_channels, *cfg.target_shape)
            feat = self.backbone(dummy)
            feat_dim = feat.shape[1]

        self.gem_pool = GeM()
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes),
        )

        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _load_local_backbone_weights(self, weight_dir: Path) -> None:
        if not weight_dir.exists():
            raise FileNotFoundError(
                f"Backbone local weights not found at {weight_dir}. "
                "Place the timm weights under cfg.pretrained_path before training."
            )

        weight_file = None
        for ext in ("*.safetensors", "*.pth", "*.bin", "*.pt"):
            found = list(weight_dir.glob(ext))
            if found:
                weight_file = found[0]
                break
        if weight_file is None:
            raise FileNotFoundError(f"No backbone weight file found in {weight_dir}")

        print(f"Loading backbone weights from {weight_file.name}")
        if weight_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(weight_file))
        else:
            state = torch.load(str(weight_file), map_location="cpu", weights_only=True)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        print(f"Backbone weights loaded | missing={len(missing)} unexpected={len(unexpected)}")

    def _convert_first_conv_to_single_channel(self) -> None:
        first_conv = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                first_conv = name, module
                break
        if first_conv is None:
            return

        name, conv = first_conv
        new_weight = conv.weight.data.mean(dim=1, keepdim=True)
        new_conv = nn.Conv2d(
            1,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        new_conv.weight.data = new_weight
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data

        parent = self.backbone
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)
        print(f"Converted {name} from 3-channel to 1-channel input")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = self.gem_pool(feat)
        return self.head(pooled)
