# models.py
"""
Neural network architectures for Fingerprint Spoof Detection.
All torch.nn.Module classes and model factory extracted from the main Streamlit app.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


# ---------------------------------------------------------------------------
# Architecture definitions
# ---------------------------------------------------------------------------

class FingerprintResNet(nn.Module):
    """ResNet18 with Transfer Learning for Fingerprint Spoof Detection."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
        )
        if not pretrained:
            nn.init.xavier_uniform_(self.model.fc[1].weight)
            nn.init.constant_(self.model.fc[1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FingerprintResNetA600(nn.Module):
    """ResNet18 compatible with Colab-trained checkpoints (Dropout → Linear head)."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.model = models.resnet18(weights=weights)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
        )
        if pretrained:
            nn.init.xavier_uniform_(self.model.fc[1].weight)
            nn.init.constant_(self.model.fc[1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class WorkingModel(nn.Module):
    """Lightweight CNN baseline used for smoke-testing."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class FingerprintResNetCustom(nn.Module):
    """Custom shallow ResNet variant for best_fingerprint_model.pth checkpoints."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class FingerprintResNetNagaraju(nn.Module):
    """
    ResNet18 backbone with a 4-layer custom head that exactly matches
    Nagaraju checkpoints (FC.1, FC.2, FC.5 keys).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_features = self.model.fc.in_features

        # Named sub-modules matching checkpoint key layout
        self.fc1_linear = nn.Linear(num_features, 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc5_linear = nn.Linear(512, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet backbone (excluding original fc)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        # Custom head
        x = self.dropout1(x)
        x = self.fc1_linear(x)
        x = self.fc_bn(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return self.fc5_linear(x)

    def load_state_dict_custom(self, state_dict: dict) -> None:
        """
        Split the flat checkpoint into backbone weights and custom-head
        weights, then load each group separately.
        """
        backbone_dict, custom_dict = {}, {}
        for key, value in state_dict.items():
            (custom_dict if key.startswith("model.fc.") else backbone_dict)[key] = value

        self.model.load_state_dict(backbone_dict, strict=False)

        # FC.1 – first linear layer
        self.fc1_linear.weight.data = custom_dict["model.fc.1.weight"]
        self.fc1_linear.bias.data = custom_dict["model.fc.1.bias"]

        # FC.2 – BatchNorm layer
        self.fc_bn.weight.data = custom_dict["model.fc.2.weight"]
        self.fc_bn.bias.data = custom_dict["model.fc.2.bias"]
        self.fc_bn.running_mean.data = custom_dict["model.fc.2.running_mean"]
        self.fc_bn.running_var.data = custom_dict["model.fc.2.running_var"]
        self.fc_bn.num_batches_tracked.data = custom_dict["model.fc.2.num_batches_tracked"]

        # FC.5 – final linear layer
        self.fc5_linear.weight.data = custom_dict["model.fc.5.weight"]
        self.fc5_linear.bias.data = custom_dict["model.fc.5.bias"]


class FingerprintResNet25Layer(nn.Module):
    """ResNet18 with a 2.5-layer head: Dropout→Linear→BN→ReLU→Dropout→Linear."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FingerprintResNetLegacy(nn.Module):
    """Legacy ResNet18 variant (identical head to FingerprintResNet25Layer, kept for compat)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FingerprintResNetEnhanced(nn.Module):
    """ResNet18 with a 3-layer head for the optimised / fine-tuned model."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_state_dict(checkpoint: object) -> dict:
    """Return a plain state-dict regardless of checkpoint format."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
        return checkpoint  # already a flat state-dict
    return checkpoint


def _detect_architecture(model_path: str, state_dict: dict):
    """
    Return an uninitialised model instance whose architecture matches the
    checkpoint found at *model_path*.

    Detection priority
    ------------------
    1. Checkpoint key fingerprint   – most reliable
    2. Filename heuristics          – fallback
    3. Default                      – FingerprintResNetA600
    """
    filename = Path(model_path).name.lower()
    fc_keys = [k for k in state_dict if "fc" in k]

    # --- key-based detection ---
    has_nagaraju_keys = any(
        k in fc_keys for k in ("model.fc.2.weight", "model.fc.5.weight")
    )
    if has_nagaraju_keys:
        return FingerprintResNetNagaraju(num_classes=2), "nagaraju"

    # --- filename-based detection ---
    if "nagaraju" in filename:
        return FingerprintResNetNagaraju(num_classes=2), "nagaraju"
    if "working_test_model" in filename:
        return WorkingModel(num_classes=2), "standard"
    if "best_fingerprint_model" in filename:
        return FingerprintResNetCustom(num_classes=2), "standard"
    if any(kw in filename for kw in ("a600", "colab", "resnet18", "final", "best")):
        return FingerprintResNetA600(num_classes=2, pretrained=False), "standard"

    # --- default ---
    return FingerprintResNetA600(num_classes=2, pretrained=False), "standard"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_model_by_name(
    model_path: str,
) -> tuple[nn.Module, torch.device]:
    """
    Detect architecture from *model_path*, load weights, and return an
    eval-mode model together with the active device.

    Parameters
    ----------
    model_path:
        Path to a ``.pth`` checkpoint file.

    Returns
    -------
    model : nn.Module
        Weights loaded, moved to device, set to ``eval()``.
    device : torch.device
        The device the model lives on.

    Raises
    ------
    RuntimeError
        If the checkpoint cannot be loaded or the weights cannot be applied.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load raw checkpoint
    # ------------------------------------------------------------------
    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    except Exception as primary_err:
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as fallback_err:
            raise RuntimeError(
                f"Cannot load checkpoint '{model_path}'.\n"
                f"  Primary error  : {primary_err}\n"
                f"  Fallback error : {fallback_err}"
            ) from fallback_err

    # ------------------------------------------------------------------
    # 2. Extract state dict
    # ------------------------------------------------------------------
    try:
        state_dict = _extract_state_dict(checkpoint)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract state-dict from '{model_path}': {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 3. Select matching architecture
    # ------------------------------------------------------------------
    try:
        model, load_mode = _detect_architecture(model_path, state_dict)
    except Exception as exc:
        raise RuntimeError(
            f"Architecture detection failed for '{model_path}': {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 4. Load weights
    # ------------------------------------------------------------------
    try:
        if load_mode == "nagaraju" and hasattr(model, "load_state_dict_custom"):
            model.load_state_dict_custom(state_dict)
        else:
            # Prefer lenient loading; escalate to strict only to surface
            # meaningful errors when lenient loading silently succeeds but
            # leaves the model in a degraded state.
            missing, unexpected = model.load_state_dict(
                state_dict, strict=False
            )
            if missing:
                print(
                    f"[models.py] Warning – missing keys ({len(missing)}): "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            if unexpected:
                print(
                    f"[models.py] Warning – unexpected keys ({len(unexpected)}): "
                    f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
                )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load weights into {type(model).__name__} "
            f"from '{model_path}': {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 5. Finalise
    # ------------------------------------------------------------------
    model = model.to(device)
    model.eval()

    print(
        f"[models.py] Loaded  : {Path(model_path).name}\n"
        f"            Arch    : {type(model).__name__}\n"
        f"            Device  : {device}"
    )
    return model, device