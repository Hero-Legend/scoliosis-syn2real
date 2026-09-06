"""Functions extracted without body changes from the existing training implementation."""
from __future__ import annotations
import csv, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from engine.utils import stable_seed, ordinal_targets, decode_ordinal_probabilities


def set_determinism(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _clip_and_rescale(grayscale: np.ndarray) -> np.ndarray:
    low, high = np.percentile(grayscale, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(grayscale, dtype=np.float32)
    return np.clip((grayscale - low) / (high - low), 0.0, 1.0).astype(np.float32)


def prepare_image(
    path: Path,
    size: int,
    train: bool,
    augmentation_seed: int,
    augmentation: dict,
) -> np.ndarray:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf

    with Image.open(path) as source:
        grayscale = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
    normalized = _clip_and_rescale(grayscale)
    height, width = normalized.shape
    resize_scale = min(size / width, size / height)
    target_w = max(1, round(width * resize_scale))
    target_h = max(1, round(height * resize_scale))
    resized = Image.fromarray((normalized * 255).round().astype(np.uint8)).resize(
        (target_w, target_h), Image.Resampling.BICUBIC
    )
    padding = int(round(float(np.median(normalized)) * 255.0))
    canvas = Image.new("L", (size, size), color=padding)
    canvas.paste(resized, ((size - target_w) // 2, (size - target_h) // 2))

    rng = np.random.default_rng(augmentation_seed)
    if train:
        angle = float(rng.uniform(-augmentation["rotation_degrees"], augmentation["rotation_degrees"]))
        scale = float(rng.uniform(*augmentation["scale"]))
        translate_max = float(augmentation["translation_fraction_max"]) * size
        translate = [int(round(rng.uniform(-translate_max, translate_max))) for _ in range(2)]
        canvas = tf.affine(
            canvas,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BICUBIC,
            fill=padding,
        )
        if rng.random() < float(augmentation["horizontal_flip_probability"]):
            canvas = tf.hflip(canvas)
        brightness = 1.0 + float(rng.uniform(-augmentation["brightness_fraction"], augmentation["brightness_fraction"]))
        contrast = 1.0 + float(rng.uniform(-augmentation["contrast_fraction"], augmentation["contrast_fraction"]))
        gamma = float(rng.uniform(*augmentation["gamma"]))
        canvas = ImageEnhance.Brightness(canvas).enhance(brightness)
        canvas = ImageEnhance.Contrast(canvas).enhance(contrast)
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        array = np.power(np.clip(array, 0.0, 1.0), gamma).astype(np.float32)
        if augmentation["mild_blur"] and rng.random() < 0.25:
            radius = float(rng.uniform(0.2, 0.8))
            array = np.asarray(
                Image.fromarray((array * 255).round().astype(np.uint8)).filter(
                    ImageFilter.GaussianBlur(radius=radius)
                ),
                dtype=np.float32,
            ) / 255.0
        if augmentation["gaussian_noise"]:
            noise_sd = float(rng.uniform(0.0, 0.02))
            array = np.clip(array + rng.normal(0.0, noise_sd, array.shape), 0.0, 1.0).astype(np.float32)
    else:
        array = np.asarray(canvas, dtype=np.float32) / 255.0

    rgb = np.repeat(array[None, :, :], 3, axis=0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    result = ((rgb - mean) / std).astype(np.float32)
    if result.shape != (3, size, size) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid preprocessed image: {path}")
    return result


def build_torch_components(config: dict, objective: str, seed: int):
    import timm
    import torch
    from torch import nn
    from torch.nn import functional as f

    class OrderedOrdinalHead(nn.Module):
        def __init__(self, features: int):
            super().__init__()
            self.score = nn.Linear(features, 1)
            self.center = nn.Parameter(torch.zeros(1))
            self.raw_gap = nn.Parameter(torch.zeros(1))

        def forward(self, features):
            score = self.score(features)
            gap = f.softplus(self.raw_gap) + 1e-4
            return torch.cat(
                [score + self.center + gap / 2.0, score + self.center - gap / 2.0], dim=1
            )

    class SeverityModel(nn.Module):
        def __init__(self):
            super().__init__()
            backbone_cfg = config["backbone"]
            self.backbone = timm.create_model(backbone_cfg["name"], pretrained=False, num_classes=0)
            source_state = torch.load(
                Path(backbone_cfg["official_weight_path"]), map_location="cpu", weights_only=True
            )
            source_state.pop("mask_token")
            self.backbone.load_state_dict(source_state, strict=True)
            if config["training"]["gradient_checkpointing"]:
                self.backbone.set_grad_checkpointing(True)
            if objective == "softmax_cross_entropy":
                self.head = nn.Linear(self.backbone.num_features, 3)
            elif objective == "two_threshold_cumulative_bce":
                self.head = OrderedOrdinalHead(self.backbone.num_features)
            else:
                raise ValueError(f"unsupported objective: {objective}")

        def forward(self, images):
            raw = self.backbone.forward_features(images)
            if isinstance(raw, dict):
                features = raw["x_norm_clstoken"]
            elif raw.ndim == 3:
                features = raw[:, 0]
            else:
                features = raw
            return self.head(features)

    set_determinism(seed)
    return SeverityModel()


def loss_for(objective: str, logits, grades):
    import torch
    from torch.nn import functional as f

    if objective == "softmax_cross_entropy":
        return f.cross_entropy(logits, grades)
    targets = torch.stack([grades > 0, grades > 1], dim=1).to(dtype=logits.dtype)
    return f.binary_cross_entropy_with_logits(logits, targets)


def probabilities_and_predictions(objective: str, logits):
    import torch

    if objective == "softmax_cross_entropy":
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        return probabilities, predictions
    cumulative = torch.sigmoid(logits)
    if torch.any(cumulative[:, 0] + 1e-6 < cumulative[:, 1]):
        raise RuntimeError("ordered ordinal head violated monotonicity")
    probabilities = torch.stack(
        [1.0 - cumulative[:, 0], cumulative[:, 0] - cumulative[:, 1], cumulative[:, 1]],
        dim=1,
    )
    predictions = (cumulative > 0.5).sum(dim=1)
    return probabilities, predictions


def make_dataset(rows: list[dict], config: dict, train: bool, seed: int):
    import torch
    from torch.utils.data import Dataset

    class RadiographDataset(Dataset):
        def __init__(self):
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __len__(self):
            return len(rows)

        def __getitem__(self, index: int):
            row = rows[index]
            array = prepare_image(
                Path(row["image_path"]),
                int(config["backbone"]["input_size"]),
                train,
                stable_seed(seed, self.epoch, row["case_id"]),
                config["training"]["augmentation"],
            )
            return torch.from_numpy(array), int(row["severity_grade"]), row["case_id"]

    return RadiographDataset()


def make_loader(dataset, batch_size: int, seed: int, epoch: int, workers: int, replacement=False, num_samples=None):
    import torch
    from torch.utils.data import DataLoader, RandomSampler

    dataset.set_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(stable_seed(seed, "loader", epoch))
    if replacement:
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples, generator=generator)
    else:
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )


def make_eval_loader(dataset, batch_size: int, workers: int):
    from torch.utils.data import DataLoader

    dataset.set_epoch(0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def create_optimizer_scheduler(model, config: dict, total_steps: int, warmup_steps: int):
    import torch

    training = config["training"]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": training["backbone_learning_rate"]},
            {"params": model.head.parameters(), "lr": training["head_learning_rate"]},
        ],
        weight_decay=training["weight_decay"],
    )

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def validate_optimizer_step_authority(optimizer, expected_steps: int) -> None:
    observed = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        value = state["step"]
        observed.append(int(value.item()) if hasattr(value, "item") else int(value))
    if expected_steps == 0 and not observed:
        return
    if not observed or min(observed) != expected_steps or max(observed) != expected_steps:
        raise RuntimeError(
            "optimizer step authority mismatch: "
            f"expected={expected_steps}, observed_min={min(observed) if observed else None}, "
            f"observed_max={max(observed) if observed else None}"
        )


def atomic_torch_save(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def better_metrics(candidate: dict, incumbent: dict | None, epoch: int, incumbent_epoch: int | None) -> bool:
    if incumbent is None:
        return True
    candidate_key = (candidate["qwk"], candidate["macro_f1"], -candidate["grade_mae"], -epoch)
    incumbent_key = (incumbent["qwk"], incumbent["macro_f1"], -incumbent["grade_mae"], -(incumbent_epoch or 0))
    return candidate_key > incumbent_key
