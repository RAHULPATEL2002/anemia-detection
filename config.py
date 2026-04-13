"""Central configuration for the AnemiaVision AI project.

This module keeps all runtime settings in one place so the training scripts,
evaluation utilities, and Flask application stay aligned.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parent


def normalize_token(value: str) -> str:
    """Normalize a string for case-insensitive filesystem matching."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_path(path_value: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve relative and absolute paths consistently."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def first_existing_path(candidates: Iterable[str | Path]) -> Path:
    """Return the first path that exists, or the first resolved candidate."""

    resolved_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        resolved = resolve_path(candidate)
        resolved_candidates.append(resolved)
        if resolved.exists():
            return resolved

    if not resolved_candidates:
        raise ValueError("At least one candidate path must be provided.")

    return resolved_candidates[0]


def find_split_directory(dataset_root: Path, aliases: Iterable[str]) -> Path:
    """Resolve a dataset split folder by matching aliases case-insensitively."""

    alias_tokens = {normalize_token(alias) for alias in aliases}
    if dataset_root.exists():
        for child in dataset_root.iterdir():
            if child.is_dir() and normalize_token(child.name) in alias_tokens:
                return child.resolve()
    primary_alias = next(iter(aliases))
    return (dataset_root / primary_alias).resolve()


def detect_device() -> torch.device:
    """Pick the best available device in CUDA -> MPS -> CPU order."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


TRAINING_PROFILE_ENV = os.getenv("ANEMIA_TRAIN_PROFILE", "auto").strip().lower()


def resolve_training_profile(device: torch.device) -> str:
    """Resolve the training profile from the environment and hardware."""

    profile = TRAINING_PROFILE_ENV
    if profile not in {"auto", "full", "fast"}:
        profile = "auto"

    if profile == "auto":
        return "fast" if device.type == "cpu" else "full"
    return profile


DATASET_ROOT = first_existing_path(
    [
        os.getenv("ANEMIA_DATASET_ROOT"),
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "dataset",
        PROJECT_ROOT.parent / "data",
    ]
)

TRAIN_DATASET_DIR = find_split_directory(DATASET_ROOT, ("train", "training"))
VALID_DATASET_DIR = find_split_directory(DATASET_ROOT, ("valid", "val", "validation"))
TEST_DATASET_DIR = find_split_directory(DATASET_ROOT, ("test", "testing"))

MODELS_DIR = resolve_path(os.getenv("ANEMIA_MODELS_DIR", PROJECT_ROOT / "models"))
STATIC_DIR = resolve_path(os.getenv("ANEMIA_STATIC_DIR", PROJECT_ROOT / "static"))
TEMPLATES_DIR = resolve_path(os.getenv("ANEMIA_TEMPLATES_DIR", PROJECT_ROOT / "templates"))
DATABASE_DIR = resolve_path(os.getenv("ANEMIA_DATABASE_DIR", PROJECT_ROOT / "database"))
REPORTS_DIR = resolve_path(os.getenv("ANEMIA_REPORTS_DIR", PROJECT_ROOT / "reports"))
LOGS_DIR = resolve_path(os.getenv("ANEMIA_LOGS_DIR", PROJECT_ROOT / "logs"))
EVALUATION_LOGS_DIR = resolve_path(
    os.getenv("ANEMIA_EVALUATION_LOGS_DIR", LOGS_DIR / "evaluation")
)
UPLOADS_DIR = resolve_path(os.getenv("ANEMIA_UPLOADS_DIR", STATIC_DIR / "uploads"))
GRADCAM_DIR = resolve_path(os.getenv("ANEMIA_GRADCAM_DIR", STATIC_DIR / "gradcam"))

DATABASE_PATH = resolve_path(os.getenv("ANEMIA_DATABASE_PATH", DATABASE_DIR / "anemia_vision.db"))
BEST_CHECKPOINT_PATH = resolve_path(
    os.getenv("ANEMIA_BEST_CHECKPOINT", MODELS_DIR / "best_model.pth")
)
LATEST_CHECKPOINT_PATH = resolve_path(
    os.getenv("ANEMIA_LATEST_CHECKPOINT", MODELS_DIR / "last_checkpoint.pth")
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = ("non-anemic", "anemic")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
API_RATE_LIMIT_PER_MINUTE = int(os.getenv("ANEMIA_RATE_LIMIT_PER_MINUTE", "10"))
API_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("ANEMIA_RATE_LIMIT_WINDOW_SECONDS", "60"))
GUNICORN_WORKERS = int(os.getenv("ANEMIA_GUNICORN_WORKERS", "2"))


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters shared across scripts."""

    image_size: tuple[int, int] = (300, 300)
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    early_stopping_patience: int = 10
    warmup_epochs: int = 5
    scheduler_t_max: int = 50
    scheduler_eta_min: float = 1e-6
    num_workers: int = field(default_factory=lambda: max(0, min(8, (os.cpu_count() or 2) - 1)))
    random_seed: int = 42
    device: torch.device = field(default_factory=detect_device)
    profile: str = field(default="auto")
    use_amp: bool = field(init=False)
    pin_memory: bool = field(init=False)

    def __post_init__(self) -> None:
        resolved_profile = resolve_training_profile(self.device)
        object.__setattr__(self, "profile", resolved_profile)

        if resolved_profile == "fast":
            object.__setattr__(self, "image_size", (224, 224))
            object.__setattr__(self, "batch_size", min(self.batch_size, 16))
            object.__setattr__(self, "epochs", min(self.epochs, 12))
            object.__setattr__(self, "early_stopping_patience", min(self.early_stopping_patience, 4))
            object.__setattr__(self, "warmup_epochs", min(self.warmup_epochs, 2))

        object.__setattr__(self, "use_amp", self.device.type == "cuda")
        object.__setattr__(self, "pin_memory", self.device.type == "cuda")

        num_workers_override = os.getenv("ANEMIA_NUM_WORKERS")
        if num_workers_override is not None:
            try:
                object.__setattr__(self, "num_workers", max(0, int(num_workers_override)))
            except ValueError:
                pass
        elif sys.platform.startswith("win") and self.device.type == "cpu":
            object.__setattr__(self, "num_workers", 0)


@dataclass(frozen=True)
class AugmentationConfig:
    """Image augmentation settings for training."""

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.3
    rotation_degrees: int = 15
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.1
    affine_translate: tuple[float, float] = (0.05, 0.05)
    affine_scale: tuple[float, float] = (0.95, 1.05)
    affine_shear: int = 5
    gaussian_blur_probability: float = 0.2
    gaussian_blur_kernel_size: int = 3
    gaussian_blur_sigma: tuple[float, float] = (0.1, 2.0)

    def __post_init__(self) -> None:
        if TRAINING_CONFIG.profile != "fast":
            return

        object.__setattr__(self, "vertical_flip_probability", 0.0)
        object.__setattr__(self, "rotation_degrees", 10)
        object.__setattr__(self, "brightness", 0.2)
        object.__setattr__(self, "contrast", 0.2)
        object.__setattr__(self, "saturation", 0.1)
        object.__setattr__(self, "hue", 0.05)
        object.__setattr__(self, "affine_translate", (0.0, 0.0))
        object.__setattr__(self, "affine_scale", (1.0, 1.0))
        object.__setattr__(self, "affine_shear", 0)
        object.__setattr__(self, "gaussian_blur_probability", 0.0)


@dataclass(frozen=True)
class FlaskConfig:
    """Flask settings for the hospital-facing application."""

    secret_key: str = field(
        default_factory=lambda: os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
    )
    max_content_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "16")) * 1024 * 1024
    )
    upload_folder: Path = UPLOADS_DIR
    gradcam_folder: Path = GRADCAM_DIR
    reports_folder: Path = REPORTS_DIR
    database_uri: str = field(
        default_factory=lambda: f"sqlite:///{DATABASE_PATH.as_posix()}"
    )
    sqlalchemy_track_modifications: bool = False
    allowed_extensions: tuple[str, ...] = tuple(sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS))

    def to_flask_dict(self) -> dict[str, object]:
        """Convert the config into a structure Flask understands."""

        return {
            "SECRET_KEY": self.secret_key,
            "MAX_CONTENT_LENGTH": self.max_content_length,
            "UPLOAD_FOLDER": str(self.upload_folder),
            "GRADCAM_FOLDER": str(self.gradcam_folder),
            "REPORTS_FOLDER": str(self.reports_folder),
            "SQLALCHEMY_DATABASE_URI": self.database_uri,
            "SQLALCHEMY_TRACK_MODIFICATIONS": self.sqlalchemy_track_modifications,
        }


TRAINING_CONFIG = TrainingConfig()
AUGMENTATION_CONFIG = AugmentationConfig()
FLASK_CONFIG = FlaskConfig()

MODEL_ARCH_ENV = os.getenv("ANEMIA_MODEL_ARCH")
MODEL_ARCH = (
    MODEL_ARCH_ENV.strip().lower()
    if MODEL_ARCH_ENV
    else ("efficientnet_b0" if TRAINING_CONFIG.profile == "fast" else "efficientnet_b3")
)


def ensure_runtime_directories() -> None:
    """Create every runtime directory used by the project."""

    for directory in (
        MODELS_DIR,
        STATIC_DIR,
        TEMPLATES_DIR,
        DATABASE_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        EVALUATION_LOGS_DIR,
        UPLOADS_DIR,
        GRADCAM_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def serializable_project_config() -> dict[str, object]:
    """Return a JSON-friendly snapshot of the project configuration."""

    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(DATASET_ROOT),
        "train_dataset_dir": str(TRAIN_DATASET_DIR),
        "valid_dataset_dir": str(VALID_DATASET_DIR),
        "test_dataset_dir": str(TEST_DATASET_DIR),
        "models_dir": str(MODELS_DIR),
        "database_path": str(DATABASE_PATH),
        "best_checkpoint_path": str(BEST_CHECKPOINT_PATH),
        "latest_checkpoint_path": str(LATEST_CHECKPOINT_PATH),
        "evaluation_logs_dir": str(EVALUATION_LOGS_DIR),
        "class_names": list(CLASS_NAMES),
        "image_extensions": sorted(IMAGE_EXTENSIONS),
        "training_profile": TRAINING_CONFIG.profile,
        "model_arch": MODEL_ARCH,
        "training": {
            **asdict(TRAINING_CONFIG),
            "device": str(TRAINING_CONFIG.device),
        },
        "augmentation": asdict(AUGMENTATION_CONFIG),
        "flask": {
            "secret_key": FLASK_CONFIG.secret_key,
            "max_content_length": FLASK_CONFIG.max_content_length,
            "upload_folder": str(FLASK_CONFIG.upload_folder),
            "gradcam_folder": str(FLASK_CONFIG.gradcam_folder),
            "reports_folder": str(FLASK_CONFIG.reports_folder),
            "database_uri": FLASK_CONFIG.database_uri,
            "allowed_extensions": list(FLASK_CONFIG.allowed_extensions),
        },
    }


ensure_runtime_directories()
