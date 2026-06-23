"""
EdgeGuard: Model Registry Module

Versioned model storage and hot-update management for edge devices.
Tracks model versions, validates integrity, and manages deployment lifecycle.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelVersion:
    """Metadata for a single model version."""
    version: str
    created_at: float
    file_path: str
    file_size: int
    checksum: str
    precision: str  # fp32, fp16, int8
    framework: str  # pytorch, tensorrt
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    is_active: bool = False


class ModelRegistry:
    """
    Versioned model registry for edge deployment.

    Manages model lifecycle:
    - Store multiple model versions
    - Track active vs inactive versions
    - Validate checksums on download
    - Enforce max version retention
    - Hot-swap active inference engine

    Args:
        storage_dir: Directory for model storage.
        max_versions: Maximum number of versions to retain.
    """

    METADATA_FILE = "registry.json"
    INDEX_FILE = "model_index.json"

    def __init__(
        self,
        storage_dir: str = "models",
        max_versions: int = 5,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_versions = max_versions
        self._metadata_path = self.storage_dir / self.METADATA_FILE
        self.versions: dict[str, ModelVersion] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load registry metadata from disk."""
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for v in data.get("versions", []):
                    self.versions[v["version"]] = ModelVersion(**v)
            except Exception as e:
                print(f"Warning: Could not load registry metadata: {e}")

    def _save_metadata(self) -> None:
        """Save registry metadata to disk."""
        data = {
            "versions": [
                {
                    "version": v.version,
                    "created_at": v.created_at,
                    "file_path": v.file_path,
                    "file_size": v.file_size,
                    "checksum": v.checksum,
                    "precision": v.precision,
                    "framework": v.framework,
                    "metrics": v.metrics,
                    "config": v.config,
                    "is_active": v.is_active,
                }
                for v in self.versions.values()
            ],
            "active_version": self.get_active_version(),
        }
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register_model(
        self,
        model_path: str,
        version: str,
        precision: str = "fp16",
        framework: str = "tensorrt",
        metrics: dict | None = None,
        config: dict | None = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_path: Path to model file.
            version: Semantic version string (e.g., "1.0.0").
            precision: Model precision mode.
            framework: Model framework.
            metrics: Optional performance metrics.
            config: Optional model configuration.

        Returns:
            ModelVersion object for the registered model.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        file_size = model_path.stat().st_size
        checksum = self._compute_checksum(model_path)

        for v in self.versions.values():
            v.is_active = False

        model_version = ModelVersion(
            version=version,
            created_at=time.time(),
            file_path=str(model_path),
            file_size=file_size,
            checksum=checksum,
            precision=precision,
            framework=framework,
            metrics=metrics or {},
            config=config or {},
            is_active=True,
        )

        self.versions[version] = model_version
        self._save_metadata()
        self._enforce_max_versions()

        print(f"[Registry] Registered model version {version} ({precision})")
        print(f"  File: {model_path}")
        print(f"  Size: {file_size / 1024 / 1024:.1f} MB")
        print(f"  Checksum: {checksum[:16]}...")

        return model_version

    def set_active_version(self, version: str) -> bool:
        """
        Set a model version as active for inference.

        Args:
            version: Version string to activate.

        Returns:
            True if successful, False if version not found.
        """
        if version not in self.versions:
            print(f"[Registry] Version {version} not found")
            return False

        for v in self.versions.values():
            v.is_active = False

        self.versions[version].is_active = True
        self._save_metadata()

        print(f"[Registry] Activated model version {version}")
        return True

    def get_active_version(self) -> Optional[str]:
        """Return the currently active model version."""
        for v in self.versions.values():
            if v.is_active:
                return v.version
        return None

    def get_active_model_path(self) -> Optional[Path]:
        """Return the path to the active model file."""
        active = self.get_active_version()
        if active and active in self.versions:
            return Path(self.versions[active].file_path)
        return None

    def list_versions(self) -> list[ModelVersion]:
        """List all registered model versions."""
        return sorted(
            self.versions.values(),
            key=lambda v: v.created_at,
            reverse=True,
        )

    def validate_checksum(self, version: str, file_path: str) -> bool:
        """
        Validate model file integrity against stored checksum.

        Args:
            version: Model version to validate.
            file_path: Path to the model file.

        Returns:
            True if checksum matches, False otherwise.
        """
        if version not in self.versions:
            return False

        actual = self._compute_checksum(Path(file_path))
        expected = self.versions[version].checksum
        return actual == expected

    def delete_version(self, version: str) -> bool:
        """
        Delete a model version from the registry.

        Args:
            version: Version string to delete.

        Returns:
            True if deleted, False if not found.
        """
        if version not in self.versions:
            return False

        model_path = Path(self.versions[version].file_path)
        if model_path.exists():
            model_path.unlink()

        del self.versions[version]
        self._save_metadata()

        print(f"[Registry] Deleted model version {version}")
        return True

    def _enforce_max_versions(self) -> None:
        """Remove oldest versions when exceeding max_versions limit."""
        if len(self.versions) <= self.max_versions:
            return

        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1].created_at,
        )

        to_delete = sorted_versions[:-self.max_versions]
        for version, _ in to_delete:
            self.delete_version(version)

    @staticmethod
    def _compute_checksum(file_path: Path, algorithm: str = "sha256") -> str:
        """Compute file checksum."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def export_index(self, output_path: str | None = None) -> dict:
        """
        Export model index for cloud-side tracking.

        Args:
            output_path: Optional path to save index JSON.

        Returns:
            Dictionary of available model versions.
        """
        index = {
            "generated_at": time.time(),
            "device_id": "edge-001",
            "active_version": self.get_active_version(),
            "models": {},
        }

        for version, mv in self.versions.items():
            index["models"][version] = {
                "file_size": mv.file_size,
                "checksum": mv.checksum,
                "precision": mv.precision,
                "framework": mv.framework,
                "created_at": mv.created_at,
                "metrics": mv.metrics,
            }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
            print(f"[Registry] Index exported to: {output_path}")

        return index
