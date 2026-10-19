"""GCP, Google Cloud Storage, Vertex AI, and GPU execution utilities."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def is_gcs_uri(path: str | Path | None) -> bool:
    """Return True if path starts with gs://."""
    if path is None:
        return False
    return str(path).strip().startswith("gs://")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split gs://bucket/prefix into (bucket, prefix)."""
    if not is_gcs_uri(uri):
        raise ValueError(f"Invalid GCS URI: {uri}")
    trimmed = uri[len("gs://") :]
    parts = trimmed.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def download_from_gcs(gcs_uri: str, local_dest: str | Path) -> Path:
    """Download a file or directory from GCS to a local path.

    Tries python google-cloud-storage if installed, then falls back to gcloud/gsutil.
    """
    dest = Path(local_dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s...", gcs_uri, dest)

    try:
        from google.cloud import storage  # type: ignore[import-untyped]

        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blob = bucket.blob(blob_name)
        if blob.exists():
            blob.download_to_filename(str(dest))
            logger.info("Downloaded file %s", dest)
            return dest

        # Check if it's a directory/prefix
        dest.mkdir(parents=True, exist_ok=True)
        blobs = list(bucket.list_blobs(prefix=blob_name.rstrip("/") + "/"))
        if blobs:
            prefix_len = len(blob_name.rstrip("/") + "/")
            for b in blobs:
                rel_path = b.name[prefix_len:]
                if not rel_path or rel_path.endswith("/"):
                    continue
                file_dest = dest / rel_path
                file_dest.parent.mkdir(parents=True, exist_ok=True)
                b.download_to_filename(str(file_dest))
            logger.info("Downloaded %d blobs to directory %s", len(blobs), dest)
            return dest
    except ImportError:
        pass
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning("google-cloud-storage download failed (%s); trying CLI...", exc)

    # CLI fallback: gcloud storage or gsutil
    cmd = None
    if shutil.which("gcloud"):
        cmd = ["gcloud", "storage", "cp", "-r", gcs_uri, str(dest)]
    elif shutil.which("gsutil"):
        cmd = ["gsutil", "-m", "cp", "-r", gcs_uri, str(dest)]

    if cmd:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            logger.info("Downloaded via CLI %s", cmd[0])
            return dest
        raise RuntimeError(f"GCS CLI download failed: {res.stderr}")

    raise RuntimeError(
        f"Cannot download {gcs_uri}: neither google-cloud-storage nor gcloud/gsutil CLI is available"
    )


def upload_to_gcs(local_src: str | Path, gcs_uri: str) -> None:
    """Upload a local file or directory to GCS."""
    src = Path(local_src)
    if not src.exists():
        raise FileNotFoundError(f"Local file/dir not found: {src}")

    logger.info("Uploading %s to %s...", src, gcs_uri)
    try:
        from google.cloud import storage  # type: ignore[import-untyped]

        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        if src.is_file():
            target_blob = blob_name if blob_name and not blob_name.endswith("/") else (
                f"{blob_name.rstrip('/')}/{src.name}" if blob_name else src.name
            )
            blob = bucket.blob(target_blob)
            blob.upload_from_filename(str(src))
            logger.info("Uploaded %s to gs://%s/%s", src, bucket_name, target_blob)
            return
        elif src.is_dir():
            for root, _, files in os.walk(src):
                for f in files:
                    full_path = Path(root) / f
                    rel_path = full_path.relative_to(src)
                    target_blob = (
                        f"{blob_name.rstrip('/')}/{rel_path}" if blob_name else str(rel_path)
                    )
                    blob = bucket.blob(target_blob)
                    blob.upload_from_filename(str(full_path))
            logger.info("Uploaded directory %s to %s", src, gcs_uri)
            return
    except ImportError:
        pass
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning("google-cloud-storage upload failed (%s); trying CLI...", exc)

    cmd = None
    if shutil.which("gcloud"):
        cmd = ["gcloud", "storage", "cp", "-r", str(src), gcs_uri]
    elif shutil.which("gsutil"):
        cmd = ["gsutil", "-m", "cp", "-r", str(src), gcs_uri]

    if cmd:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            logger.info("Uploaded via CLI %s", cmd[0])
            return
        raise RuntimeError(f"GCS CLI upload failed: {res.stderr}")

    raise RuntimeError(
        f"Cannot upload {src} to {gcs_uri}: neither google-cloud-storage nor gcloud/gsutil CLI is available"
    )


def get_vertex_ai_environment() -> dict[str, str]:
    """Retrieve Vertex AI training environment variables if present."""
    keys = (
        "AIP_MODEL_DIR",
        "AIP_CHECKPOINT_DIR",
        "AIP_TENSORBOARD_LOG_DIR",
        "AIP_DATA_FORMAT",
        "CLOUD_ML_JOB_ID",
    )
    env = {}
    for key in keys:
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


def resolve_device(requested_device: str = "auto") -> tuple[torch.device, dict[str, Any]]:
    """Resolve target execution device and collect accelerator hardware metrics."""
    hardware_info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device_count": 0,
        "device_name": "CPU",
        "total_memory_gb": 0.0,
    }

    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    if device.type == "cuda" and torch.cuda.is_available():
        hardware_info["device_count"] = torch.cuda.device_count()
        hardware_info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        hardware_info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
        hardware_info["cuda_version"] = torch.version.cuda
        hardware_info["cudnn_version"] = torch.backends.cudnn.version()
    elif device.type == "mps":
        hardware_info["device_name"] = "Apple Silicon GPU (MPS)"
    else:
        hardware_info["device_name"] = "CPU"

    logger.info("Resolved device: %s (%s)", device, hardware_info["device_name"])
    return device, hardware_info
