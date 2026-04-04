# src/utils/checkpoints.py
from __future__ import annotations

from pathlib import Path
import shutil

def save_checkpoints(
    checkpoints_dir: Path,
    in_colab: bool,
    drive_dest: Path | None = None,
) -> None:
    """
    Save model checkpoints, copying them to Google Drive if running in Colab.

    Args:
        checkpoints_dir (Path):
            Local directory containing checkpoint files (e.g. best.pt, last.pt).

        in_colab (bool):
            Whether the code is running in Google Colab.

        drive_dest (Path | None):
            Destination directory on Google Drive. Required if in_colab=True.
    """
    checkpoints_dir = checkpoints_dir.resolve()

    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    if in_colab:
        if drive_dest is None:
            raise ValueError("drive_dest must be provided when running in Colab")

        drive_dest = drive_dest.resolve()
        drive_dest.mkdir(parents=True, exist_ok=True)

        for name in ["best.pt", "last.pt"]:
            src = checkpoints_dir / name
            if src.exists():
                shutil.copy(src, drive_dest / name)

        print(f"Checkpoints saved to {drive_dest}")
    else:
        print(f"Checkpoints available at {checkpoints_dir}")
