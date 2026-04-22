from __future__ import annotations

from pathlib import Path
import os

def ensure_env_setup(
    repo_root: Path | None = None,
) -> tuple[bool, Path]:
    """
    Detect runtime environment and set working directory.

    Args:
        repo_root (Path | None):
            Path to the repository root. If None, a reasonable default
            is chosen based on the execution environment.

    Returns:
        in_colab (bool): Whether running in Google Colab
        repo_root (Path): Absolute path to repository root
    """
    in_colab = "COLAB_GPU" in os.environ or Path("/content").exists()

    if repo_root is None:
        if in_colab:
            repo_root = Path("/content/pose-form-coach/gym-form")
        else:
            # assumes notebook is inside repo or a subfolder of it
            repo_root = Path.cwd().resolve()

    repo_root = repo_root.resolve()

    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")

    os.chdir(repo_root)
    print(f"Working directory set to: {repo_root}")

    return in_colab, repo_root