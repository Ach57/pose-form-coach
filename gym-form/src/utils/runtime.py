from __future__ import annotations
import os

def ensure_env_setup() -> tuple[bool, str]:
    """
    Detect runtime environment and set working directory.

    Returns:
        IN_COLAB (bool): Whether running in Google Colab
        repo_root (str): Absolute path to repository root
    """
    in_colab = "COLAB_GPU" in os.environ or os.path.exists("/content")

    if in_colab:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)

        repo_root = "/content/gym-form"
    else:
        repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}")

    return in_colab, repo_root