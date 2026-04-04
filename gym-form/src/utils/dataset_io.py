from __future__ import annotations

from pathlib import Path
from utils.io import unzip_file

def prepare_dataset_archives(base_dir: Path) -> None:   
    # TODO: This needs to be generalized
    # TODO: No exception handling in this method. If the process breaks for one, it should continue for the others
    # TODO: Should accept any number of arguments 
    """
    Unzip features, labels, poses, and splits archives and fix folder structure.

    Expected zip files in base_dir:
        - features.zip
        - labels.zip
        - poses.zip
        - Splits.zip
    """
    base_dir = base_dir.resolve()

    # ---- Unzip core data ----
    data_dir = base_dir / "data"

    unzip_file(base_dir / "features.zip", data_dir)
    unzip_file(base_dir / "labels.zip", data_dir)
    unzip_file(base_dir / "poses.zip", data_dir)

    # ---- Unzip splits ----
    splits_target = (
        base_dir
        / "Fitness-AQA_dataset_release"
        / "OHP"
        / "Labeled_Dataset"
    )

    unzip_file(base_dir / "Splits.zip", splits_target)

    # ---- Move Fitness-AQA_dataset_release one level up (if nested) ----
    nested_fitness_dir = splits_target / "Fitness-AQA_dataset_release"

    if nested_fitness_dir.exists():
        final_fitness_dir = base_dir / "Fitness-AQA_dataset_release"

        if final_fitness_dir.exists():
            shutil.rmtree(final_fitness_dir)

        shutil.move(str(nested_fitness_dir), str(final_fitness_dir))