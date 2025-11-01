import os
import shutil
from pathlib import Path

import kagglehub


def download(kaggle_dataset, download_path="dataset/data/base_dataset"):
    """
    Create folder of files with specified extension from a Kaggle dataset
    """

    # kagglehub.login()
    os.makedirs(download_path, exist_ok=True)

    print("Downloading dataset")
    dataset_path = kagglehub.dataset_download(kaggle_dataset)

    specified_files = list(Path(dataset_path).rglob("*_TCI.jp2"))
    specified_files.extend(list(Path(dataset_path).rglob("*.geojson")))
    print(f"Found {len(specified_files)} specified files")

    for specified_file in specified_files:
      shutil.copy2(specified_file, Path(download_path))

    return specified_files

if __name__ == "__main__":
    specified_files = download("isaienkov/deforestation-in-ukraine")