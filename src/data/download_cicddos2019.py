import kagglehub
import shutil
from pathlib import Path


def download_cicddos2019():
    print("Téléchargement du dataset CIC-DDoS2019 via kagglehub...")
    path = kagglehub.dataset_download("dhoogla/cicddos2019")
    print("Dataset téléchargé dans :", path)

    src = Path(path)
    dst = Path("data/raw/cicddos2019")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"[INFO] Le dossier {dst} existe déjà, aucun déplacement.")
    else:
        print(f"Copie des fichiers vers {dst}...")
        shutil.copytree(src, dst)

    print("✅ Dataset prêt dans data/raw/cicddos2019")


if __name__ == "__main__":
    download_cicddos2019()