import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    path = Path("data/raw/cicddos2019/UDP-training.parquet")
    print("Chargement de :", path)

    df = pd.read_parquet(path)
    print("Nombre de lignes :", len(df))
    print(df.head())