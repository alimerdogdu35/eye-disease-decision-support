import argparse
from pathlib import Path
import pandas as pd
from PIL import Image

def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()  # dosya bütünlüğünü kontrol eder
        # verify sonrası yeniden açmak bazen gerekir ama burada yeterli
        return True
    except Exception:
        return False

def clean_csv(csv_path: Path, out_path: Path):
    df = pd.read_csv(csv_path)
    ok_rows = []
    bad = 0

    for i, row in df.iterrows():
        p = row["path"]
        if Path(p).exists() and is_valid_image(p):
            ok_rows.append(row)
        else:
            bad += 1

    out_df = pd.DataFrame(ok_rows)
    out_df.to_csv(out_path, index=False)
    print(f"{csv_path.name}: total={len(df)} ok={len(out_df)} bad={bad} -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ["train.csv", "val.csv", "test.csv"]:
        clean_csv(splits_dir / name, out_dir / name)

   
    (out_dir / "labels.json").write_text((splits_dir / "labels.json").read_text(encoding="utf-8"), encoding="utf-8")
    print("labels.json copied.")

if __name__ == "__main__":
    main()
