import argparse
from pathlib import Path
from collections import Counter, defaultdict
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import scan_images
from .utils import set_seed, ensure_dir, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Unzipped dataset root folder")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    items = scan_images(data_dir)
    df = pd.DataFrame(items, columns=["path","label"])

    # Clean labels: keep as-is but stable ordering by frequency then name
    counts = df["label"].value_counts().to_dict()
    labels_sorted = sorted(counts.keys(), key=lambda k: (-counts[k], k))
    label_to_id = {lab:i for i,lab in enumerate(labels_sorted)}
    id_to_label = {i:lab for lab,i in label_to_id.items()}

    # Stratified split
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["label"]
    )
    # val is portion of remaining
    val_ratio = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=args.seed, stratify=train_df["label"]
    )

    train_df.to_csv(out_dir/"train.csv", index=False)
    val_df.to_csv(out_dir/"val.csv", index=False)
    test_df.to_csv(out_dir/"test.csv", index=False)

    save_json({
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "counts": counts,
        "seed": args.seed,
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }, out_dir/"labels.json")

    print("Saved splits to:", out_dir)
    print("Class counts:", counts)

if __name__ == "__main__":
    main()
