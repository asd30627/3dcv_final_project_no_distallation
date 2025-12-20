#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import os.path as osp
import numpy as np

def summarize_array(name, arr, max_unique=20):
    print(f"\n[{name}]")
    print(f"  type   : {type(arr)}")
    if isinstance(arr, np.ndarray):
        print(f"  shape  : {arr.shape}")
        print(f"  dtype  : {arr.dtype}")

        # 基本統計（避免 object / string）
        if arr.dtype != object and np.issubdtype(arr.dtype, np.number):
            finite = np.isfinite(arr)
            n_finite = int(finite.sum())
            n_total = arr.size
            print(f"  finite : {n_finite}/{n_total} ({n_finite/n_total*100:.2f}%)")
            if n_finite > 0:
                a = arr[finite]
                print(f"  min/max: {a.min():.6f} / {a.max():.6f}")
                print(f"  mean   : {a.mean():.6f}")
                print(f"  std    : {a.std():.6f}")
            else:
                print("  min/max: (no finite values)")

            # unique（只針對整數或小量）
            if np.issubdtype(arr.dtype, np.integer):
                u = np.unique(arr)
                if u.size <= max_unique:
                    print(f"  unique : {u.tolist()}")
                else:
                    print(f"  unique : {u.size} values (show first {max_unique}) => {u[:max_unique].tolist()}")

        else:
            print("  stats  : (skip non-numeric or object dtype)")

        # 前幾個元素（安全）
        flat = arr.ravel()
        n_show = min(10, flat.size)
        print(f"  head   : {flat[:n_show].tolist()}")

    else:
        print(f"  value  : {arr}")

def inspect_file(path):
    assert osp.exists(path), f"File not found: {path}"
    ext = osp.splitext(path)[1].lower()

    print("=" * 80)
    print(f"Inspect: {path}")
    print("=" * 80)

    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        summarize_array("npy", arr)

    elif ext == ".npz":
        z = np.load(path, allow_pickle=True)
        keys = list(z.keys())
        print(f"[npz] keys: {keys}")
        for k in keys:
            summarize_array(f"npz:{k}", z[k])

    else:
        raise ValueError(f"Unsupported extension: {ext}")

def inspect_dir(dir_path, pattern="*.npy", max_files=5):
    assert osp.isdir(dir_path), f"Not a directory: {dir_path}"
    import glob
    files = sorted(glob.glob(osp.join(dir_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {dir_path}")

    print(f"Found {len(files)} files under {dir_path} (pattern={pattern}). Show first {min(max_files, len(files))}.\n")
    for fp in files[:max_files]:
        inspect_file(fp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to .npy/.npz file OR a directory")
    ap.add_argument("--pattern", type=str, default="*.npy", help="If path is directory, glob pattern (e.g. *.npy or *.npz)")
    ap.add_argument("--max-files", type=int, default=3, help="If path is directory, inspect first N files")
    args = ap.parse_args()

    if osp.isdir(args.path):
        inspect_dir(args.path, pattern=args.pattern, max_files=args.max_files)
    else:
        inspect_file(args.path)

if __name__ == "__main__":
    main()
