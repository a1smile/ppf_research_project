#!/usr/bin/env python3
"""
Generate a CSV for batch validation on segmented LM-O instance point clouds.

Default mapping in scene 000002:
000000_000000.ply -> obj_000001.ply
000000_000001.ply -> obj_000005.ply
000000_000002.ply -> obj_000006.ply
000000_000003.ply -> obj_000008.ply
000000_000004.ply -> obj_000009.ply
000000_000005.ply -> obj_000010.ply
000000_000006.ply -> obj_000011.ply
000000_000007.ply -> obj_000012.ply
"""

import argparse
import csv
from pathlib import Path

OBJ_IDS = [1, 5, 6, 8, 9, 10, 11, 12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help=r"LM-O root, e.g. F:\Research\daihongsong\data\LM-O (Linemod-Occluded)")
    parser.add_argument("--scene_id", default="000002", help="Scene folder, e.g. 000002")
    parser.add_argument("--frame_start", type=int, default=0, help="Start frame index, inclusive")
    parser.add_argument("--frame_end", type=int, default=10, help="End frame index, inclusive")
    parser.add_argument("--output_csv", default="lmo_scene000002_000000_to_000010.csv", help="Output CSV path")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    inst_root = data_root / "lmo_test_all" / "test" / "_preprocessed_lmo" / "inst_pcd_visib" / args.scene_id
    depth_root = data_root / "lmo_test_all" / "test" / args.scene_id / "depth"
    models_dir = data_root / "lmo_models" / "models_eval"

    rows = []
    missing = []

    for frame in range(args.frame_start, args.frame_end + 1):
        frame_str = f"{frame:06d}"
        depth_path = depth_root / f"{frame_str}.png"

        for obj_token, obj_id in enumerate(OBJ_IDS):
            pcd_path = inst_root / f"{frame_str}_{obj_token:06d}.ply"
            model_path = models_dir / f"obj_{obj_id:06d}.ply"

            if not pcd_path.exists():
                missing.append(str(pcd_path))
                continue
            if not model_path.exists():
                missing.append(str(model_path))
                continue

            rows.append({
                "pcd_path": str(pcd_path),
                "obj_token": obj_token,
                "frame_id": frame,
                "depth_path": str(depth_path),
                "expected_model_path": str(model_path),
                "expected_obj_id": obj_id,
            })

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pcd_path",
                "obj_token",
                "frame_id",
                "depth_path",
                "expected_model_path",
                "expected_obj_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {out_path}")
    print(f"[OK] rows={len(rows)}")
    if missing:
        print(f"[WARN] missing files: {len(missing)}")
        for p in missing[:20]:
            print("  ", p)
        if len(missing) > 20:
            print("  ...")


if __name__ == "__main__":
    main()
