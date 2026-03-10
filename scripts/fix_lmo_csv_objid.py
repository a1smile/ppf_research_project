import os
import sys
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from ppf.bop_gt import scene_dir_from_depth_path, try_get_bop_gt_pose
from ppf.utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--t_scale", type=float, default=1.0, help="1.0 for mm, 0.001 for meters")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    for c in ["depth_path", "frame_id", "obj_token"]:
        if c not in df.columns:
            raise ValueError(f"Input CSV missing required column for BOP GT: {c}")

    obj_ids = []
    gt_ok = []
    gt_err = []
    gt_T = []

    for i, r in df.iterrows():
        depth_path = str(r["depth_path"])
        frame_id = int(r["frame_id"])
        gt_id = int(r["obj_token"])

        scene_dir = scene_dir_from_depth_path(depth_path)
        obj_id, T_gt, err = try_get_bop_gt_pose(scene_dir, frame_id, gt_id, t_scale=float(args.t_scale))

        if err is None and obj_id is not None and T_gt is not None:
            obj_ids.append(int(obj_id))
            gt_ok.append(True)
            gt_err.append("")
            gt_T.append(T_gt.reshape(-1).tolist())
        else:
            obj_ids.append(int(r["obj_id"]) if "obj_id" in df.columns else -1)
            gt_ok.append(False)
            gt_err.append(str(err))
            gt_T.append(None)

    df_out = df.copy()
    df_out["obj_id_bop"] = obj_ids
    df_out["gt_ok"] = gt_ok
    df_out["gt_err"] = gt_err
    df_out["gt_T_flat"] = gt_T

    ensure_dir(os.path.dirname(args.out_csv))
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print("Saved fixed CSV:", args.out_csv)
    print("GT ok:", sum(gt_ok), "GT fail:", len(gt_ok) - sum(gt_ok))


if __name__ == "__main__":
    main()
