# 消融配置说明

本配置包按当前论文主实验组织方式整理。

## 主消融（5 组）

1. `ablation_baseline.yaml`
2. `ablation_ours_no_rsmrq.yaml`
3. `ablation_ours_no_robustvote.yaml`
4. `ablation_ours_no_pose_pipeline.yaml`
5. `ablation_ours.yaml`

## 说明

- `ablation_ours.yaml` 是当前主方法。
- 三个核心创新点分别为：
  - `RS-MRQ`
  - `Robust Vote`
  - `Pose Pipeline`（包含 `Pose Selection`、`Visibility`、`Candidate Veto`、`Pose Clustering`）
- `ablation_ours_no_pose_pipeline.yaml` 采用与 Stanford 自建数据集一致的设计思路：
  - 不直接退回旧路径；
  - 尽量保留 Top-1；
  - 关闭完整后端姿态处理流水线，仅保留最小化的 vote-only 选择。

## 备注

- `ablation_rsmrq.yaml`
- `ablation_robustvote.yaml`
- `ablation_rsmrq_robustvote.yaml`

以上配置可作为补充实验或附录实验使用，但不再作为论文主消融配置。

- `default.yaml` 建议保留为 legacy/debug 配置，不再作为论文主配置使用。







## 命令集合

按你现在的 **LMO + 5 组主消融** 方案，我建议直接用下面这套命令集。

前提默认是：

- 脚本：`scripts/run_ablation_main5.py`
- 配置目录：`configs`
- CSV：`lmo_subset_csvs/lmo_scene000002_eval_500.csv`
- 模型目录：`data/LM-O (Linemod-Occluded)/lmo_models/models_eval`

原脚本的参数接口本来就包含 `--csv`、`--models_dir`、`--config_dir`、`--use_bop_gt`、`--t_scale`、`--out_prefix`、`--num_workers`、`--limit`、`--groups` 这些选项，所以你现在这套 5 组版沿用这个调用风格是合理的。

先做 smoke test：

```
python scripts/run_ablation_main5.py \
  --csv "lmo_subset_csvs/lmo_scene000002_eval_500.csv" \
  --models_dir "data/LM-O (Linemod-Occluded)/lmo_models/models_eval" \
  --config_dir configs \
  --use_bop_gt \
  --t_scale 1.0 \
  --out_prefix smoke_ablation_main5 \
  --num_workers 4 \
  --limit 16
```

正式跑这 500 条：

```
python scripts/run_ablation_main5.py \
  --csv "lmo_subset_csvs/lmo_scene000002_eval_500.csv" \
  --models_dir "data/LM-O (Linemod-Occluded)/lmo_models/models_eval" \
  --config_dir configs \
  --use_bop_gt \
  --t_scale 1.0 \
  --out_prefix lmo_scene000002_ablation_main5 \
  --num_workers 16
```

只跑 baseline 和 ours，先看主对比：

```
python scripts/run_ablation_main5.py \
  --csv "lmo_subset_csvs/lmo_scene000002_eval_500.csv" \
  --models_dir "data/LM-O (Linemod-Occluded)/lmo_models/models_eval" \
  --config_dir configs \
  --use_bop_gt \
  --t_scale 1.0 \
  --out_prefix lmo_scene000002_main_compare \
  --num_workers 16 \
  --groups baseline,ours
```

只跑 3 个去件组，专门看消融：

```
python scripts/run_ablation_main5.py \
  --csv "lmo_subset_csvs/lmo_scene000002_eval_500.csv" \
  --models_dir "data/LM-O (Linemod-Occluded)/lmo_models/models_eval" \
  --config_dir configs \
  --use_bop_gt \
  --t_scale 1.0 \
  --out_prefix lmo_scene000002_ablation_only \
  --num_workers 16 \
  --groups ours_no_rsmrq,ours_no_robustvote,ours_no_pose_pipeline
```

单独检查 `no_pose_pipeline`：

```
python scripts/run_ablation_main5.py \
  --csv "lmo_subset_csvs/lmo_scene000002_eval_500.csv" \
  --models_dir "data/LM-O (Linemod-Occluded)/lmo_models/models_eval" \
  --config_dir configs \
  --use_bop_gt \
  --t_scale 1.0 \
  --out_prefix lmo_scene000002_no_pose_pipeline \
  --num_workers 8 \
  --groups ours_no_pose_pipeline
```

单独检查 `ours`：

```
python scripts/run_ablation_main5.py \
  --csv "lmo_subset_csvs/lmo_scene000002_eval_500.csv" \
  --models_dir "data/LM-O (Linemod-Occluded)/lmo_models/models_eval" \
  --config_dir configs \
  --use_bop_gt \
  --t_scale 1.0 \
  --out_prefix lmo_scene000002_ours \
  --num_workers 8 \
  --groups ours
```

你现在这 5 组主消融应当对应：

- `baseline`
- `ours_no_rsmrq`
- `ours_no_robustvote`
- `ours_no_pose_pipeline`
- `ours`