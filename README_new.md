# PPF Research Project

基于 Open3D 的 PPF（Point Pair Features）6D 位姿估计与点云配准工程。

当前项目已经从“baseline + 若干增强模块”的组织方式，收敛为一个更清晰的 **三阶段鲁棒位姿估计框架**：

1. **候选召回增强（RS-MRQ）**：通过多分辨率随机移位量化哈希，提高 PPF 候选匹配的召回率。
2. **稳健投票（RobustVote）**：通过鲁棒统计加权与桶内截断，抑制噪声匹配对投票峰值的污染。
3. **非学习式多假设推理（MHSMC）**：通过 Pose Selection、Top-K 轻量精修和 SE(3) 模态聚类，从多个候选中稳健选出最终位姿。

> 当前研究结论：**KDE refinement 不再作为主流程默认模块**。它保留在代码中，但主要作为对照消融使用；默认主方法是 **PPF + RS-MRQ + RobustVote + Pose Selection + Pose Clustering**。

---

## 1. 当前方法定位

### 1.1 主方法（Ours）

当前推荐的主方法配置为：

- `configs/ablation_ours.yaml`

它包含：

- `enable_rsmrq: true`
- `enable_robust_vote: true`
- `enable_kde_refine: false`
- `pose_selection.enable: true`
- `pose_clustering.enable: true`

也就是说，主方法定义为：

```text
PPF
 + RS-MRQ
 + RobustVote
 + Pose Selection
 + Top-K Light Refine
 + SE(3) Mode Clustering
```

### 1.2 KDE 的当前定位

- `configs/ablation_ours_kde.yaml`
- 用于：**主方法 + KDE** 的对照实验
- 不再作为默认配置使用

原因是：当前实验表明，KDE 会增加计算负担，但在 hardest cases 上没有带来稳定收益，因此它更适合作为消融项，而不是主方法组成部分。

---

## 2. 三阶段框架说明

### Stage A：候选召回增强（RS-MRQ）

对应代码：

- `ppf/rsmrq_hash.py`
- `ppf/model_builder.py`
- `ppf/registration.py`

作用：

- 在 PPF 固定量化查表的基础上，引入多分辨率、随机移位、多表哈希；
- 减少由于边界量化导致的漏检；
- 为后续 voting 与 pose inference 提供更充分的候选支持。

### Stage B：稳健投票（RobustVote）

对应代码：

- `ppf/voting_robust.py`
- `ppf/registration.py`

作用：

- 用鲁棒核函数降低外点对投票的影响；
- 支持桶内 `top_m_per_bucket` 截断，减少错误峰值；
- 提高 vote peak 的可信度。

### Stage C：非学习式多假设推理（MHSMC）

对应代码：

- `ppf/pose_selection.py`
- `ppf/pose_clustering.py`
- `ppf/registration.py`

作用：

- 不再直接选最高 vote 的单个 pose；
- 先对候选进行多线索评分：
  - vote
  - inlier ratio
  - coverage
  - normal consistency
  - residual
- 然后进行 Top-K 轻量精修；
- 最后在 SE(3) 空间做模式聚类，选择最稳定的 pose mode，而不是单个 pose hypothesis。

这一步是当前项目区别于原始 PPF baseline 的核心。

---

## 3. 当前代码结构

```text
ppf_research_project/
├── configs/
│   ├── ablation_baseline.yaml
│   ├── ablation_ours.yaml
│   ├── ablation_ours_kde.yaml
│   ├── ablation_ours_no_rsmrq.yaml
│   ├── ablation_ours_no_robustvote.yaml
│   ├── ablation_mhsmc.yaml
│   └── default.yaml
│
├── ppf/
│   ├── bop_gt.py
│   ├── build_model_cache.py
│   ├── clustering.py
│   ├── io.py
│   ├── kde_refine.py
│   ├── metrics.py
│   ├── model_builder.py
│   ├── model_cache_io.py
│   ├── pose_clustering.py
│   ├── pose_selection.py
│   ├── ppf_features.py
│   ├── preprocess.py
│   ├── refine_icp.py
│   ├── registration.py
│   ├── rsmrq_hash.py
│   ├── utils.py
│   └── voting_robust.py
│
├── scripts/
│   ├── run_demo.py
│   ├── run_batch.py
│   ├── run_ablation.py
│   ├── generate_tables.py
│   ├── plot_results.py
│   └── *.csv
│
├── tests/
│   ├── test_kde_refine.py
│   ├── test_robust_vote.py
│   └── test_rsmrq.py
│
├── paper/
│   ├── experiment_report_template.md
│   └── experiment_report_template.tex
│
├── ppf_baseline_open3d.py
├── requirements.txt
└── README.md
```

---

## 4. 环境安装

推荐使用 Conda：

```bash
conda create -n ppfproj python=3.10 -y
conda activate ppfproj
pip install -r requirements.txt
```

依赖主要包括：

- `open3d`
- `numpy`
- `pyyaml`
- `pandas`
- `matplotlib`
- `tqdm`

---

## 5. 输入数据说明

### 5.1 单样本运行

你需要提供：

- 模型点云：`model.ply`
- 场景实例点云：`scene.ply`

### 5.2 批处理运行

`scripts/run_batch.py` 读取 CSV，每一行对应一个实例任务。

当前代码要求：

- 必须有列：`pcd_path`
- 若启用 `--use_bop_gt`，还需要：
  - `obj_token`
  - `frame_id`
  - `depth_path`

> 对于 LM-O/BOP 数据，推荐始终打开 `--use_bop_gt`，这样可以通过 `scene_gt.json` 自动获得真实 `obj_id` 和 GT 位姿，避免 mask 文件名与类别编号不一致导致的评估错位。

---

## 6. 快速开始

### 6.1 单对点云 Demo

```bash
python scripts/run_demo.py --config configs/ablation_ours.yaml --model path/to/model.ply --scene path/to/scene.ply --log_name demo_ours
```

### 6.2 批处理运行主方法（推荐）

```bash
python scripts/run_batch.py --csv "C:\Users\D1519\Desktop\ppf_research_project\scripts\lmo_scene000002_000000_to_000010.csv" --models_dir "F:\Research\daihongsong\data\LM-O (Linemod-Occluded)\lmo_models\models_eval" --use_bop_gt --t_scale 1.0 --out_prefix lmo_000002_ours_main --config "C:\Users\D1519\Desktop\ppf_research_project\configs\ablation_ours.yaml"
```

### 6.3 批处理运行 KDE 对照组

```bash
python scripts/run_batch.py --csv "C:\Users\D1519\Desktop\ppf_research_project\scripts\lmo_scene000002_000000_to_000010.csv" --models_dir "F:\Research\daihongsong\data\LM-O (Linemod-Occluded)\lmo_models\models_eval" --use_bop_gt --t_scale 1.0 --out_prefix lmo_000002_ours_kde --config "C:\Users\D1519\Desktop\ppf_research_project\configs\ablation_ours_kde.yaml"
```

### 6.4 批处理运行 baseline

```bash
python scripts/run_batch.py --csv "C:\Users\D1519\Desktop\ppf_research_project\scripts\lmo_scene000002_000000_to_000010.csv" --models_dir "F:\Research\daihongsong\data\LM-O (Linemod-Occluded)\lmo_models\models_eval" --use_bop_gt --t_scale 1.0 --out_prefix lmo_000002_baseline --config "C:\Users\D1519\Desktop\ppf_research_project\configs\ablation_baseline.yaml"
```

---

## 7. 配置文件说明

### `ablation_baseline.yaml`
纯 PPF baseline，不启用 RS-MRQ、RobustVote、KDE、Pose Selection、Pose Clustering。

### `ablation_ours.yaml`
当前推荐主方法配置，即本文默认方法。

### `ablation_ours_kde.yaml`
在 `ablation_ours.yaml` 的前端参数保持不变的前提下，只额外打开 KDE。用于消融，不建议作为默认配置。

### `ablation_ours_no_rsmrq.yaml`
去掉 RS-MRQ，用于验证 recall enhancement 的贡献。

### `ablation_ours_no_robustvote.yaml`
去掉 RobustVote，用于验证稳健投票的贡献。

### `ablation_mhsmc.yaml`
用于早期或单独验证 Pose Selection + Mode Clustering 逻辑，可根据实验需要保留。

### `default.yaml`
建议仅作为调试或历史兼容配置，不建议再作为论文主实验入口。

---

## 8. 当前推荐参数（主方法）

当前 `ablation_ours.yaml` 的核心参数含义如下：

### Pose Selection

- `pre_top_m_by_vote: 60`
- `candidate_top_k: 7`
- `refine_top_k: 3`
- `weights:`
  - `vote: 0.15`
  - `inlier: 0.35`
  - `coverage: 0.25`
  - `normal: 0.15`
  - `residual: 0.10`

### Pose Clustering

- `pos_thresh: 20.0`
- `rot_thresh_deg: 18.0`
- `score_weights:`
  - `size: 0.60`
  - `mean: 0.25`
  - `max: 0.15`

当前经验结论：

- `candidate_top_k = 5` 会过度收缩，导致 recall 降低；
- `candidate_top_k = 7` 比 `5` 更平衡；
- `KDE` 在当前 hardest cases 上增益不稳定，因此不作为默认主流程保留。

---

## 9. 输出结果说明

`scripts/run_batch.py` 运行后，会在 `experiments/results/...` 下生成：

- `*_batch.json`：每个样本的完整预测结果、指标、调试信息
- `summary.json`：本次批处理的汇总信息
- `logs/*.log`：完整运行日志

`*_batch.json` 中每个样本通常包含：

- `T_pred`
- `T_gt`
- `metrics`
  - `ADD`
  - `ADD_S`
  - `rotation_error_deg`
  - `translation_error`
  - `inlier_ratio`
- `stats`
  - `model_build_time`
  - `registration_time`
  - `total_time`
- `debug`
  - `robust_vote`
  - `pose_selection`
  - `pose_clustering`
  - `scene_downsample`
  - `model_downsample`

这也是后续分析 hardest cases、写论文表格和做消融的核心数据来源。

---

## 10. 实验组织建议

当前推荐的论文实验结构：

1. **Baseline**
   - `ablation_baseline.yaml`
2. **Ours（主方法）**
   - `ablation_ours.yaml`
3. **Ours + KDE**
   - `ablation_ours_kde.yaml`
4. **Ours w/o RS-MRQ**
   - `ablation_ours_no_rsmrq.yaml`
5. **Ours w/o RobustVote**
   - `ablation_ours_no_robustvote.yaml`

当前建议的论文结论方向：

- 主要性能提升来自：
  - RS-MRQ
  - RobustVote
  - Multi-hypothesis pose inference（Pose Selection + Mode Clustering）
- KDE 的作用不稳定，更适合作为消融结论，而不是主方法模块。

---

## 11. 当前论文对应的三个创新点

### 创新点 1：RS-MRQ 多分辨率随机移位匹配
提高了 PPF 候选匹配的召回率，缓解了固定量化引起的漏检问题。

### 创新点 2：RobustVote 稳健投票机制
通过鲁棒核函数和桶内截断，抑制噪声匹配对投票峰值的破坏，提高投票可靠性。

### 创新点 3：非学习式多假设位姿推理框架
不再直接选择最高 vote 单峰，而是引入多线索评分、Top-K 轻量精修和 SE(3) 模态聚类，实现从 pose hypotheses 到 pose mode 的稳健决策。

---

## 12. 注意事项

- `scripts/run_ablation.py` 仍然保留，但它更偏向早期“模块开关型”实验；
  对于当前三阶段框架，**推荐优先使用 `scripts/run_batch.py` + 明确的 YAML 配置文件** 来组织实验。
- 若你继续扩展方法，建议优先围绕：
  - hardest cases（如强对称 / 强歧义类别）
  - Pose Selection 的判别性
  - Mode Clustering 的歧义消解能力
  而不是继续增强 KDE。

---

## 13. License / Usage

本仓库用于学术研究与论文实验复现。若用于论文，请在文中明确说明：

- baseline 为 Open3D 风格 PPF 实现；
- 当前主方法为基于 PPF 的三阶段鲁棒位姿估计框架；
- KDE 只作为对照消融而非默认主方法。
