# Stanford 配置文件说明（Stanford config set）

本配置集基于 `ablation_ours_stanford.yaml` 派生，专门用于 **Stanford Retrieval 数据集** 的实验。  
所有配置均保持 **Stanford 数据尺度参数一致**，只对方法模块做开关控制，用于主实验对比和消融实验。

---

## 1. 配置文件说明

### `baseline_stanford.yaml`
Stanford 数据集上的基础基线配置。  
关闭以下增强模块：

- RS-MRQ
- Robust Vote
- Pose 后处理流水线（Pose Selection / Pose Clustering）

用途：
- 作为 Stanford 数据集上的 **baseline**
- 用于和 full 版本进行主对比

---

### `ablation_no_rsmrq_stanford.yaml`
在 full 配置基础上，去掉 **RS-MRQ**。

用途：
- 验证 **创新点 1：RS-MRQ** 的作用

---

### `ablation_no_robust_vote_stanford.yaml`
在 full 配置基础上，去掉 **Robust Vote**。

用途：
- 验证 **创新点 2：Robust Vote** 的作用

---

### `ablation_no_pose_pipeline_stanford.yaml`
在 full 配置基础上，去掉 **Pose 后处理流水线**，包括：

- Pose Selection
- Pose Clustering

说明：
- `visibility`
- `candidate_veto`

这两个模块本身依赖 `pose_selection`，因此关闭 `pose_selection` 后，它们也不会生效。

用途：
- 验证 **创新点 3：Pose 后处理流水线** 的作用

---

### `ablation_ours_stanford.yaml`
Stanford 数据集上的完整方法（full 配置）。

包含：
- RS-MRQ
- Robust Vote
- Pose Selection
- Visibility / Candidate Veto
- Pose Clustering

用途：
- 作为 Stanford 数据集上的 **最终 full 结果**

---

### `debug_fast_stanford.yaml`
更快的调试配置，用于快速检查流程是否正常。

特点：
- scene 采样更稀疏
- 关闭 pose 后处理流水线
- 运行更快，但不是正式结果

用途：
- 调试
- 快速验证路径、缓存和输出是否正常

---

## 2. 推荐实验分组

### 主对比实验（Main comparison）
建议至少跑这两组：

- `baseline_stanford.yaml`
- `ablation_ours_stanford.yaml`

用于展示：
- 基础方法性能
- 完整方法性能

---

### 消融实验（Ablation study）
建议保留这三组核心消融：

- `ablation_no_rsmrq_stanford.yaml`
- `ablation_no_robust_vote_stanford.yaml`
- `ablation_no_pose_pipeline_stanford.yaml`

对应三个创新点：

1. **RS-MRQ**
2. **Robust Vote**
3. **Pose 后处理流水线**

---

## 3. 缓存（cache）重建说明

当以下模型侧参数发生变化时，建议重建 cache：

- `enable_rsmrq`
- `enable_robust_vote`
- `sampling_leaf`
- `normal_k`
- `angle_step_deg`
- `distance_step_ratio`
- `adaptive_downsample*`
- `rsmrq`
- `robust_vote`

### 一般需要重建 cache 的配置
- `baseline_stanford.yaml`
- `ablation_no_rsmrq_stanford.yaml`
- `ablation_no_robust_vote_stanford.yaml`

### 一般不需要重建 cache 的配置
- `ablation_no_pose_pipeline_stanford.yaml`

### full 配置
- `ablation_ours_stanford.yaml` 已经有结果时，一般不需要重复跑
- 如果你修改了 full 配置内容，再考虑重建

---

## 4. 运行命令

### 4.1 运行 baseline
```bash
python scripts/run_batch_stanford.py --config configs/baseline_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_baseline --num_workers 8 --rebuild_cache
```

### 4.2 运行 no RS-MRQ

```bash
python scripts/run_batch_stanford.py --config configs/ablation_no_rsmrq_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_no_rsmrq --num_workers 8 --rebuild_cache
```

### 4.3 运行 no Robust Vote

```bash
python scripts/run_batch_stanford.py --config configs/ablation_no_robust_vote_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_no_robust_vote --num_workers 8 --rebuild_cache
```

### 4.4 运行 no Pose Pipeline

```bash
python scripts/run_batch_stanford.py --config configs/ablation_no_pose_pipeline_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_no_pose_pipeline --num_workers 8
```

### 4.5 运行 full

```bash
python scripts/run_batch_stanford.py --config configs/ablation_ours_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_ours_full --num_workers 8
```

### 4.6 运行 debug-fast

```bash
python scripts/run_batch_stanford.py --config configs/debug_fast_stanford.yaml --csv stanford_retrieval_batch_all
```

## 5. 当前推荐实验顺序

推荐按下面顺序执行：

1. `baseline_stanford`
2. `ablation_no_rsmrq_stanford`
3. `ablation_no_robust_vote_stanford`
4. `ablation_no_pose_pipeline_stanford`

说明：

- `ablation_ours_stanford` 当前已经有 full 结果时，可以先不重复跑
- 等第二个数据集实验完成后，再统一整理表格

------

## 6. 备注

- Stanford 数据集中存在高对称物体，因此建议同时关注：
  - `ADD`
  - `ADD-S`
- 对称物体上出现普通姿态误差较大、但 `ADD-S` 较小的情况是正常现象
- 正式写论文时，建议把 Stanford 结果作为：
  - 补充实验
  - 泛化验证
  - 自建/整理测试集实验
