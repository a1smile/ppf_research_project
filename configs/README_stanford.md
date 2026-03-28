# Stanford 配置说明

本目录提供 Stanford 数据集实验所需的配置文件，并统一采用**相同的预处理、采样和几何参数**，仅对算法模块进行开启/关闭，以保证消融实验的可比性。

---

## 1. 实验方法定义

### Baseline
原始 PPF 流水线，不启用以下模块：

- RS-MRQ
- Robust Vote
- Backend hypothesis consolidation

对应配置：

- `baseline_stanford.yaml`

---

### Front-end only
仅启用前端增强：

- RS-MRQ
- Robust Vote

最终位姿**仅按最高 vote 的候选直接选出**，不使用后端多假设整合能力，即不使用：

- 多指标 pose selection
- candidate veto
- light refine
- pose clustering

对应配置：

- `ablation_no_pose_pipeline_stanford.yaml`

> 说明：这里的 “no pose pipeline” 实际含义是  
> **without backend hypothesis consolidation**，  
> 而不是简单关闭 `pose_selection` 入口。  
> 这样做是为了避免退回旧的 O(N²) clustering 路径，保证实验可运行且定义合理。

---

### Ours (Full)
完整方法，启用：

- RS-MRQ
- Robust Vote
- Backend hypothesis consolidation

其中 backend hypothesis consolidation 包括：

- pose selection
- top-k light refine
- pose clustering

对应配置：

- `ablation_ours_stanford.yaml`

---

### 其它消融

#### No RS-MRQ
关闭 RS-MRQ，其它与完整方法保持一致。

对应配置：

- `ablation_no_rsmrq_stanford.yaml`

#### No Robust Vote
关闭 Robust Vote，其它与完整方法保持一致。

对应配置：

- `ablation_no_robust_vote_stanford.yaml`

---

## 2. 推荐实验分组

建议最终论文/报告按以下分组展示：

| Method         | Description                                             |
| -------------- | ------------------------------------------------------- |
| Baseline       | Original PPF                                            |
| No RS-MRQ      | Full method without RS-MRQ                              |
| No Robust Vote | Full method without robust voting                       |
| Front-end only | RS-MRQ + Robust Vote, final pose selected by top-1 vote |
| Ours (Full)    | RS-MRQ + Robust Vote + backend hypothesis consolidation |

---

## 3. 运行命令

以下命令默认在项目根目录执行。

### 3.1 Baseline
```bash
python scripts/run_batch_stanford.py --config configs/baseline_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_baseline --num_workers 8 --rebuild_cache
```

### 3.2 No RS-MRQ

> 需要重建 cache，因为模型检索结构发生变化。

```
python scripts/run_batch_stanford.py --config configs/ablation_no_rsmrq_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_no_rsmrq --num_workers 8 --rebuild_cache
```

### 3.3 No Robust Vote

> 不需要重建 cache。

```
python scripts/run_batch_stanford.py --config configs/ablation_no_robust_vote_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_no_robust_vote --num_workers 8
```

### 3.4 Front-end only

> 不需要重建 cache。
>  这是替代旧 no-pose-pipeline 错误路径的正确版本。

```
python scripts/run_batch_stanford.py --config configs/ablation_no_pose_pipeline_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_frontend_only --num_workers 8
```

### 3.5 Ours (Full)

> 不需要重建 cache。

```
python scripts/run_batch_stanford.py --config configs/ablation_ours_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_ours_full --num_workers 8
```

### 3.6 Debug Fast

> 用于快速检查流程是否正常，不用于最终结果汇报。
>  该配置采用更稀的参考点采样，并使用 front-end only 的安全路径，避免回退到旧的 O(N²) clustering。

```
python scripts/run_batch_stanford.py --config configs/debug_fast_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_debug_fast --num_workers 8
```

------

## 4. 关于 cache 的说明

以下情况需要使用 `--rebuild_cache`：

1. 第一次运行 Stanford 实验
2. 修改了模型特征相关配置，例如：
   - `enable_rsmrq`
   - `sampling_leaf`
   - `angle_step_deg`
   - 其它会影响模型哈希/检索结构的参数

以下情况通常**不需要**重建 cache：

- 只修改 robust vote 参数
- 只修改 backend pose selection / clustering 参数
- 只修改输出目录或可视化参数

------

## 5. 结果目录建议

建议结果按如下方式保存，避免新旧实验混淆：

```
experiments/results/
  stanford_baseline/
  stanford_no_rsmrq/
  stanford_no_robust_vote/
  stanford_frontend_only/
  stanford_ours_full/
  stanford_debug_fast/
```

------

## 6. 备注

- Baseline 很慢是预期现象，因为其本质上接近原始 PPF 的全量候选生成与投票过程。
- Full 方法的主要加速来自：
  - RS-MRQ 带来的高效候选检索
  - Robust Vote 带来的受控投票
- Backend hypothesis consolidation 的主要作用是：
  - 提升候选决策质量
  - 解决多峰歧义
  - 提高 hardest cases 的鲁棒性

因此，建议将 Stanford 消融的核心结论组织为：

1. RS-MRQ 负责提升候选召回并显著改善运行效率
2. Robust Vote 负责抑制噪声候选、提升投票稳定性
3. Backend hypothesis consolidation 负责从多候选中选择最可信的 pose mode