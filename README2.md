
# PPF Research Project (Open3D) — Paper-Grade Reproducible Enhancements

本项目是一个面向论文发表的 **可复现 6D 位姿估计 / 点云配准工程**，以 **PPF (Point Pair Features)** 为主干，严格兼容你提供的 `ppf_baseline_open3d.py` 流程接口与默认参数，并实现三项可插拔、可消融、可记录日志的核心创新：

1) **RS–MRQ Hashing**：随机移位 + 多分辨率量化哈希（多表召回、减漏检）
2) **Robust M-estimation Voting**：鲁棒统计加权投票（抑制桶碰撞外点、top-m 截断）
3) **KDE + Mean-Shift Peak Refinement**：连续峰值精化（打破离散 bin 上限、角度周期性处理）

此外，项目面向 **LM-O（BOP 格式）**实验，补齐了 GT 读取与 obj_id 纠正的工程链路，保证 ADD/ADD-S/RotErr/TransErr 的评估可靠。

---

## 1. Baseline 对齐说明（严格兼容）

`ppf_baseline_open3d.py` 是你的**严格基线**参考，我们的工程实现保持其“算法阶段、参数默认值、接口风格”一致，以便做严格消融与 apples-to-apples 对比。主要对齐点包括： :contentReference[oaicite:1]{index=1}

- 下采样：`sampling_leaf = 5.0`（voxel）
- 法向：KNN `k=5`
- PPF pair feature：`compute_pair_features(p1,n1,p2,n2) -> (f1,f2,f3,f4)`，并包含 `alpha_m`
- 量化：`angle_step = 12 deg`，`distance_step = 0.6 * leaf`
- 查表：`nearest_neighbor_search()`
- 投票：二维 `accumulator[mr, alpha_bin] += 1`
- 峰值：max bin（90% peak 的风格在工程实现中也保留兼容）
- 位姿：`T = inv(T_sg) * T_x * T_mg`
- 聚类：`cluster_poses(pos_thresh, rot_thresh)`
- baseline 默认的 `matching_rate` 评估思想仍提供

> 重要：三创新点默认 **全部关闭**，此时行为应与 baseline 尽可能一致（除了工程组织、日志系统与输出格式）。

---

## 2. 项目目录结构

ppf_research_project/
 │
 ├── README.md
 ├── requirements.txt
 │
 ├── configs/
 │   ├── default.yaml
 │   ├── ablation_baseline.yaml
 │   ├── ablation_no_rsmrq.yaml
 │   ├── ablation_no_robustvote.yaml
 │   ├── ablation_no_kde.yaml
 │   ├── ablation_full.yaml
 │
 ├── ppf/
 │   ├── **init**.py
 │   ├── io.py
 │   ├── preprocess.py
 │   ├── ppf_features.py
 │   ├── model_builder.py
 │   ├── rsmrq_hash.py
 │   ├── voting_robust.py
 │   ├── kde_refine.py
 │   ├── clustering.py
 │   ├── registration.py
 │   ├── refine_icp.py
 │   ├── metrics.py
 │   ├── bop_gt.py
 │   ├── utils.py
 │
 ├── scripts/
 │   ├── run_demo.py
 │   ├── run_batch.py
 │   ├── run_ablation.py
 │   ├── fix_lmo_csv_objid.py
 │   ├── generate_tables.py
 │   ├── plot_results.py
 │
 ├── experiments/
 │   ├── results/
 │   ├── logs/
 │   ├── tables/
 │   ├── figures/
 │
 ├── paper/
 │   ├── experiment_report_template.md
 │   ├── experiment_report_template.tex
 │
 └── tests/
 ├── test_rsmrq.py
 ├── test_robust_vote.py
 ├── test_kde_refine.py

```
---

## 3. 安装（Windows 推荐）

### 3.1 Conda 环境
```bat
conda create -n ppfproj python=3.10 -y
conda activate ppfproj
pip install -r requirements.txt
```

## 3.依赖说明

- Open3D：点云 I/O、KDTree、ICP
- NumPy：数值计算
- PyYAML：配置系统
- pandas：实验表格
- matplotlib：绘图
- tqdm：进度条

------

## 4. 快速运行

### 4.1 单对点云 Demo

```
python scripts\run_demo.py --config configs\default.yaml --model path\to\model.ply --scene path\to\scene.ply
```

输出：

- 日志：`experiments/logs/demo.log`
- 结果 JSON：`experiments/results/demo.json`

### 4.2 批处理（直接吃 CSV）

你已生成实例 CSV（如 `experiment_list_visib.csv`），包含：
 `pcd_path, depth_path, frame_id, obj_token ...`

推荐用 BOP GT 自动纠正 obj_id：

```
python scripts\run_batch.py ^
  --config configs\ablation_full.yaml ^
  --csv F:\...\experiment_list_visib.csv ^
  --models_dir F:\...\models ^
  --use_bop_gt ^
  --t_scale 1.0 ^
  --out_prefix lmo_full
```

说明：

- LM-O/BOP 里 `mask` 文件名经常是 `frame_gtid.png`，第二段 `gtid` 是该帧 `scene_gt.json` 列表索引，不是 obj 类别 id。
- `--use_bop_gt` 会读取 `scene_gt.json` 获取 **真实 obj_id 和 T_gt**，避免“模型选错/评估错位”。

### 4.3 离线修正 CSV（可选）

把 `obj_id_bop` 和 `gt_T_flat` 写进 CSV，方便后续完全脱离 json 批跑：

```
python scripts\fix_lmo_csv_objid.py ^
  --in_csv F:\...\experiment_list_visib.csv ^
  --out_csv F:\...\experiment_list_visib_fixed.csv ^
  --t_scale 1.0
```

------

## 5. Ablation（自动跑 5 种方法 + 自动出表 + 自动画图）

### 5.1 一键 ablation

```
python scripts\run_ablation.py --config configs\default.yaml --model path\to\model.ply --scene path\to\scene.ply --repeat 10
```

默认方法集（固定顺序）：

- `baseline`：三创新点全关
- `plus_rsmrq`：只开 RS–MRQ
- `plus_robustvote`：只开 RobustVote
- `plus_kde`：只开 KDE refine
- `full`：三者全开

每次运行输出：

- `experiments/results/{method}_run{i}.json`
- `experiments/logs/{method}_run{i}.log`

### 5.2 生成表格

```
python scripts\generate_tables.py
```

输出：

- `experiments/tables/ablation_results.csv`
- `experiments/tables/ablation_results.md`
- `experiments/tables/ablation_results.tex`（论文表格，包含 `\toprule` 等，并对最优项加粗）

### 5.3 生成图像（png+pdf）

```
python scripts\plot_results.py
```

输出到：

- `experiments/figures/*.png`
- `experiments/figures/*.pdf`

------

## 6. 三个创新点：数学含义、工程落地、复杂度与消融

### 6.1 创新点1：RS–MRQ Hashing（随机移位 + 多分辨率量化哈希）

**改动位置**：baseline 的“特征量化 + hash key + 查表候选集生成”。

baseline 固定量化：

- key = floor(g / w)，其中 g 是 PPF 变换后的特征（角度/距离分量），w 为固定 bin 宽。

问题：

- 真匹配点对可能落在 bin 边界两侧 → 漏召回。

RS–MRQ：

- 多层分辨率：`L` 层，w_levels[ℓ]
- 每层 `T_tables` 张随机移位表：u ~ Uniform([0,w))
- key_{ℓ,t}(g) = floor((g + u_{ℓ,t}) / w_{ℓ})

候选融合：

- `union`：取并集
- `count`：同一候选出现次数作为后续投票权重（与创新2组合更强）

**复杂度**：

- baseline build O(N^2)
- RS–MRQ build O(L*T*N^2)，查询同理增加 L*T 倍候选访问
- 通过 `merge_mode=count` + RobustVote 的 top-m 可以控制后端开销

**消融**：

- `ablation_no_rsmrq.yaml`：关闭 `enable_rsmrq`
- `ablation_full.yaml`：开启 `enable_rsmrq`

------

### 6.2 创新点2：Robust M-estimation Voting（鲁棒统计加权投票）

**改动位置**：baseline 的“投票累加器更新”阶段。

baseline 是硬计数：

- accumulator[mr, bin] += 1

鲁棒投票：

- residual r = || normalize(f_s - f_m) ||_2
- weight ω(r) 采用核函数（trunc / gaussian / huber / tukey）
- 单次投票贡献：X = min(B, ω(r))
- accumulator[mr, bin] += X

桶碰撞控制（关键工程点）：

- 每个 hash 桶的 matches 只取 top_m_per_bucket 个最优 residual，再投票

**核函数支持**（必须实现）：

- trunc, gaussian, huber, tukey

**复杂度**：

- 每次桶访问，增加 residual 计算与排序
- top-m 截断将最坏情况从“桶内无限碰撞”变成 O(m log m) 或 O(m)（可用 partial sort）

**消融**：

- `ablation_no_robustvote.yaml`

------

### 6.3 创新点3：KDE + Mean-Shift Peak Refinement（连续峰值精化）

**改动位置**：baseline 的“离散 alpha bin 取 argmax → 输出角度”阶段。

baseline：

- bj = argmax accumulator[mr,:]
- θ = (bj+0.5)*angle_step - π

问题：

- 精度被 bin 宽限制，且 bin 抖动明显。

连续精化：

- 收集 top_k 个峰附近的角度样本 {θ_i} 及权重 {w_i}
- 角度周期性用嵌入表示：x_i=[cosθ_i, sinθ_i]
- KDE 密度：f(x)= Σ w_i exp( -||x-x_i||^2 / (2h^2) )
- Mean-Shift 更新保证密度单调上升（日志中记录每次 f(x)）

输出 θ_refined 替代 bin 中心角度。

**复杂度**：

- 局部 top-k 样本规模 M，小于全局
- 迭代 I 次，O(I*M)

**消融**：

- `ablation_no_kde.yaml`

------

## 7. YAML 参数总表（论文复现必备）

| Key                            | Type  | Default         | Meaning                                  |
| ------------------------------ | ----- | --------------- | ---------------------------------------- |
| seed                           | int   | 0               | 全局随机种子（也用于 RS–MRQ 偏置可复现） |
| sampling_leaf                  | float | 5.0             | voxel downsample leaf（单位同点云）      |
| normal_k                       | int   | 5               | 法向 KNN 数                              |
| angle_step_deg                 | float | 12.0            | alpha 离散步长（deg）                    |
| distance_step_ratio            | float | 0.6             | distance_step = ratio * leaf             |
| scene_ref_sampling_rate        | int   | 20              | 场景参考点步进采样                       |
| pos_thresh                     | float | 0.005           | 聚类平移阈值                             |
| rot_thresh_deg                 | float | 30.0            | 聚类旋转阈值（deg）                      |
| registration_runs              | int   | 1               | 重复注册次数（测时间均值）               |
| visualize                      | bool  | false           | 是否可视化                               |
| enable_rsmrq                   | bool  | false           | 开启 RS–MRQ Hashing                      |
| rsmrq.L                        | int   | 2               | 多分辨率层数                             |
| rsmrq.T_tables                 | int   | 4               | 每层随机移位表数                         |
| rsmrq.w_levels                 | list  | see yaml        | 每层 w=(rad,rad,rad,dist)                |
| rsmrq.merge_mode               | str   | union           | union / count                            |
| rsmrq.seed                     | int   | 0               | 偏置随机种子                             |
| enable_robust_vote             | bool  | false           | 开启鲁棒投票                             |
| robust_vote.kernel             | str   | gaussian        | trunc/gaussian/huber/tukey               |
| robust_vote.sigma              | float | 0.6             | 高斯 sigma                               |
| robust_vote.tau                | float | 1.0             | 截断/Huber/Tukey 阈值                    |
| robust_vote.B                  | float | 1.0             | 单次投票贡献上界                         |
| robust_vote.top_m_per_bucket   | int   | 50              | 桶内 top-m 截断                          |
| robust_vote.normalize_feature  | bool  | true            | residual 归一化尺度                      |
| enable_kde_refine              | bool  | false           | 开启 KDE 精化                            |
| kde_refine.use_angle_embedding | bool  | true            | cos/sin 嵌入处理周期性                   |
| kde_refine.top_k               | int   | 3               | top-k bin 收集样本                       |
| kde_refine.bandwidth_h         | float | 0.35            | KDE 带宽                                 |
| kde_refine.max_iter            | int   | 50              | 最大迭代                                 |
| kde_refine.tol                 | float | 1e-4            | 收敛阈值                                 |
| icp_refine.enable              | bool  | false           | 可选 ICP refine                          |
| icp_refine.max_iter            | int   | 30              | ICP 迭代                                 |
| icp_refine.distance_threshold  | float | 5.0             | ICP 阈值（单位同点云）                   |
| bop_gt.t_scale                 | float | 1.0             | BOP GT 平移缩放（mm->m 等）              |
| output.*                       | str   | experiments/... | 输出路径                                 |

------

## 8. 输出文件说明（如何引用到论文）

- 单次运行（demo/batch/ablation）：

  - 结果：`experiments/results/*.json`
  - 日志：`experiments/logs/*.log`

- Ablation 汇总表：

  - `experiments/tables/ablation_results.tex`
     直接在 LaTeX 里：

    ```
    \input{../experiments/tables/ablation_results.tex}
    ```

- 图：

  - `experiments/figures/*.pdf`（论文首选）
  - `experiments/figures/*.png`（调试/预览）

- 报告模板：

  - `paper/experiment_report_template.md`
  - `paper/experiment_report_template.tex`

------

## 9. 常见错误排查（LM-O/点云 PPF 高频坑）

### 9.1 单位不一致（mm vs m）

baseline 的默认 leaf=5.0、ICP 阈值=5.0 很像 “mm”。 ppf_baseline_open3d
 如果你的点云是米（m），需同步修改：

- sampling_leaf（例如 0.005）
- icp_refine.distance_threshold
- bop_gt.t_scale（m 则为 0.001）

### 9.2 obj_token ≠ obj_id（LM-O/BOP 常见）

mask 文件名经常是 `000000_000003.png`：

- 后面的 `000003` 更可能是 `gt_id`（scene_gt.json 列表索引）
- 真实 obj 类别 id 在 `scene_gt.json` 的 `obj_id`

解决：

- batch 运行时加 `--use_bop_gt`
- 或离线跑 `fix_lmo_csv_objid.py`

### 9.3 候选爆炸（RS–MRQ 开启）

- RS–MRQ 多表会增加候选数（日志中会输出 inflation）
- 解决：使用 RobustVote 的 `top_m_per_bucket` 截断，必要时减小 `T_tables` 或减少 L

### 9.4 法向不稳（反光/噪声）

你当前先做“只读点云版本、法向后续模块”是合理的工程节奏。
 但一旦开启 PPF 注册，法向质量直接影响 PPF 特征与 alpha。

建议：

- normal_k 增大（例如 15~30）
- 后续引入更稳的法向估计/方向一致性（或使用你计划的 patch/分割策略）

------

## 10. 复杂度分析（论文写法可直接用）

设模型点数 N，场景点数 M。

- Baseline build：O(N^2)
- Baseline query + vote：与场景邻域点对数量相关，最坏可近似 O(M * K * avg_bucket_size)

RS–MRQ：

- build：O(L*T*N^2)
- query：候选访问增长约 L*T 倍（日志输出 inflation）

RobustVote：

- 桶内 residual + top-m 截断：将桶碰撞的最坏情况控制为 O(m log m)（或 partial sort 近似 O(m)）

KDE refine：

- 对少量 top-k 局部样本做 Mean-Shift：O(I*M_local)

------

## 11. 如何撰写论文实验段落（快速提示）

- 用 `paper/experiment_report_template.tex` 作为骨架
- Ablation 表直接 `\input{../experiments/tables/ablation_results.tex}`
- 图引用 `experiments/figures/*.pdf`
- 每个创新点：用 “off/on 对比” + “候选数/权重统计/迭代收敛日志” 做可解释分析

------

## 12. 引用本项目结果（推荐写法）

在论文中建议记录：

- commit hash（或 zip 版本号）
- config 文件名（如 `configs/ablation_full.yaml`）
- 数据集版本（LM-O/BOP 的目录结构）
- 机器配置（CPU/GPU/RAM/OS/Python）

------

## License & Disclaimer

本项目用于科研对比与复现，baseline 部分强调结构一致性与可读性，未做极致性能优化。

```
---

如果你希望 README 里把 **“三创新点的公式”**写得更像论文（含更严格符号、概率界/鲁棒统计界/MeanShift 单调性说明的文字版），我也可以在 README 的第 6 节下面再加一个 **“Theory Notes（可直接复制进方法章节）”** 小节，把公式写成 LaTeX，并严格对应你 baseline 的 `g = [f1+pi, acos(f2), acos(f3), f4]` 这套变换。
::contentReference[oaicite:3]{index=3}
```

>