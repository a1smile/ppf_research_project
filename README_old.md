# PPF Research Project (Open3D) — Paper-Grade Reproducible Enhancements

本项目是一个**论文级可复现**的 PPF(Point Pair Features) 6D 位姿估计/配准工程，严格基于你提供的 Open3D Python baseline 结构实现，并在此基础上实现三项可插拔增强：

1. **RS–MRQ Hashing**（随机移位 + 多分辨率量化哈希，多表召回）
2. **Robust M-estimation Voting**（鲁棒统计加权投票 + top-m 桶内截断）
3. **KDE + Mean-Shift Continuous Peak Refinement**（连续峰值精化，角度嵌入处理周期性）

项目支持：
- YAML 配置系统
- 一键 demo / batch / ablation
- 自动生成实验 JSON → 汇总表格 CSV/Markdown/LaTeX
- 自动绘图（matplotlib，png+pdf）
- 自动生成论文风格实验报告模板（Markdown + LaTeX）
- 完整日志系统（包含三创新点关键统计 + 分阶段耗时）
- 单元测试（3 个创新模块）

---

## 1. Baseline 兼容性说明（重要）

你提供的 baseline 文件已作为参考保留在项目根目录：
- `ppf_baseline_open3d.py`（原样保留，未改动）

我们的实现保持 baseline 关键逻辑一致：
- `compute_pair_features(p1,n1,p2,n2) -> (f1,f2,f3,f4)`：保持同签名/退化处理
- `angle_from_transformed_point(vec)`：保持同签名/符号规则
- 模型建表 `build_ppf_model()`：仍是 O(N^2) 枚举点对、计算 alpha_m
- 查表：baseline 固定量化 + 哈希桶检索（创新1关闭时完全一致）
- 投票器：`accumulator[mr, alpha_bin] += 1`（创新2关闭时完全一致）
- 峰值选择：每个 mr 行取 argmax（创新3关闭时完全一致）
- 位姿生成：`T = inv(T_sg) * T_x * T_mg`
- 聚类：`cluster_poses(pos_thresh, rot_thresh)`（保持接口与逻辑一致）
- 默认参数与 baseline 一致（见 `configs/default.yaml`）

---

## 2. 三个创新点（数学意义 + 工程落地）

### 2.1 创新点1：RS–MRQ Hashing（随机移位 + 多分辨率量化哈希）

**Baseline：固定量化**
- 特征向量（内部采用与 baseline discretize 一致的角域变换）：
  - g1 = f1 + π (in [0,2π))
  - g2 = acos(f2) (in [0,π])
  - g3 = acos(f3) (in [0,π])
  - g4 = f4 (distance)
- baseline key：`k = floor(g / w0)`，其中 `w0=[angle_step, angle_step, angle_step, distance_step]`

**RS–MRQ：对每一层 ℓ、每张表 t 使用随机偏置 u**
- u_t^(ℓ) ~ Uniform([0,w1^(ℓ)) × ... × [0,w4^(ℓ)))
- 量化：
  Q_{w,u}(g) = floor((g + u) / w)
- 多表漏检概率（理论动机）：
  P(miss) = (1-p)^T

**工程实现：**
- `ppf/rsmrq_hash.py`：`RSMRQHashTable`
- 支持 L 层、每层 T 张表，seed 固定可复现
- 支持合并策略：
  - `union`：候选取并集
  - `count`：同一候选出现次数作为投票额外权重
- 日志输出：
  - 每层 w^(ℓ)
  - 每表 u_t^(ℓ)
  - 查询候选膨胀比例（相对 baseline 桶大小）

复杂度（建表）：
- baseline：O(N^2)
- RS–MRQ：O(L*T*N^2)（常数较大，适合做 ablation 或较小 N）

---

### 2.2 创新点2：Robust M-estimation Voting（鲁棒统计加权投票 + top-m 截断）

**动机：**
- baseline 投票是硬计数，桶碰撞/外点会堆出虚假峰。

**残差：**
- 对场景特征 g_s 与模型特征 g_m（同维同域）：
  r = || (g_s - g_m) / s ||_2
  - 默认归一化尺度 s = [π, π, π, model_diameter]（可配置）

**核函数 ω(r)（必须实现 4 种）**：
- trunc: 1(r<=τ) else 0
- gaussian: exp(-r^2/(2σ^2))
- huber: 1 if r<=τ else τ/r
- tukey: (1-(r/τ)^2)^2 if r<=τ else 0

**有界贡献：**
- X = min(B, ω(r))

**Top-m per bucket：**
- 每个 hash 桶内只取 residual 最小的 top_m 个候选参与投票，防止桶爆炸。

**工程实现：**
- `ppf/voting_robust.py`：`RobustVoter`
- 日志输出：
  - kernel 类型
  - 权重统计（min/mean/max）
  - top-m 截断比例（used/total）
  - 每个 sr 参考点的峰值统计（max/mean）

---

### 2.3 创新点3：KDE + Mean-Shift Continuous Peak Refinement

**动机：**
- baseline 取离散 bin 中心角度作为 α，精度受 angle_step 限制且抖动。

**做法：**
- 对每个 mr 行，取 top-K bins
- 收集这些 bins 内的连续样本角度 {θ_i} 与权重 {w_i}
- 角度周期性：使用嵌入 x=[cosθ, sinθ]
- KDE：
  f(x)=Σ w_i exp(-||x-x_i||^2/(2h^2))
- Mean-Shift：
  x^{t+1}= Σ w_i x_i exp(-||x^t-x_i||^2/(2h^2)) / Σ w_i exp(...)
- 日志：每次迭代 KDE 值，收敛迭代次数，refine 前后角度变化

**工程实现：**
- `ppf/kde_refine.py`：`KDEMeanShiftRefiner`
- `enable_kde_refine` 打开后，registration 会记录每个 (mr,bin) 的 (alpha, weight) 样本用于精化

复杂度（每个 mr）：
- 仅在 top-K 峰附近局部精化，样本数远小于全量投票数
- O(I * M)（I 迭代次数，M 样本数）

---

## 3. 项目结构