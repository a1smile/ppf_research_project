# 方法说明文档（中文）

## 1. 文档定位

本文档用于说明当前项目中主方法的技术路线、模块职责、实验定位与论文写作映射关系。它面向研究与论文撰写，不是面向最终用户的安装说明。

当前项目的主方法已经不再包含 KDE refinement。根据现阶段实验结果，KDE 在统一前端参数后没有带来稳定、可重复的收益，反而增加了运行开销。因此，当前默认主方法定义为：

**PPF + RS-MRQ + Robust Vote + Pose Selection + Top-K Light Refine + Pose Clustering**

推荐配置文件命名：
- `ablation_baseline.yaml`
- `ablation_ours.yaml`
- `ablation_ours_kde.yaml`
- `ablation_ours_no_rsmrq.yaml`
- `ablation_ours_no_robustvote.yaml`

其中：
- `ablation_ours.yaml`：当前主方法
- `ablation_ours_kde.yaml`：仅作为 “+KDE” 对照消融

---

## 2. 研究问题与方法动机

传统 PPF 的主要流程是：构造点对特征、查表得到候选对应、对姿态投票、选取最高投票峰值作为最终结果。

该流程的核心问题不是“生成不出 pose”，而是：

- 候选姿态数量多；
- 多个错误姿态可能拥有较高 vote；
- 对称物体或局部几何重复时，容易出现 180° ambiguity；
- 单纯依赖最高 vote peak 容易选错最终 pose。

因此，本研究把重点从“更重的后端 refine”转移到：

**如何从大量候选姿态中，稳健地选出正确模式（mode），而不是错误峰值。**

---

## 3. 总体框架

当前方法是一个三阶段鲁棒位姿估计框架：

### 阶段一：匹配增强（RS-MRQ）
目标：提升候选对应的召回率，使正确匹配不容易在前端被遗漏。

### 阶段二：鲁棒投票（Robust Vote）
目标：降低噪声匹配对 vote peak 的干扰，使高分候选更可信。

### 阶段三：多假设推理（Pose Selection + Mode Clustering）
目标：解决 PPF 的多峰歧义问题，不再直接使用单个最高票峰值，而是在候选集合中通过多线索评分、轻量 refine 与 SE(3) 模式合并，得到更稳定的最终姿态。

整体流程如下：

1. 输入模型点云与场景点云；
2. 执行 PPF 特征提取与候选匹配；
3. 使用 RS-MRQ 扩展匹配召回；
4. 使用 Robust Vote 生成更鲁棒的姿态候选；
5. 对候选 pose 进行多线索几何评分；
6. 仅保留 Top-K 候选做轻量 refine；
7. 在 SE(3) 空间进行模式聚类；
8. 选择最佳 pose mode 对应的结果作为最终输出。

---

## 4. 三阶段模块说明

### 4.1 RS-MRQ（匹配增强）

RS-MRQ 的作用不是直接决定最终 pose，而是：

- 提高前端候选对应的 recall；
- 缓解因采样稀疏、法向不稳定或局部噪声导致的匹配缺失；
- 为后续投票与候选生成提供更丰富的正确姿态线索。

论文贡献表达可写为：

> 提出一种多分辨率匹配策略，以提升 PPF 在稀疏、噪声条件下的匹配召回率。

### 4.2 Robust Vote（鲁棒投票）

传统投票方法的问题在于：

- 错误匹配也会参与投票；
- 高噪声条件下容易形成错误 peak；
- recall 提高的同时，precision 会明显下降。

Robust Vote 的目标是：

- 对投票样本加权；
- 抑制由错误对应引起的假峰；
- 提高真正姿态峰值的相对可信度。

论文贡献表达可写为：

> 引入一种鲁棒投票机制，以抑制错误匹配带来的假峰并提高投票可靠性。

### 4.3 多假设推理框架

这是当前工作的核心。

#### 4.3.1 Pose Selection

候选姿态不再只按 vote 排序，而是综合考虑：

- vote score
- inlier ratio
- coverage
- normal consistency
- residual

其本质是一个非学习的多线索几何评分过程，用于判断：

- 哪个候选 pose 更像真实 pose；
- 哪个高 vote 峰值实际上只是错误峰。

当前阶段性推荐权重为：

- vote: 0.15
- inlier: 0.35
- coverage: 0.25
- normal: 0.15
- residual: 0.10

当前推荐：
- `candidate_top_k = 7`
- `refine_top_k = 3`

这组参数在当前实验中表现出较好的平衡性：
- 比 `candidate_top_k = 5` 更不容易误杀正确模式；
- 比 `candidate_top_k = 10` 更能抑制噪声候选。

#### 4.3.2 Top-K Light Refine

只对少量高置信候选进行轻量 refine，其目的不是从完全错误的 pose 中“救出正确解”，而是：

- 对接近正确的候选做小范围修正；
- 提高候选之间的几何可分性；
- 为后续 clustering 提供更稳定的输入。

#### 4.3.3 Pose Clustering / Mode Consolidation

该模块的本质不是“简单聚类”，而是：

- 在 SE(3) 空间中寻找候选姿态的模式（mode）；
- 从“选择单一 pose”转变为“选择最可信的 pose 模式”；
- 解决多峰、歧义和对称结构下的错误峰值问题。

cluster score 当前更偏向 cluster size，而不是单峰 max score。当前推荐权重为：

- size: 0.60
- mean: 0.25
- max: 0.15

这一设计体现了本文的核心思想：

> 不是选最高票的 pose，而是选最可信的 pose 模式。

---

## 5. 当前主方法定义

当前项目主方法定义为：

**Ours = PPF + RS-MRQ + Robust Vote + Pose Selection + Top-K Light Refine + Pose Clustering**

不包括 KDE。

原因如下：

1. 在统一前端参数后，KDE 没有带来稳定、显著的提升；
2. KDE 对 hardest cases 的 170°~180° ambiguity 基本无效；
3. KDE 会增加额外运行时间与调用次数；
4. KDE 的收益不如 hypothesis selection 与 mode consolidation 明确。

因此，KDE 当前仅保留为：

- 对照消融模块；
- 可选 refine 组件；
- 不作为主流程默认模块。

---

## 6. 当前实验结论（阶段性）

经过多轮调参，当前方法已经表现出如下特点：

### 优点
- 能显著改善一部分样本的 rotation error；
- inlier ratio 相比旧版有明显提升；
- 在若干类别上表现出较高稳定性；
- clustering 已不再是摆设，而是能真正合并 pose 模式。

### 主要问题
- hardest cases 仍集中在少数类别，例如：
  - obj_id = 8
  - obj_id = 10
  - obj_id = 11
  - obj_id = 12
- 这些类别中仍存在明显的 180° ambiguity；
- 说明当前瓶颈已经不是“是否 refine”，而是“错误模式仍能进入最终决策”。

因此，当前阶段的研究重点应继续放在：

- hypothesis selection
- mode consolidation
- hardest cases 分析

而不是继续投入过多精力在 KDE 上。

---

## 7. 论文创新点映射

当前工作可归纳为三个阶段性创新点：

### 创新点 1：RS-MRQ
提出一种多分辨率匹配增强策略，以提升 PPF 在稀疏与噪声条件下的匹配召回率。

### 创新点 2：Robust Vote
提出一种鲁棒投票机制，以抑制错误匹配带来的假峰并提高投票可靠性。

### 创新点 3：多假设姿态推理框架
提出一种非学习的多假设姿态推理框架，通过多线索评分、Top-K 轻量 refine 和 mode-based clustering 解决 PPF 中的多峰歧义问题。

注意：
- Top-K refine 不单独写成创新点；
- clustering 不单独写成普通聚类；
- 应统一写成 mode-based pose inference / consolidation。

---

## 8. 推荐实验组织方式

建议后续实验按如下结构组织：

### 主结果
- Baseline
- Ours

### 消融实验
- Ours w/o RS-MRQ
- Ours w/o Robust Vote
- Ours + KDE

### 难例分析
重点分析：
- obj_id = 8
- obj_id = 10
- obj_id = 11
- obj_id = 12

---

## 9. 当前默认结论

在现阶段代码与实验基础上，可以把项目结论概括为：

> 本研究的核心不在于更重的后端 refinement，而在于通过前端匹配增强、中间层鲁棒投票以及后端多假设模式推理，实现对 PPF 歧义问题的系统性抑制。

这也是当前 README、配置文件命名与论文 Method 部分应该保持一致的核心表述。
