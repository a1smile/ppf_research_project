# 消融配置说明

本配置包按建议的论文实验组织方式整理。

## 主消融（7 组）
1. `ablation_baseline.yaml`
2. `ablation_rsmrq.yaml`
3. `ablation_robustvote.yaml`
4. `ablation_rsmrq_robustvote.yaml`
5. `ablation_ours_no_rsmrq.yaml`
6. `ablation_ours_no_robustvote.yaml`
7. `ablation_ours.yaml`

## 附加对照
- `ablation_ours_kde.yaml`

## 说明
- `ablation_ours.yaml` 是当前主方法。
- `ablation_ours_kde.yaml` 只比 Ours 多打开 KDE。
- `default.yaml` 和 `ablation_mhsmc.yaml` 建议保留为 legacy/debug，不再作为论文主配置使用。
