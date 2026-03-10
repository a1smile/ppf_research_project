# Experiment Report Template (PPF Enhancements)

## Experimental Setup
### Dataset / Inputs
- Dataset: LM-O / custom
- Model cloud(s): ...
- Scene cloud(s): ...
- Coordinate units: (mm / m), ensure consistency with PPF f4 and voxel size.

### Hardware / OS / Python version
- GPU/CPU:
- RAM:
- OS:
- Python:
- open3d version:

### Baseline parameters (from configs/default.yaml)
- sampling_leaf:
- normal_k:
- angle_step_deg:
- scene_ref_sampling_rate:
- pos_thresh:
- rot_thresh_deg:
- distance_step_ratio (=0.6*leaf)

### Proposed parameters
#### RS–MRQ Hashing
- enable_rsmrq:
- L:
- T_tables:
- w_levels:
- merge_mode:
- seed:

#### Robust Voting
- enable_robust_vote:
- kernel:
- sigma / tau:
- B:
- top_m_per_bucket:
- normalize_feature:

#### KDE + MeanShift Refinement
- enable_kde_refine:
- top_k:
- bandwidth_h:
- max_iter:
- tol:
- use_angle_embedding:

## Evaluation Metrics
### ADD
Definition, threshold, unit.

### ADD-S
Definition and symmetry considerations.

### Rotation/Translation Error
- RotErr (deg): geodesic on SO(3)
- TransErr: L2 distance

### Runtime
- model_build_time
- registration_time
- total_time

## Ablation Study
Run:
```bash
python scripts/run_ablation.py --config configs/default.yaml --model ... --scene ... --repeat 10
python scripts/generate_tables.py
python scripts/plot_results.py
