推荐保留原 scripts/run_batch.py 给 LMO/BOP 使用。
Stanford Retrieval / Bunny 检索集请使用新脚本：scripts/run_batch_stanford.py

示例：
python scripts/run_batch_stanford.py \
  --config configs/ablation_ours.yaml \
  --csv ppf_csv_outputs/stanford_retrieval_batch_all_variants.csv \
  --out_prefix stanford_retrieval_ours \
  --num_workers 8

如果 CSV 里只有 model_name 或 obj_id，没有 expected_model_path，再补 --models_dir：
python scripts/run_batch_stanford.py \
  --config configs/ablation_ours.yaml \
  --csv ppf_csv_outputs/stanford_retrieval_batch_all_variants.csv \
  --models_dir data/stanford_bunny_ppf/models \
  --out_prefix stanford_retrieval_ours \
  --num_workers 8
