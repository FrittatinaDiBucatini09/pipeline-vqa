# TO RUN THIS TEST MOVE IT TO THE ROOT OF THE PROJECT
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("/home/rbalzani/medical-vqa/Thesis")
# We add 'orchestrator' dir to path so we can import orchestrator.py as a module
sys.path.append(str(PROJECT_ROOT / "orchestrator"))

# Import Orchestrator modules
# Now 'orchestrator' refers to orchestrator.py
from orchestrator import STAGE_REGISTRY, StageConfig, run_pipeline

def main():
    print("[TEST] Starting Headless Orchestrator Test...")
    
    # Select Stages:
    # 0: BBox (T1)
    # 1: Attn (T2)
    # 2: Seg (T3)
    # 4: VQA Gen
    # We use indices based on STAGE_REGISTRY order in orchestrator.py
    # [0] = bbox_preproc
    # [1] = attn_map
    # [2] = segmentation
    # [4] = vqa_gen
    
    target_indices = [0, 1, 2, 4]
    selected_stages = [STAGE_REGISTRY[i] for i in target_indices]
    
    print(f"[TEST] Selected Stages: {[s.name for s in selected_stages]}")
    
    configs = []
    
    # 1. BBox (T1) configuration
    # Relative to preprocessing/bounding_box/
    configs.append(StageConfig(
        config_file="configs/gemex/exp_01_vqa.conf"
    ))
    
    # 2. Attention Map (T2) configuration
    # Relative to preprocessing/attention_map/
    configs.append(StageConfig(
        config_file="configs/gemex/exp_01_vqa.conf"
    ))
    
    # 3. Segmentation (T3) configuration
    # Uses env vars, no config file arg
    # LIMIT="50" to match the user's manual test limit
    configs.append(StageConfig(
        env_overrides={
            "TARGET_MODE": "all",
            "LIMIT": "50"
        }
    ))
    
    # 4. VQA Generation
    # Relative to vqa/
    configs.append(StageConfig(
        config_file="configs/generation/hard_coded_gen.conf"
    ))
    
    # Run Pipeline
    # Using the small mapped dataset for testing
    dataset = "gemex_VQA_mimic_mapped.csv"
    
    print(f"[TEST] Dataset: {dataset}")
    print("[TEST] Submitting pipeline...")
    
    run_pipeline(
        stages=selected_stages,
        stage_configs=configs,
        dataset_override=dataset,
        dry_run=False
    )

if __name__ == "__main__":
    main()
