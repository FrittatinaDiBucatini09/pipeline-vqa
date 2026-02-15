import os

# Base path for configs (Relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Script is in generation/scripts/, we want generation/configs/benchmark/
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../configs/benchmark")

# Grid Search Dimensions
STRATEGIES = {
    "S_standard": {
        "USE_VISUAL_REGIONS": "true",
        "INCLUDE_CONTEXT_IN_INFERENCE": "false",
        "USE_DYNAMIC_PROMPTS": "false",  # In standard mode, we use visual regions, but maybe we should set this to false to be explicit? 
                                         # The README says: Strategy 1 (Standard): USE_VISUAL_REGIONS=true, INCLUDE_CONTEXT=false
                                         # It implies USE_DYNAMIC_PROMPTS might be ignored or false?
                                         # Let's check run_bbox_preprocessing.sh defaults.
                                         # Defaults: USE_DYNAMIC_PROMPTS="true", USE_VISUAL_REGIONS="false"
                                         # So for Standard we need to override.
        "USE_DYNAMIC_PROMPTS": "false" 
    },
    "S_context": {
        "USE_VISUAL_REGIONS": "true",
        "INCLUDE_CONTEXT_IN_INFERENCE": "true",
        "USE_DYNAMIC_PROMPTS": "false"
    },
    "S_question": { # Baseline / Exp E1
        "USE_VISUAL_REGIONS": "false",
        "USE_DYNAMIC_PROMPTS": "true",
        "INCLUDE_CONTEXT_IN_INFERENCE": "false"
    }
}

THRESHOLDS = {
    "T0.30": "0.30",
    "T0.45": "0.45",
    "T0.60": "0.60"
}

# Refinement: CRF vs Raw
REFINEMENTS = {
    "C_crf_on": {
        "SKIP_CRF": "false"
    },
    "C_raw": {
        "SKIP_CRF": "true"
    }
}

# Post-Processing: Smart Padding
PADDINGS = {
    "P_smart": {
        "ENABLE_SMART_PADDING": "true"
    },
    "P_loose": {
        "ENABLE_SMART_PADDING": "false"
    }
}

def generate_configs():
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)

    count = 0
    for strat_name, strat_vars in STRATEGIES.items():
        for thresh_name, thresh_val in THRESHOLDS.items():
            for ref_name, ref_vars in REFINEMENTS.items():
                for pad_name, pad_vars in PADDINGS.items():
                    
                    # unique config name
                    config_name = f"{strat_name}_{thresh_name}_{ref_name}_{pad_name}.conf"
                    file_path = os.path.join(BASE_OUTPUT_DIR, config_name)
                    
                    # Create content
                    content = []
                    content.append(f"# GRID SEARCH CONFIG: {config_name}")
                    content.append(f"# Strategy: {strat_name}")
                    content.append(f"# Threshold: {thresh_val}")
                    content.append(f"# Refinement: {ref_name}")
                    content.append(f"# Padding: {pad_name}")
                    content.append("")
                    
                    # 1. Strategy Variables
                    for k, v in strat_vars.items():
                        content.append(f'{k}="{v}"')
                    
                    # 2. Threshold
                    content.append(f'CAM_THRESHOLD={thresh_val}')
                    
                    # 3. Refinement
                    for k, v in ref_vars.items():
                        content.append(f'{k}="{v}"')

                    # 4. Padding
                    for k, v in pad_vars.items():
                        content.append(f'{k}="{v}"')
                        
                    # 5. Output Management (To avoid overwriting)
                    # Unique results folder for this config
                    results_dir = f"results/benchmark/{config_name.replace('.conf', '')}"
                    content.append(f'OUTPUT_DIR="{results_dir}"')
                    content.append(f'OUTPUT_FORMAT="jsonl"') # Enforce JSONL for evaluation

                    # Write to file
                    with open(file_path, "w") as f:
                        f.write("\n".join(content))
                    
                    count += 1
    
    print(f"Generated {count} configuration files in {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    generate_configs()
