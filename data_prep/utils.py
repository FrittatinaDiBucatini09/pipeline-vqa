import shutil
from pathlib import Path
import os
import json

TARGET_DIRS = [
    "../vqa",
    "../preprocessing/bounding_box",
    "../preprocessing/attention_map",
    "../preprocessing/segmentation",
    "../preprocessing/medclip_routing"
]

REGISTRY_FILE = Path(__file__).parent / "generated_datasets_registry.json"

def log_to_registry(file_path: Path):
    """
    Logs the generated file path to a JSON registry.
    """
    try:
        registry = []
        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE, 'r') as f:
                try:
                    registry = json.load(f)
                except json.JSONDecodeError:
                    registry = []
        
        # Resolve absolute path to be safe
        abs_path = str(file_path.resolve())
        
        if abs_path not in registry:
            registry.append(abs_path)
            with open(REGISTRY_FILE, 'w') as f:
                json.dump(registry, f, indent=4)
                
    except Exception as e:
        print(f"[WARNING] Failed to log {file_path} to registry: {e}")



def get_output_filename(base_name: str, n_samples: int = None) -> str:
    """
    Generates the output filename based on sampling.
    e.g. 'mimic_ext.csv' or 'mimic_ext_2000_samples.csv'
    """
    if n_samples:
        return f"{base_name}_{n_samples}_samples.csv"
    return f"{base_name}.csv"

def distribute_file(source_path: str, target_dirs: list = TARGET_DIRS):
    """
    Copies the source file to all target directories.
    """
    src = Path(source_path)
    if not src.exists():
        print(f"[ERROR] Source file not found: {src}")
        return

    print("\n" + "=" * 40)
    print(f"  DISTRIBUTING {src.name}")
    print("=" * 40)

    for target_dir in target_dirs:
        # Resolve relative to the script location (assumed to be in data_prep)
        # But wait, source_path might be relative.
        # Let's assume the script is running from data_prep or we resolve strictly.
        
        # We need to ensure we are relative to the project root or the script.
        # The scripts usually run from `data_prep` folder.
        
        dest_dir = Path(target_dir)
        dest_path = dest_dir / src.name
        
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_path)
            log_to_registry(dest_path)
            print(f"  ✓ Copied to: {dest_path}")
        except Exception as e:
            print(f"  ✗ Failed to copy to {dest_path}: {e}")

    print("=" * 40 + "\n")
