
import os
import sys
import torch
# Helper to run scene optimization easily
# Usage: python run_optimization_custom.py

def main():
    # Command to run
    # Ensure environment variables are set for Triton
    
    # We are using hydra override syntax
    # Sequence index = test_seq
    # Category = cat
    # Dataset source = CUSTOM
    
    cmd = (
        "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 "
        "python main_optimize_scene.py "
        "'exp.sequence_index=\"test_seq\"' "
        "'exp.dataset_source=\"CUSTOM\"' "
        "'exp.category=\"cat\"' "
        "'exp.frame_limit=20' "
        "'exp.experiment_folder=\"experiments_custom\"'"
    )
    
    print(f"Running command: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()
