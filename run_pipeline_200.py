
import os
import sys
import subprocess
from config.keys import Keys

# Helper to run commands and exit on error
def run_cmd(cmd, desc):
    print(f"\n--- {desc} ---")
    print(f"Command: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error executing {desc}. Exit code: {ret}")
        sys.exit(ret)

def main():
    # Configuration
    video_path = "/root/cat_test.mp4"
    sequence_name = "test_seq_200" # New sequence name to avoid conflict/confusion with 20 frames
    frame_limit = 200
    dataset_source = "CUSTOM"
    category = "cat"
    python_exe = "/root/miniconda3/envs/animal/bin/python"
    sam_checkpoint = "/root/4D-Animal/sam_vit_h_4b8939.pth"
    
    print(f"Starting Full Pipeline for {frame_limit} frames...")

    # 1. SAM Preprocessing (Extract frames + Masks)
    # output folder: external_data/cop3d_data/cat/test_seq_200
    output_data_path = f"external_data/cop3d_data/{category}/{sequence_name}"
    
    # Note: custom_pipeline/preprocess_sam.py needs to handle the frame limit arg correctly.
    # I will create a temporary runner script for this to pass args cleaner or call via python command string.
    
    cmd_sam = (
        f"{python_exe} custom_pipeline/preprocess_sam.py "
        f"{video_path} "
        f"{output_data_path} "
        f"{sam_checkpoint} "
        # The script hardcoded 20 in previous run, let's update it or pass it if I update the script.
        # Actually I updated preprocess_sam.py to take max_frames but the previous call had hardcoded 20 in the 'else' block.
        # I should update preprocess_sam.py to accept an argument or I can't change it easily from CLI without editing file.
        # Let's do a quick edit to preprocess_sam.py first to ensure it accepts command line args for frame count.
    )
    
    # Wait, I need to edit preprocess_sam.py to accept max_frames from argv[4] if provided.
    
    # 2. CSE Preprocessing
    # We need a runner script for this that accepts args, or we use the 'run_preprocess_custom.py' and edit it dynamically/pass args.
    # Easier to write a specific small script here or use hydra/argparse in the called script.
    # Let's write a specialized preprocess runner for this sequence.
    
    with open("run_preprocess_200.py", "w") as f:
        f.write(f"""
import os
from cse_embedding.preprocess_cse import preprocess_cse
from pnp.preprocess_pnp import preprocess_pnp
from config.keys import Keys

def main():
    sequence_index = "{sequence_name}"
    dataset_source = "{dataset_source}"
    frame_limit = {frame_limit}
    category = "{category}"
    
    print(f"Running CSE/PnP for {{sequence_index}} with limit {{frame_limit}}")
    
    # CSE
    preprocess_cse(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cache_path=Keys().preprocess_path_cse,
        frame_limit=frame_limit,
        category=category
    )
    
    # PnP
    preprocess_pnp(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cache_path=Keys().preprocess_path_pnp,
        frame_limit=frame_limit,
        category=category,
        device="cuda"
    )

if __name__ == "__main__":
    main()
""")

    # 3. Optimization
    # We need to override the config params via command line
    cmd_optim = (
        f"TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 {python_exe} main_optimize_scene.py "
        f"'exp.sequence_index=\"{sequence_name}\"' "
        f"'exp.dataset_source=\"{dataset_source}\"' "
        f"'exp.category=\"{category}\"' "
        f"'exp.frame_limit={frame_limit}' "
        f"'exp.experiment_folder=\"experiments_200\"'"
    )

    # --- EXECUTION FLOW ---
    
    # A. Update SAM script to accept frame count
    # (Done via separate tool call before running this, or assuming I do it now)
    # I will invoke the sam script but I need to make sure it respects 200.
    
    # B. Run SAM
    # run_cmd(cmd_sam, "SAM Preprocessing") 
    # (I'll assume I fix the SAM script in next step)

    # C. Run CSE/PnP
    run_cmd(f"{python_exe} run_preprocess_200.py", "CSE & PnP Preprocessing")

    # D. Run Optimization
    run_cmd(cmd_optim, "Scene Optimization")

if __name__ == "__main__":
    main()

