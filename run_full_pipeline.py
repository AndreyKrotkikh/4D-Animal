
import os
import sys
import subprocess
from config.keys import Keys

def run_command(cmd, description):
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    
    # Use subprocess.run to capture output and check for errors
    # We use shell=True and executable='/bin/bash' to ensure proper environment execution
    result = subprocess.run(
        cmd, 
        shell=True, 
        executable='/bin/bash',
        env=os.environ.copy()
    )
    
    if result.returncode != 0:
        print(f"\nERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\nSUCCESS: {description} completed.")

def main():
    print("Starting Full 4D-Animal Pipeline for Custom Video...")
    
    # Paths
    video_path = "/root/cat_test.mp4"
    project_root = "/root/4D-Animal"
    python_exe = "/root/miniconda3/envs/animal/bin/python"
    
    # Change to project root
    os.chdir(project_root)
    
    # 1. Preprocessing: Extract Frames & Generate Masks (SAM)
    # Note: Using the SAM checkpoint we verified exists at /root/4D-Animal/sam_vit_h_4b8939.pth
    cmd_sam = f"{python_exe} custom_pipeline/preprocess_sam.py {video_path} external_data/cop3d_data/cat/test_seq sam_vit_h_4b8939.pth"
    run_command(cmd_sam, "Step 1: Extract Frames & Generate SAM Masks")
    
    # 2. Preprocessing: CSE & PnP
    # This script (run_preprocess_custom.py) imports from cse_embedding and pnp modules
    cmd_preprocess = f"{python_exe} run_preprocess_custom.py"
    run_command(cmd_preprocess, "Step 2: CSE and PnP Preprocessing")
    
    # 3. Scene Optimization
    # This runs the main training loop
    cmd_optim = f"{python_exe} run_optimization_custom.py"
    run_command(cmd_optim, "Step 3: Scene Optimization")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Check results in {project_root}/experiments_custom/test_seq")
    print("="*80)

if __name__ == "__main__":
    main()

