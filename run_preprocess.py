
import os
import sys
from cse_embedding.preprocess_cse import preprocess_cse
from pnp.preprocess_pnp import preprocess_pnp
from config.keys import Keys

def main():
    sequence_index = "1030_23106_17099"
    dataset_source = "COP3D"
    frame_limit = 20
    category = "dog"
    
    print(f"Starting preprocessing for sequence {sequence_index}, frame_limit {frame_limit}...")

    # 1. Run CSE Preprocessing
    cse_cache_path = Keys().preprocess_path_cse
    print(f"--- Step 1: CSE Preprocessing ---")
    print(f"Cache path: {cse_cache_path}")
    preprocess_cse(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cache_path=cse_cache_path,
        frame_limit=frame_limit,
        category=category
    )
    print("CSE Preprocessing completed.")

    # 2. Run PnP Preprocessing
    pnp_cache_path = Keys().preprocess_path_pnp
    print(f"--- Step 2: PnP Preprocessing ---")
    print(f"Cache path: {pnp_cache_path}")
    preprocess_pnp(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cache_path=pnp_cache_path,
        frame_limit=frame_limit,
        category=category,
        device="cuda"
    )
    print("PnP Preprocessing completed.")
    print("All preprocessing done.")

if __name__ == "__main__":
    main()

