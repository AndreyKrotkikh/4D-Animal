
import os
import sys
from cse_embedding.preprocess_cse import preprocess_cse
from pnp.preprocess_pnp import preprocess_pnp
from config.keys import Keys

def main():
    # Configuration for custom cat video
    sequence_index = "test_seq"
    dataset_source = "CUSTOM" # Needs support in InputCop
    frame_limit = 20
    category = "cat"
    
    print(f"Starting preprocessing for sequence {sequence_index}, frame_limit {frame_limit}...")
    
    # Override Keys to point to where we extracted data if needed, 
    # but Keys().dataset_root is usually "external_data/cop3d_data"
    # We extracted to external_data/cop3d_data/cat/test_seq so it matches standard structure roughly
    
    # 1. Run CSE Preprocessing
    cse_cache_path = Keys().preprocess_path_cse
    print(f"--- Step 1: CSE Preprocessing ---")
    preprocess_cse(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cache_path=cse_cache_path,
        frame_limit=frame_limit,
        category=category
    )
    
    # 2. Run PnP Preprocessing
    pnp_cache_path = Keys().preprocess_path_pnp
    print(f"--- Step 2: PnP Preprocessing ---")
    preprocess_pnp(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cache_path=pnp_cache_path,
        frame_limit=frame_limit,
        category=category,
        device="cuda"
    )
    print("All preprocessing done.")

if __name__ == "__main__":
    main()

