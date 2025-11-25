
import os
import sys
from pnp.preprocess_pnp import preprocess_pnp
from config.keys import Keys

def main():
    sequence_index = "1030_23106_17099"
    dataset_source = "COP3D"
    cache_path = Keys().preprocess_path_pnp
    frame_limit = 20
    category = "dog"
    
    print(f"Running preprocessing for sequence {sequence_index} with frame_limit {frame_limit}...")
    print(f"Cache path: {cache_path}")
    
    preprocess_pnp(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cache_path=cache_path,
        frame_limit=frame_limit,
        category=category,
        device="cuda"
    )
    print("Preprocessing completed.")

if __name__ == "__main__":
    main()

