
import os
from cse_embedding.preprocess_cse import preprocess_cse
from pnp.preprocess_pnp import preprocess_pnp
from config.keys import Keys

def main():
    sequence_index = "test_seq_200"
    dataset_source = "CUSTOM"
    frame_limit = 200
    category = "cat"
    
    print(f"Running CSE/PnP for {sequence_index} with limit {frame_limit}")
    
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
