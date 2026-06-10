# Make synthetic dataset using our ground data for training StyleCOMET

import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = "./data/4_way_mwe_ground.ods"
TEST_METADATA_FILE = "test_metadata.csv"

def build_synthetic_comet_dataset():
    df = pd.read_excel(INPUT_FILE)
    
    df = df.dropna(subset=[
        "Greek original", "Greek literal", 
        "English MWE", "English literal"
    ])
    
    train_val_base_df, test_base_df = train_test_split(df, test_size=200, random_state=42)
    
    test_base_df.to_csv(TEST_METADATA_FILE, index=True, index_label="original_index")
    print(f"Saved {len(test_base_df)} holdout rows to '{TEST_METADATA_FILE}' for final evaluation.")

    comet_data = []

    for _, row in train_val_base_df.iterrows():
        gr_mwe = str(row["Greek original"]).strip()
        gr_plain = str(row["Greek literal"]).strip()
        en_mwe = str(row["English MWE"]).strip()
        en_plain = str(row["English literal"]).strip()
        
        if any(x == "" for x in [gr_mwe, gr_plain, en_mwe, en_plain]):
            continue

        # =====================================================================
        # GREEK TO ENGLISH
        # =====================================================================
        
        # --- Source MWE ---
        # Perfect Match: Translated to MWE
        comet_data.append({
            "src": gr_mwe, "mt": en_mwe, "ref": en_mwe, "score": 0.98
        })
        # Penalty: Dropped MWE
        comet_data.append({
            "src": gr_mwe, "mt": en_plain, "ref": en_mwe, "score": 0.80
        })

        # --- Source Plain ---
        # Perfect Match: Translated to Plain
        comet_data.append({
            "src": gr_plain, "mt": en_plain, "ref": en_plain, "score": 0.98
        })
        # Penalty: Added MWE
        comet_data.append({
            "src": gr_plain, "mt": en_mwe, "ref": en_plain, "score": 0.80
        })

        # =====================================================================
        # ENGLISH TO GREEK
        # =====================================================================
        
        # --- Source MWE ---
        # Perfect Match: Translated to MWE
        comet_data.append({
            "src": en_mwe, "mt": gr_mwe, "ref": gr_mwe, "score": 0.98
        })
        # Penalty: Dropped MWE
        comet_data.append({
            "src": en_mwe, "mt": gr_plain, "ref": gr_mwe, "score": 0.80
        })

        # --- Source Plain ---
        # Perfect Match: Translated to Plain
        comet_data.append({
            "src": en_plain, "mt": gr_plain, "ref": gr_plain, "score": 0.98
        })
        # Penalty: Added MWE
        comet_data.append({
            "src": en_plain, "mt": gr_mwe, "ref": gr_plain, "score": 0.80
        })

    comet_df = pd.DataFrame(comet_data)
    
    comet_df = comet_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df, val_df = train_test_split(comet_df, test_size=0.1, random_state=42)
    
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    
    print(f"Generated {len(comet_df)} total synthetic scoring triplets.")
    print(f"  -> {len(train_df)} training samples.")
    print(f"  -> {len(val_df)} validation samples.")

if __name__ == "__main__":
    build_synthetic_comet_dataset()