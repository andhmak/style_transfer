import pandas as pd
from comet import download_model, load_from_checkpoint
import torch
import os
import sys

# ================= CONFIGURATION =================
INPUT_FILE = "./data/translated_sentences_checkpoint_gemini.ods"
OUTPUT_FILE = "./data/evaluated_sentences_gemini.ods"

# Model selection (wmt22-comet-da is the standard for translation quality)
MODEL_NAME = "Unbabel/wmt22-comet-da"

# Source & Reference Column Headers (Ground Truths)
COL_GREEK_WITH_MWE = "Greek original"
COL_GREEK_NO_MWE = "Greek literal"
COL_ENG_WITH_MWE = "English MWE" 
COL_ENG_NO_MWE = "English literal"

# Generated Translation Headers (Machine Outputs)
COL_OUT_GR_MWE_TO_EN_MWE = "Trans_SrcMWE_to_EngMWE"
COL_OUT_GR_MWE_TO_EN_PLAIN = "Trans_SrcMWE_to_EngPlain"
COL_OUT_GR_PLAIN_TO_EN_MWE = "Trans_SrcPlain_to_EngMWE"
COL_OUT_GR_PLAIN_TO_EN_PLAIN = "Trans_SrcPlain_to_EngPlain"

COL_OUT_EN_MWE_TO_GR_MWE = "Trans_EngMWE_to_GrMWE"
COL_OUT_EN_MWE_TO_GR_PLAIN = "Trans_EngMWE_to_GrPlain"
COL_OUT_EN_PLAIN_TO_GR_MWE = "Trans_EngPlain_to_GrMWE"
COL_OUT_EN_PLAIN_TO_GR_PLAIN = "Trans_EngPlain_to_GrPlain"
# =================================================

def evaluate_comet_batch(model, df, src_col, mt_col, ref_col):
    """
    Runs COMET evaluation on a specific combination of columns.
    """
    print(f"  Running: MT [{mt_col}] vs Ref [{ref_col}]...")
    
    data = []
    for _, row in df.iterrows():
        data.append({
            "src": str(row[src_col]),
            "mt": str(row[mt_col]),
            "ref": str(row[ref_col])
        })
    
    gpus_available = 1 if torch.cuda.is_available() else 0
    model_output = model.predict(data, batch_size=16, gpus=gpus_available)
    
    return model_output.system_score, model_output.scores

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        sys.exit(1)
        
    print(f"Loading translations from {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)

    print("Loading COMET evaluation model...")
    model_path = download_model(MODEL_NAME)
    model = load_from_checkpoint(model_path)
    
    # Define the exhaustive 16-way matrix
    # Format: (Source, Machine Translation, Human Reference, Output Score Column Name)
    evaluation_matrix = [
        # =====================================================================
        # DIRECTION 1: GREEK TO ENGLISH
        # =====================================================================
        # 1. Greek MWE -> English MWE
        (COL_GREEK_WITH_MWE, COL_OUT_GR_MWE_TO_EN_MWE, COL_ENG_WITH_MWE, "CMT_GrMWE_to_EnMWE_vs_RefMWE"),
        (COL_GREEK_WITH_MWE, COL_OUT_GR_MWE_TO_EN_MWE, COL_ENG_NO_MWE,  "CMT_GrMWE_to_EnMWE_vs_RefPlain"),
        
        # 2. Greek MWE -> English Plain
        (COL_GREEK_WITH_MWE, COL_OUT_GR_MWE_TO_EN_PLAIN, COL_ENG_WITH_MWE, "CMT_GrMWE_to_EnPlain_vs_RefMWE"),
        (COL_GREEK_WITH_MWE, COL_OUT_GR_MWE_TO_EN_PLAIN, COL_ENG_NO_MWE,  "CMT_GrMWE_to_EnPlain_vs_RefPlain"),
        
        # 3. Greek Plain -> English MWE
        (COL_GREEK_NO_MWE, COL_OUT_GR_PLAIN_TO_EN_MWE, COL_ENG_WITH_MWE, "CMT_GrPlain_to_EnMWE_vs_RefMWE"),
        (COL_GREEK_NO_MWE, COL_OUT_GR_PLAIN_TO_EN_MWE, COL_ENG_NO_MWE,  "CMT_GrPlain_to_EnMWE_vs_RefPlain"),
        
        # 4. Greek Plain -> English Plain
        (COL_GREEK_NO_MWE, COL_OUT_GR_PLAIN_TO_EN_PLAIN, COL_ENG_WITH_MWE, "CMT_GrPlain_to_EnPlain_vs_RefMWE"),
        (COL_GREEK_NO_MWE, COL_OUT_GR_PLAIN_TO_EN_PLAIN, COL_ENG_NO_MWE,  "CMT_GrPlain_to_EnPlain_vs_RefPlain"),

        # =====================================================================
        # DIRECTION 2: ENGLISH TO GREEK
        # =====================================================================
        # 5. English MWE -> Greek MWE
        (COL_ENG_WITH_MWE, COL_OUT_EN_MWE_TO_GR_MWE, COL_GREEK_WITH_MWE, "CMT_EnMWE_to_GrMWE_vs_RefMWE"),
        (COL_ENG_WITH_MWE, COL_OUT_EN_MWE_TO_GR_MWE, COL_GREEK_NO_MWE,   "CMT_EnMWE_to_GrMWE_vs_RefPlain"),
        
        # 6. English MWE -> Greek Plain
        (COL_ENG_WITH_MWE, COL_OUT_EN_MWE_TO_GR_PLAIN, COL_GREEK_WITH_MWE, "CMT_EnMWE_to_GrPlain_vs_RefMWE"),
        (COL_ENG_WITH_MWE, COL_OUT_EN_MWE_TO_GR_PLAIN, COL_GREEK_NO_MWE,   "CMT_EnMWE_to_GrPlain_vs_RefPlain"),
        
        # 7. English Plain -> Greek MWE
        (COL_ENG_NO_MWE, COL_OUT_EN_PLAIN_TO_GR_MWE, COL_GREEK_WITH_MWE, "CMT_EnPlain_to_GrMWE_vs_RefMWE"),
        (COL_ENG_NO_MWE, COL_OUT_EN_PLAIN_TO_GR_MWE, COL_GREEK_NO_MWE,   "CMT_EnPlain_to_GrMWE_vs_RefPlain"),
        
        # 8. English Plain -> Greek Plain
        (COL_ENG_NO_MWE, COL_OUT_EN_PLAIN_TO_GR_PLAIN, COL_GREEK_WITH_MWE, "CMT_EnPlain_to_GrPlain_vs_RefMWE"),
        (COL_ENG_NO_MWE, COL_OUT_EN_PLAIN_TO_GR_PLAIN, COL_GREEK_NO_MWE,   "CMT_EnPlain_to_GrPlain_vs_RefPlain")
    ]
    
    print(f"\n--- Starting Evaluation of {len(evaluation_matrix)} Combinations ---")
    system_results = []
    
    for src, mt, ref, out_col in evaluation_matrix:
        # Check if the generated translation column exists and isn't entirely empty
        if mt not in df.columns or df[mt].isna().all():
            print(f"Skipping {out_col}: Column '{mt}' missing or empty.")
            continue
            
        sys_score, seg_scores = evaluate_comet_batch(model, df, src, mt, ref)
        
        # Assign individual sentence scores to the dataframe
        df[out_col] = seg_scores
        system_results.append((out_col, sys_score))

    # Output Summary Table
    print("\n" + "="*50)
    print("                SYSTEM-LEVEL SCORES")
    print("      (Higher is closer to the human reference)")
    print("="*50)
    for name, score in system_results:
        print(f"{name.ljust(36)} : {score:.4f}")
    print("="*50)

    print(f"\nWriting results back to spreadsheet: {OUTPUT_FILE}")
    df.to_excel(OUTPUT_FILE, index=False)
    print("Evaluation pipeline successfully finished.")

if __name__ == "__main__":
    main()