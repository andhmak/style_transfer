import pandas as pd
from comet import download_model, load_from_checkpoint
import torch
import os
import sys
import multiprocessing as mp
import pickle

# ================= CONFIGURATION =================
INPUT_FILE = "./data/4_way_mwe_ground.ods"
OUTPUT_FILE = "./data/qe_evaluated_ground_truths.ods"

MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"

COL_GREEK_WITH_MWE = "Greek original"
COL_GREEK_NO_MWE = "Greek literal"
COL_ENG_WITH_MWE = "English MWE" 
COL_ENG_NO_MWE = "English literal"

# Output Headers (Greek -> English Ground Truths)
COL_QE_GR_MWE_TO_EN_MWE = "QE_GrMWE_vs_EnMWE"
COL_QE_GR_MWE_TO_EN_PLAIN = "QE_GrMWE_vs_EnPlain"
COL_QE_GR_PLAIN_TO_EN_MWE = "QE_GrPlain_vs_EnMWE"
COL_QE_GR_PLAIN_TO_EN_PLAIN = "QE_GrPlain_vs_EnPlain"

# Output Headers (English -> Greek Ground Truths)
COL_QE_EN_MWE_TO_GR_MWE = "QE_EnMWE_vs_GrMWE"
COL_QE_EN_MWE_TO_GR_PLAIN = "QE_EnMWE_vs_GrPlain"
COL_QE_EN_PLAIN_TO_GR_MWE = "QE_EnPlain_vs_GrMWE"
COL_QE_EN_PLAIN_TO_GR_PLAIN = "QE_EnPlain_vs_GrPlain"

NUM_GPUS = 8
BATCH_SIZE = 128
# =================================================

def run_gpu_tasks(gpu_id, tasks_for_gpu, input_file, model_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    df = pd.read_excel(input_file)
    
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    
    model = load_from_checkpoint(model_path)
    gpu_results = {}
    
    for src_col, mt_col, out_col in tasks_for_gpu:
        data = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            if pd.isna(row[src_col]) or str(row[src_col]).strip() == "" or \
               pd.isna(row[mt_col]) or str(row[mt_col]).strip() == "":
                continue
                
            data.append({
                "src": str(row[src_col]),
                "mt": str(row[mt_col])
            })
            valid_indices.append(idx)
        
        if not data:
            print(f"[GPU {gpu_id}] Skipped task: {out_col} (Missing data)")
            continue
            
        print(f"[GPU {gpu_id}] Started task: {out_col} ({len(data)} pairs)")
        
        # Run prediction
        model_output = model.predict(data, batch_size=BATCH_SIZE, gpus=1, progress_bar=False)
        
        print(f"[GPU {gpu_id}] Finished task: {out_col} | Sys Score: {model_output.system_score:.4f}")
        
        gpu_results[out_col] = {
            "system_score": model_output.system_score,
            "segment_scores": list(zip(valid_indices, model_output.scores))
        }
        
    with open(f"comet_temp_gpu_{gpu_id}.pkl", "wb") as f:
        pickle.dump(gpu_results, f)


def main():
    mp.set_start_method('spawn', force=True)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        sys.exit(1)

    print("Verifying COMETKiwi evaluation model cache...")
    model_path = download_model(MODEL_NAME)
    
    # Define the 8-way Ground Truth Matrix (Src Col, Tgt Col, Output Col)
    evaluation_matrix = [
        # Greek acting as Source
        (COL_GREEK_WITH_MWE, COL_ENG_WITH_MWE, COL_QE_GR_MWE_TO_EN_MWE),
        (COL_GREEK_WITH_MWE, COL_ENG_NO_MWE,   COL_QE_GR_MWE_TO_EN_PLAIN),
        (COL_GREEK_NO_MWE,   COL_ENG_WITH_MWE, COL_QE_GR_PLAIN_TO_EN_MWE),
        (COL_GREEK_NO_MWE,   COL_ENG_NO_MWE,   COL_QE_GR_PLAIN_TO_EN_PLAIN),

        # English acting as Source
        (COL_ENG_WITH_MWE, COL_GREEK_WITH_MWE, COL_QE_EN_MWE_TO_GR_MWE),
        (COL_ENG_WITH_MWE, COL_GREEK_NO_MWE,   COL_QE_EN_MWE_TO_GR_PLAIN),
        (COL_ENG_NO_MWE,   COL_GREEK_WITH_MWE, COL_QE_EN_PLAIN_TO_GR_MWE),
        (COL_ENG_NO_MWE,   COL_GREEK_NO_MWE,   COL_QE_EN_PLAIN_TO_GR_PLAIN)
    ]
    
    tasks_per_gpu = {i: [] for i in range(NUM_GPUS)}
    for idx, task in enumerate(evaluation_matrix):
        assigned_gpu = idx % NUM_GPUS
        tasks_per_gpu[assigned_gpu].append(task)

    print(f"\n--- Launching independent worker processes across {NUM_GPUS} GPUs ---")
    
    processes = []
    for gpu_id in range(NUM_GPUS):
        if not tasks_per_gpu[gpu_id]:
            continue
            
        p = mp.Process(
            target=run_gpu_tasks,
            args=(gpu_id, tasks_per_gpu[gpu_id], INPUT_FILE, model_path)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\n--- All GPUs finished. Stitching results together ---")
    df = pd.read_excel(INPUT_FILE)
    system_results = []
    
    for gpu_id in range(NUM_GPUS):
        temp_file = f"comet_temp_gpu_{gpu_id}.pkl"
        if not os.path.exists(temp_file):
            continue
            
        with open(temp_file, "rb") as f:
            gpu_results = pickle.load(f)
            
        for out_col, data in gpu_results.items():
            df[out_col] = None 
            system_results.append((out_col, data["system_score"]))
            
            for original_idx, score in data["segment_scores"]:
                df.at[original_idx, out_col] = score
                
        os.remove(temp_file)

    print("\n" + "="*50)
    print("      GROUND TRUTH SYSTEM-LEVEL QE SCORES")
    print("="*50)
    for name, score in sorted(system_results):
        print(f"{name.ljust(30)} : {score:.4f}")
    print("="*50)

    print(f"\nWriting results back to spreadsheet: {OUTPUT_FILE}")
    df.to_excel(OUTPUT_FILE, index=False)
    print("Ground Truth Evaluation pipeline successfully finished.")

if __name__ == "__main__":
    main()