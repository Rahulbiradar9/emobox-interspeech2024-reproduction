"""
EmoBox Reproduction Script: EmoDB Intra-corpus Benchmark
INTERSPEECH 2024 Reproduction Pipeline
"""

import os
import sys
import json
import subprocess
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paper Reported Metrics for EmoDB (Table 4, INTERSPEECH 2024)
PAPER_METRICS = {
    "wav2vec2-base": {"ua": 82.06, "wa": 83.14, "f1": 82.21},
    "hubert-base": {"ua": 87.73, "wa": 87.73, "f1": 87.82},
    "wavlm-base": {"ua": 87.03, "wa": 87.12, "f1": 86.76},
}

EMODB_EMOTIONS = ["Fear", "Boredom", "Neutral", "Sad", "Angry", "Happy", "Disgust"]

def generate_plots(model_name, dataset, paper_res, repro_orig, repro_clean=None, conf_matrix=None):
    os.makedirs("figures", exist_ok=True)

    # 1. Bar Chart: Paper vs Reproduced
    metrics = ["Unweighted Accuracy (UA)", "Weighted Accuracy (WA)", "Macro F1"]
    paper_vals = [paper_res["ua"], paper_res["wa"], paper_res["f1"]]
    orig_vals = [repro_orig["ua"], repro_orig["wa"], repro_orig["f1"]]

    x = np.arange(len(metrics))
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    if repro_clean is not None:
        repro_vals = [repro_clean["ua"], repro_clean["wa"], repro_clean["f1"]]
    else:
        repro_vals = [repro_orig["ua"], repro_orig["wa"], repro_orig["f1"]]

    width = 0.35
    plt.bar(x - width/2, paper_vals, width, label="Paper (INTERSPEECH 2024)", color="#2b5c8f", alpha=0.9)
    plt.bar(x + width/2, repro_vals, width, label="Reproduction (Converged Model)", color="#2ca02c", alpha=0.9)

    for i in range(len(metrics)):
        plt.text(x[i] - width/2, paper_vals[i] + 1.2, f"{paper_vals[i]:.2f}%", ha='center', va='bottom', fontsize=10, fontweight="bold")
        plt.text(x[i] + width/2, repro_vals[i] + 1.2, f"{repro_vals[i]:.2f}%", ha='center', va='bottom', fontsize=10, fontweight="bold")

    plt.ylabel("Performance (%)", fontsize=12, fontweight="bold")
    plt.title(f"EmoBox Benchmark Reproduction: {dataset.upper()} ({model_name})", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, metrics, fontsize=11, fontweight="bold")
    plt.ylim(0, 105)
    plt.legend(frameon=True, fontsize=10, loc="lower right")

    plt.tight_layout()
    bar_path = f"figures/{dataset}_{model_name}_paper_vs_reproduction.png"
    plt.savefig(bar_path)
    plt.close()
    print(f"Generated comparison plot: {bar_path}")

    # 2. Confusion Matrix Heatmap
    if conf_matrix is not None and conf_matrix.shape == (7, 7):
        plt.figure(figsize=(8.5, 7.5), dpi=300)
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=EMODB_EMOTIONS,
            yticklabels=EMODB_EMOTIONS,
            cbar=True
        )
        plt.xlabel("Predicted Emotion", fontsize=12, fontweight="bold")
        plt.ylabel("Ground Truth Emotion", fontsize=12, fontweight="bold")
        plt.title(f"EmoDB 5-Fold Confusion Matrix (Clean N={conf_matrix.sum()}, {model_name})", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        cm_path = f"figures/{dataset}_{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Generated confusion matrix plot: {cm_path}")

def run_experiment(model_name="wav2vec2-base", lr="1e-3", hidden_size="256", seed="1234", epochs="100", device=None, eval_only=False):
    import torch
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA was requested but torch.cuda.is_available() is False. Falling back to CPU.")
        device = "cpu"

    dataset = "emodb"
    n_classes = "7"
    exp_root = f"exp/{dataset}_{model_name}_lr{lr}_h{hidden_size}_s{seed}"
    os.makedirs(exp_root, exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/processed", exist_ok=True)

    print(f"=======================================================")
    print(f"  EMOBOX INTERSPEECH 2024 REPRODUCTION")
    print(f"  Dataset:     {dataset.upper()} (Berlin Database of Emotional Speech)")
    print(f"  Model:       {model_name}")
    print(f"  Epochs:      {epochs}")
    print(f"  Learning Rt: {lr}")
    print(f"  Hidden Dim:  {hidden_size}")
    print(f"  Seed:        {seed}")
    print(f"  Device:      {device.upper()} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"=======================================================\n")

    fold_results = []
    total_conf_matrix = np.zeros((7, 7), dtype=int)

    for fold in range(1, 6):
        fold_out = os.path.join(exp_root, f"fold_{fold}")
        print(f"\n>>> Processing FOLD {fold}/5...")

        if not eval_only:
            cmd = [
                sys.executable,
                "examples/sb/train.py",
                f"examples/sb/hparams/{model_name}_freeze.yaml",
                "--output_folder", fold_out,
                "--seed", str(seed),
                "--batch_size", "32",
                "--lr", str(lr),
                "--train_annotation", f"data/{dataset}/fold_{fold}/{dataset}_train_fold_{fold}.json",
                "--valid_annotation", f"data/{dataset}/fold_{fold}/{dataset}_valid_fold_{fold}.json",
                "--test_annotation", f"data/{dataset}/fold_{fold}/{dataset}_test_fold_{fold}.json",
                "--number_of_epochs", str(epochs),
                "--feat_dir", f"dump/{dataset}/{model_name}",
                "--label_map", f"data/{dataset}/label_map.json",
                "--device", device,
                "--out_n_neurons", n_classes,
                "--hidden_size", str(hidden_size),
            ]

            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"[ERROR] Fold {fold} failed with exit code {ret.returncode}")
                return None

        # Parse test scores from fold_out/test_scores
        test_score_files = glob.glob(os.path.join(fold_out, "test_scores", "epoch_*.txt"))
        valid_score_files = glob.glob(os.path.join(fold_out, "valid_scores", "epoch_*.txt"))

        # Find best checkpoint selected by validation set
        best_valid_wa = -1.0
        best_epoch = 1
        for vf in valid_score_files:
            m = re.search(r"epoch_(\d+)\.txt", vf)
            if not m:
                continue
            ep = int(m.group(1))
            with open(vf) as f:
                content = f.read()
                wa_match = re.search(r"Overall WA\s+([0-9.]+)", content)
                if wa_match:
                    wa = float(wa_match.group(1))
                    if wa > best_valid_wa:
                        best_valid_wa = wa
                        best_epoch = ep

        # Read test score at the best validation epoch
        best_test_file = os.path.join(fold_out, "test_scores", f"epoch_{best_epoch}.txt")
        if not os.path.exists(best_test_file):
            # Fallback to the latest available test epoch
            best_test_file = sorted(test_score_files, key=lambda x: int(re.search(r"epoch_(\d+)\.txt", x).group(1)))[-1]
            best_epoch = int(re.search(r"epoch_(\d+)\.txt", best_test_file).group(1))

        with open(best_test_file) as f:
            best_test_content = f.read()
            sel_wa = float(re.search(r"Overall WA\s+([0-9.]+)", best_test_content).group(1))
            sel_ua = float(re.search(r"Overall UA\s+([0-9.]+)", best_test_content).group(1))
            sel_f1 = float(re.search(r"Overall Macro-F1\s+([0-9.]+)", best_test_content).group(1))

        # Parse final epoch (converged model at the end of schedule)
        final_test_file = os.path.join(fold_out, "test_scores", f"epoch_{epochs}.txt")
        if not os.path.exists(final_test_file):
            final_test_file = sorted(test_score_files, key=lambda x: int(re.search(r"epoch_(\d+)\.txt", x).group(1)))[-1]
        
        with open(final_test_file) as f:
            final_test_content = f.read()
            final_wa = float(re.search(r"Overall WA\s+([0-9.]+)", final_test_content).group(1))
            final_ua = float(re.search(r"Overall UA\s+([0-9.]+)", final_test_content).group(1))
            final_f1 = float(re.search(r"Overall Macro-F1\s+([0-9.]+)", final_test_content).group(1))

            # Parse confusion matrix from final converged model
            cm_match = re.search(r"confusion_matrix:\s*\n\s*\[(.*)\]", final_test_content, re.DOTALL)
            if cm_match:
                cm_str = cm_match.group(1)
                rows = [list(map(int, r.strip().strip("[]").split())) for r in cm_str.strip().split("\n") if r.strip()]
                if len(rows) == 7 and all(len(r) == 7 for r in rows):
                    total_conf_matrix += np.array(rows)

        # Also track peak test epoch for complete analysis
        max_test_wa, max_test_ua, max_test_f1 = -1.0, -1.0, -1.0
        max_test_ep = 1
        for tf in test_score_files:
            m = re.search(r"epoch_(\d+)\.txt", tf)
            if not m:
                continue
            ep = int(m.group(1))
            with open(tf) as f:
                c = f.read()
                wa_m = re.search(r"Overall WA\s+([0-9.]+)", c)
                ua_m = re.search(r"Overall UA\s+([0-9.]+)", c)
                f1_m = re.search(r"Overall Macro-F1\s+([0-9.]+)", c)
                if wa_m and ua_m and f1_m:
                    t_wa = float(wa_m.group(1))
                    if t_wa > max_test_wa:
                        max_test_wa = t_wa
                        max_test_ua = float(ua_m.group(1))
                        max_test_f1 = float(f1_m.group(1))
                        max_test_ep = ep

        print(f"Fold {fold} Results:")
        print(f"  Converged Final (Epoch {epochs}): UA={final_ua*100:.2f}%, WA={final_wa*100:.2f}%, Macro-F1={final_f1*100:.2f}%")
        print(f"  Val-Selected (Epoch {best_epoch}): UA={sel_ua*100:.2f}%, WA={sel_wa*100:.2f}%, Macro-F1={sel_f1*100:.2f}% (Valid WA: {best_valid_wa*100:.2f}%)")
        print(f"  Peak Test (Epoch {max_test_ep}): UA={max_test_ua*100:.2f}%, WA={max_test_wa*100:.2f}%, Macro-F1={max_test_f1*100:.2f}%")

        fold_results.append({
            "fold": fold,
            "converged_final": {"epoch": int(epochs), "ua": final_ua, "wa": final_wa, "f1": final_f1},
            "best_valid_epoch": best_epoch,
            "best_valid_wa": best_valid_wa,
            "test_at_best_valid": {"ua": sel_ua, "wa": sel_wa, "f1": sel_f1},
            "test_peak": {"epoch": max_test_ep, "ua": max_test_ua, "wa": max_test_wa, "f1": max_test_f1}
        })

    # Summary Statistics for Converged Model (Original Pipeline Cumulative)
    final_uas = [f["converged_final"]["ua"] * 100 for f in fold_results]
    final_was = [f["converged_final"]["wa"] * 100 for f in fold_results]
    final_f1s = [f["converged_final"]["f1"] * 100 for f in fold_results]

    mean_ua, std_ua = np.mean(final_uas), np.std(final_uas)
    mean_wa, std_wa = np.mean(final_was), np.std(final_was)
    mean_f1, std_f1 = np.mean(final_f1s), np.std(final_f1s)

    # Summary Statistics for Validation-Selected Model
    val_uas = [f["test_at_best_valid"]["ua"] * 100 for f in fold_results]
    val_was = [f["test_at_best_valid"]["wa"] * 100 for f in fold_results]
    val_f1s = [f["test_at_best_valid"]["f1"] * 100 for f in fold_results]

    val_mean_ua, val_std_ua = np.mean(val_uas), np.std(val_uas)
    val_mean_wa, val_std_wa = np.mean(val_was), np.std(val_was)
    val_mean_f1, val_std_f1 = np.mean(val_f1s), np.std(val_f1s)

    # Clean Evaluation (Independent final checkpoint evaluation without cumulative test metric buffer)
    repro_clean = None
    clean_cm = total_conf_matrix
    try:
        sys.path.append("examples/sb")
        from train import dataio_prep
        from hyperpyyaml import load_hyperpyyaml
        from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
        import torch
        import speechbrain as sb

        clean_uas, clean_was, clean_f1s = [], [], []
        clean_cm = np.zeros((7, 7), dtype=int)

        for fold in range(1, 6):
            fold_out = os.path.join(exp_root, f"fold_{fold}")
            hparams_file = f"examples/sb/hparams/{model_name}_freeze.yaml"
            with open(hparams_file) as f:
                hparams = load_hyperpyyaml(f, overrides={
                    "output_folder": fold_out,
                    "feat_dir": f"dump/{dataset}/{model_name}",
                    "train_annotation": f"data/{dataset}/fold_{fold}/{dataset}_train_fold_{fold}.json",
                    "valid_annotation": f"data/{dataset}/fold_{fold}/{dataset}_valid_fold_{fold}.json",
                    "test_annotation": f"data/{dataset}/fold_{fold}/{dataset}_test_fold_{fold}.json",
                    "label_map": f"data/{dataset}/label_map.json",
                    "out_n_neurons": 7,
                    "hidden_size": int(hidden_size),
                    "number_of_epochs": int(epochs),
                    "lr": float(lr)
                })

            datasets = dataio_prep(hparams)
            test_set = datasets["test"]
            test_loader = sb.dataio.dataloader.make_dataloader(test_set, batch_size=32)

            ckpts = hparams["checkpointer"].list_checkpoints()
            ckpts.sort()
            hparams["checkpointer"].load_checkpoint(ckpts[-1])
            model = hparams["model"]
            model.eval()

            preds, targets = [], []
            with torch.no_grad():
                for batch in test_loader:
                    feats, lens = batch.feat
                    out = model[0](feats.data)
                    preds.append(out.cpu().numpy())
                    targets.append(batch.emo_encoded.data.cpu().numpy())

            cat_preds = np.concatenate(preds, axis=0).argmax(axis=-1)
            cat_targets = np.concatenate(targets, axis=0)

            c_wa = accuracy_score(cat_targets, cat_preds) * 100
            c_ua = recall_score(cat_targets, cat_preds, average="macro") * 100
            c_f1 = f1_score(cat_targets, cat_preds, average="macro") * 100
            clean_was.append(c_wa)
            clean_uas.append(c_ua)
            clean_f1s.append(c_f1)
            clean_cm += confusion_matrix(cat_targets, cat_preds, labels=list(range(7)))

        clean_mean_ua, clean_std_ua = np.mean(clean_uas), np.std(clean_uas)
        clean_mean_wa, clean_std_wa = np.mean(clean_was), np.std(clean_was)
        clean_mean_f1, clean_std_f1 = np.mean(clean_f1s), np.std(clean_f1s)
        repro_clean = {"ua": clean_mean_ua, "wa": clean_mean_wa, "f1": clean_mean_f1}
    except Exception as e:
        print(f"[NOTE] Clean evaluation skipped: {e}")
        repro_clean = None

    paper = PAPER_METRICS.get(model_name, {"ua": 82.06, "wa": 83.14, "f1": 82.21})
    repro = {"ua": mean_ua, "wa": mean_wa, "f1": mean_f1}

    print("\n" + "="*84)
    print(f"  REPRODUCTION BENCHMARK SUMMARY: {dataset.upper()} ({model_name})")
    print("="*84)
    print(f"{'Metric':<25} | {'Paper (Table 4)':<15} | {'Repro (Orig Pipeline)':<22} | {'Repro (Clean Model)':<20}")
    print("-" * 84)
    if repro_clean is not None:
        print(f"{'Unweighted Accuracy (UA)':<25} | {paper['ua']:>13.2f}% | {mean_ua:>12.2f}% +/- {std_ua:.2f}%   | {clean_mean_ua:>11.2f}% +/- {clean_std_ua:.2f}%")
        print(f"{'Weighted Accuracy (WA)':<25}   | {paper['wa']:>13.2f}% | {mean_wa:>12.2f}% +/- {std_wa:.2f}%   | {clean_mean_wa:>11.2f}% +/- {clean_std_wa:.2f}%")
        print(f"{'Macro F1 Score':<25}           | {paper['f1']:>13.2f}% | {mean_f1:>12.2f}% +/- {std_f1:.2f}%   | {clean_mean_f1:>11.2f}% +/- {clean_std_f1:.2f}%")
    else:
        print(f"{'Unweighted Accuracy (UA)':<25} | {paper['ua']:>13.2f}% | {mean_ua:>12.2f}% +/- {std_ua:.2f}%   | {'N/A':>20}")
        print(f"{'Weighted Accuracy (WA)':<25}   | {paper['wa']:>13.2f}% | {mean_wa:>12.2f}% +/- {std_wa:.2f}%   | {'N/A':>20}")
        print(f"{'Macro F1 Score':<25}           | {paper['f1']:>13.2f}% | {mean_f1:>12.2f}% +/- {std_f1:.2f}%   | {'N/A':>20}")
    print("-" * 84)
    print(f"Validation-Selected Checkpoint Mean: WA={val_mean_wa:.2f}% +/- {val_std_wa:.2f}%, UA={val_mean_ua:.2f}% +/- {val_std_ua:.2f}%, F1={val_mean_f1:.2f}% +/- {val_std_f1:.2f}%")
    print("="*84)

    # Save to JSON
    out_json = {
        "model": model_name,
        "dataset": dataset,
        "lr": lr,
        "hidden_size": hidden_size,
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "paper_results": paper,
        "original_pipeline_mean": {"ua": mean_ua, "wa": mean_wa, "f1": mean_f1},
        "original_pipeline_std": {"ua": std_ua, "wa": std_wa, "f1": std_f1},
        "clean_model_mean": {"ua": clean_mean_ua, "wa": clean_mean_wa, "f1": clean_mean_f1} if repro_clean else None,
        "clean_model_std": {"ua": clean_std_ua, "wa": clean_std_wa, "f1": clean_std_f1} if repro_clean else None,
        "clean_gain_over_paper": {
            "ua": clean_mean_ua - paper["ua"],
            "wa": clean_mean_wa - paper["wa"],
            "f1": clean_mean_f1 - paper["f1"]
        } if repro_clean else None,
        "validation_selected_mean": {"ua": val_mean_ua, "wa": val_mean_wa, "f1": val_mean_f1},
        "validation_selected_std": {"ua": val_std_ua, "wa": val_std_wa, "f1": val_std_f1},
        "folds": fold_results
    }

    result_json_path = f"results/processed/{dataset}_{model_name}_reproduction.json"
    with open(result_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Detailed JSON results saved to: {result_json_path}")

    # Generate Plots
    generate_plots(model_name, dataset, paper, repro, repro_clean, clean_cm)
    return out_json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EmoBox Reproduction Runner")
    parser.add_argument("--model", default="wav2vec2-base", choices=["wav2vec2-base", "hubert-base", "wavlm-base"])
    parser.add_argument("--lr", default="1e-3")
    parser.add_argument("--hidden_size", default="256")
    parser.add_argument("--epochs", default="100")
    parser.add_argument("--seed", default="1234")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate existing checkpoints")
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        lr=args.lr,
        hidden_size=args.hidden_size,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        eval_only=args.eval_only
    )
