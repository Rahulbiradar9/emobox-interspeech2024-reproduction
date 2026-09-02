import os
import sys
sys.path.append("examples/sb")
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from train import dataio_prep, EmoIdBrain
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

total_cm = np.zeros((7, 7), dtype=int)
fold_results = []

EMODB_EMOTIONS = ["Angry", "Boredom", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

for fold in range(1, 6):
    fold_out = f"exp/emodb_wav2vec2-base_lr1e-3_h256_s1234/fold_{fold}"
    hparams_file = "examples/sb/hparams/wav2vec2-base_freeze.yaml"

    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides={
            "output_folder": fold_out,
            "feat_dir": "dump/emodb/wav2vec2-base",
            "train_annotation": f"data/emodb/fold_{fold}/emodb_train_fold_{fold}.json",
            "valid_annotation": f"data/emodb/fold_{fold}/emodb_valid_fold_{fold}.json",
            "test_annotation": f"data/emodb/fold_{fold}/emodb_test_fold_{fold}.json",
            "label_map": "data/emodb/label_map.json",
            "out_n_neurons": 7,
            "hidden_size": 256,
            "number_of_epochs": 100,
            "lr": 0.001
        })

    datasets = dataio_prep(hparams)
    test_set = datasets["test"]
    test_loader = sb.dataio.dataloader.make_dataloader(test_set, batch_size=32)

    # Load checkpoint from epoch 100
    ckpts = hparams["checkpointer"].list_checkpoints()
    ckpts.sort()
    last_ckpt = ckpts[-1]
    hparams["checkpointer"].load_checkpoint(last_ckpt)

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

    wa = accuracy_score(cat_targets, cat_preds) * 100
    ua = recall_score(cat_targets, cat_preds, average="macro") * 100
    f1 = f1_score(cat_targets, cat_preds, average="macro") * 100
    cm = confusion_matrix(cat_targets, cat_preds, labels=list(range(7)))
    total_cm += cm

    fold_results.append({"fold": fold, "wa": wa, "ua": ua, "f1": f1, "n": len(cat_targets)})
    print(f"Fold {fold}: N={len(cat_targets)} | WA={wa:.2f}% | UA={ua:.2f}% | F1={f1:.2f}%")

was = [x["wa"] for x in fold_results]
uas = [x["ua"] for x in fold_results]
f1s = [x["f1"] for x in fold_results]

print("\n" + "="*60)
print("CLEAN FINAL EPOCH 100 (5-FOLD CROSS VALIDATION):")
print("="*60)
print(f"Mean WA: {np.mean(was):.2f}% +/- {np.std(was):.2f}%")
print(f"Mean UA: {np.mean(uas):.2f}% +/- {np.std(uas):.2f}%")
print(f"Mean F1: {np.mean(f1s):.2f}% +/- {np.std(f1s):.2f}%")
print("="*60)
print("Total Confusion Matrix (Clean N = %d):" % total_cm.sum())
print(total_cm)
