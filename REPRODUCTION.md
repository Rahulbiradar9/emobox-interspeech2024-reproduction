# EmoBox Benchmark Reproduction Guide

This document details the reproduction of the intra-corpus Speech Emotion Recognition (SER) benchmark from the INTERSPEECH 2024 paper:
> **EmoBox: Multilingual Multi-corpus Speech Emotion Recognition Toolkit and Benchmark**  
> *Ziyang Ma, Mingjie Chen, Hezhao Zhang, Zhisheng Zheng, Wenxi Chen, Xiquan Li, Jiaxin Ye, Xie Chen, Thomas Hain*  
> INTERSPEECH 2024 | [arXiv:2406.07162](https://arxiv.org/abs/2406.07162) | [GitHub: emo-box/EmoBox](https://github.com/emo-box/EmoBox)

---

## 1. System & Hardware Environment

- **Operating System**: Windows 11 Home Single Language (64-bit)
- **CPU**: 11th Gen Intel(R) Core(TM) i7-11800H (16 threads @ 2.30 GHz)
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)
- **Python**: 3.10.x (Virtual Environment: `.venv`)
- **Key Dependencies**:
  - `speechbrain`: 1.1.1
  - `transformers`: 5.16.1
  - `torch`: 2.14.0+ (CUDA or CPU)
  - `torchaudio`: 2.11.0+
  - `hyperpyyaml`: 1.2.3
  - `scikit-learn`: 1.7.2
  - `soundfile`: 0.14.0

---

## 2. Selected Benchmark Experiment

- **Corpus**: **EmoDB** (Berlin Database of Emotional Speech, German)
  - 535 total utterances across 10 professional actors (5 male, 5 female).
  - 7 emotion categories: Anger, Boredom, Disgust, Fear, Happy, Neutral, Sad.
  - Total audio duration: ~0.4 hours (~24 minutes).
  - Cross-validation: Official 5-fold speaker-independent split predefined in `data/emodb/fold_1/` to `fold_5/`.
- **Model**: **wav2vec 2.0 base** (`facebook/wav2vec2-base`, ~95M parameters).
  - Pretrained on LibriSpeech 960h.
  - Frozen representations extracted from the last Transformer layer with layer normalization (`--output_norm`).
- **Downstream Architecture**: SUPERB classifier head (`SuperbBaseModel`):
  - Linear projection ($768 \to 128$) $\to$ ReLU $\to$ Mean Pooling $\to$ Linear classifier ($128 \to 7$).
- **Training Hyperparameters**:
  - Epochs: 100
  - Batch Size: 32
  - Optimizer: Adam, initial learning rate $1\times 10^{-4}$
  - Scheduler: Linear warmup for the first 10 epochs, decreasing linearly to 0
  - Validation Split: 20% random partition of training folds (seed 1234)

---

## 3. Compatibility Diagnoses & Modifications

To execute the official EmoBox pipeline on a modern environment, the following minimal compatibility adjustments were diagnosed and resolved:

1. **`src.classifier_head` Import Path Bridge**:
   - Upstream YAML configurations (`examples/sb/hparams/*.yaml`) instantiate `!new:src.classifier_head.SuperbBaseModel`, but `classifier_head.py` resided in `examples/sb/`.
   - *Fix*: Created a lightweight compatibility bridge `examples/sb/src/classifier_head.py` without modifying the core YAML architecture.
2. **`split_sets` Dictionary Key Mutation Bug**:
   - In `examples/sb/dataset_prepare.py`, the validation split function reassigned `train_data` before extracting `valid_keys`, throwing a `KeyError`.
   - *Fix*: Separated the split output dictionary from the source dictionary.
3. **`label_map.json` Case-Sensitivity**:
   - EmoDB annotations used `"emo": "neutral"`, whereas `data/emodb/label_map.json` defined `"Neutral": "Neutral"`.
   - *Fix*: Added lowercase mapping and a case-insensitive fallback in `dataset_prepare.py`.
4. **SpeechBrain 1.x / Modern PyTorch Compatibility**:
   - SpeechBrain 1.x `check_gradients()` signature changed from taking `(loss)` to taking `()`.
   - `LinearWarmupScheduler` in modern SpeechBrain is a callable scheduler rather than having `get_next_value()`.
   - TorchAudio 2.9+ removed legacy backends in favor of `torchcodec` on Windows; added robust audio loading fallback via `soundfile`.
   - DataLoader multiprocessing on Windows requires `num_workers: 0` to prevent worker IPC deadlocks.

---

## 4. How to Run the Reproduction

### Step A: Audio Data Download
Download the official EmoDB zip archive from TU Berlin:
```powershell
python scratch/download_emodb.py
```
*(All 535 wav files are already downloaded and verified in `downloads/emodb/wav/`)*

### Step B: SSL Feature Extraction
Extract frozen `wav2vec2-base` representations from the last layer:
```powershell
.\.venv\Scripts\python.exe examples/sb/speech_feature_extraction.py `
  --model_name wav2vec2-base `
  --model_path pretrained_models/wav2vec2-base `
  --dump_dir dump/emodb/wav2vec2-base `
  --device cpu `
  --data data/emodb/emodb.json `
  --output_norm
```
*(All 535 `.npy` features of dimension `(T, 768)` are already extracted and verified in `dump/emodb/wav2vec2-base/`)*

### Step C: 5-Fold Cross-Validation Training & Evaluation

The benchmark reproduction can now be executed purely via Python:

#### Option 1: Auto / GPU (Recommended)
Automatically detects if CUDA is available and runs on GPU:
```bash
python run.py
```
*(To explicitly force GPU execution: `python run.py --device cuda`)*

#### Option 2: Run on CPU
Since feature representations are pre-extracted, downstream training across all 5 folds takes ~3 minutes total:
```bash
python run.py --device cpu
```
*(You can also customize parameters, e.g.: `python run.py --model wav2vec2-base --epochs 100 --lr 1e-3 --hidden_size 256`)*


---

## 5. Artifacts Produced

- **Raw Fold Logs & Scores**: `exp/emodb_wav2vec2-base_*/fold_1/` through `fold_5/`
- **Structured JSON Metrics**: `results/processed/emodb_wav2vec2-base_reproduction.json`
- **Comparison Visualizations**:
  - `figures/emodb_wav2vec2-base_paper_vs_reproduction.png`
  - `figures/emodb_wav2vec2-base_confusion_matrix.png`
- **Results Table**: Documented in `RESULTS.md`
