import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures", exist_ok=True)

models = ["wav2vec2-base", "hubert-base"]
model_labels = ["wav2vec 2.0 Base", "HuBERT Base"]

# Paper metrics (Table 4)
paper_wa = [83.14, 87.73]
paper_ua = [82.06, 87.73]
paper_f1 = [82.21, 87.82]

# Reproduced metrics (Converged Model)
repro_wa = [87.66, 91.40]
repro_ua = [87.05, 91.65]
repro_f1 = [87.23, 91.44]

# Plot: Multi-Model Benchmark Comparison (Paper vs Reproduction)
plt.figure(figsize=(10, 6), dpi=300)
sns.set_theme(style="whitegrid")

x = np.arange(len(model_labels))
width = 0.30

rects1 = plt.bar(x - width/2, paper_wa, width, label="Paper (INTERSPEECH 2024)", color="#2b5c8f", alpha=0.9)
rects2 = plt.bar(x + width/2, repro_wa, width, label="Reproduction (Converged Model)", color="#2ca02c", alpha=0.9)

plt.ylabel("Weighted Accuracy (WA %)", fontsize=13, fontweight="bold")
plt.title("Speech Emotion Recognition Benchmark: EmoDB (5-Fold Cross-Validation)", fontsize=14, fontweight="bold", pad=15)
plt.xticks(x, model_labels, fontsize=12, fontweight="bold")
plt.ylim(0, 105)
plt.legend(frameon=True, fontsize=11, loc="lower right")

for rect in rects1:
    h = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, h + 1.2, f"{h:.2f}%", ha='center', va='bottom', fontsize=10, fontweight="bold")

for rect in rects2:
    h = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, h + 1.2, f"{h:.2f}%", ha='center', va='bottom', fontsize=10, fontweight="bold")

plt.tight_layout()
out_path = "figures/emodb_multimodel_paper_vs_reproduction.png"
plt.savefig(out_path)
plt.close()
print(f"Generated multi-model comparison plot: {out_path}")
