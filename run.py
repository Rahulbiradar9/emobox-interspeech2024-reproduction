"""
EmoBox Reproduction Master Runner (Pure Python)
=================================================
Replaces .bat scripts with a 100% Python cross-platform launcher.

Features:
- Automatic virtual environment (.venv) detection and re-exec
- CUDA GPU check & device auto-selection (falls back to CPU if no GPU)
- Direct Python invocation of EmoBox reproduction pipeline
- Formatted console output with clean status banners

Usage:
    python run.py                     # Runs wav2vec2-base with auto device (GPU if available)
    python run.py --model hubert-base # Runs hubert-base
    python run.py --device cpu        # Explicitly run on CPU
    python run.py --device cuda       # Explicitly run on GPU
    python run.py --eval_only         # Skip training and evaluate existing checkpoints
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Environment Self-Bootstrapping (.venv check)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

def ensure_virtualenv():
    """If running outside the project's .venv, re-exec using .venv python."""
    # Path to local .venv python
    if os.name == "nt":
        venv_py = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = ROOT_DIR / ".venv" / "bin" / "python"
        
    if venv_py.exists():
        current_py = Path(sys.executable).resolve()
        target_py = venv_py.resolve()
        if current_py != target_py and not os.environ.get("__EMOBOX_VENV_SWITCHED"):
            env = os.environ.copy()
            env["__EMOBOX_VENV_SWITCHED"] = "1"
            cmd = [str(venv_py)] + sys.argv
            result = subprocess.run(cmd, env=env)
            sys.exit(result.returncode)

ensure_virtualenv()

# ---------------------------------------------------------------------------
# 2. Add ROOT_DIR to Python Path so scripts/ and examples/ are importable
# ---------------------------------------------------------------------------
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# 3. Environment & Hardware Diagnostics
# ---------------------------------------------------------------------------
def check_environment():
    """Prints diagnostic information about PyTorch and CUDA."""
    print("=" * 70)
    print("  EmoBox INTERSPEECH 2024 Benchmark Reproduction Runner")
    print("=" * 70)
    print(f" Python Executable : {sys.executable}")
    print(f" Python Version    : {sys.version.split()[0]}")
    
    try:
        import torch
        print(f" PyTorch Version   : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f" CUDA Available    : {cuda_avail}")
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f" GPU Device        : {device_name} ({vram_gb:.2f} GB VRAM, sm_{capability[0]}{capability[1]})")
        else:
            print(" GPU Device        : None (Running in CPU mode)")
    except ImportError:
        print(" [!] PyTorch is not installed in this environment.")
        print("     Please run: pip install torch torchaudio")
        sys.exit(1)
    print("-" * 70)

# ---------------------------------------------------------------------------
# 4. CLI Argument Parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EmoBox reproduction benchmark (100% Python)"
    )
    parser.add_argument(
        "--model",
        default="wav2vec2-base",
        choices=["wav2vec2-base", "hubert-base", "wavlm-base"],
        help="SSL pretrained model backbone (default: wav2vec2-base)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device: 'auto' (detects GPU), 'cuda', or 'cpu' (default: auto)"
    )
    parser.add_argument(
        "--lr",
        default="1e-3",
        help="Learning rate for downstream classifier (default: 1e-3)"
    )
    parser.add_argument(
        "--hidden_size",
        default="256",
        help="Hidden dimension of downstream classifier (default: 256)"
    )
    parser.add_argument(
        "--epochs",
        default="100",
        help="Maximum training epochs per fold (default: 100)"
    )
    parser.add_argument(
        "--seed",
        default="1234",
        help="Random seed for reproducibility (default: 1234)"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and evaluate already trained fold checkpoints"
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# 5. Main Execution Entry Point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    check_environment()
    
    # Resolve device if 'auto'
    selected_device = args.device
    if selected_device == "auto":
        import torch
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Selected Device: {selected_device.upper()}")
    print(f"[*] Selected Model : {args.model}")
    print(f"[*] Hyperparameters: LR={args.lr}, Hidden={args.hidden_size}, Epochs={args.epochs}, Seed={args.seed}")
    print("=" * 70)
    print("[*] Launching reproduction pipeline...")
    print()

    # Import the reproduction pipeline directly inside Python
    from scripts.run_emodb_reproduction import run_experiment
    
    run_experiment(
        model_name=args.model,
        lr=args.lr,
        hidden_size=args.hidden_size,
        seed=args.seed,
        epochs=args.epochs,
        device=selected_device,
        eval_only=args.eval_only
    )
    
    print("\n" + "=" * 70)
    print(" [✓] Reproduction complete! Check 'results/processed/' and 'figures/'.")
    print("=" * 70)

if __name__ == "__main__":
    main()
