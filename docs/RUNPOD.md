# RunPod Training

This project is ready to run on a RunPod PyTorch pod. The classifier uses `--device auto`, which
selects CUDA first, then Apple MPS, then CPU.

## Pod Setup

Use a RunPod PyTorch template with a CUDA GPU and enough disk for model and dataset caches. A 16 GB+
GPU is a practical default for `answerdotai/ModernBERT-base` with `--max-length 768`; reduce batch
size if you hit memory pressure.

## Commands

```bash
git clone https://github.com/gupta799/finance-router-classifier.git
cd finance-router-classifier
git submodule update --init --recursive
chmod +x scripts/train_runpod.sh
./scripts/train_runpod.sh
```

Useful overrides:

```bash
BATCH_SIZE=4 MAX_LENGTH=768 EPOCHS=2 ./scripts/train_runpod.sh
```

Weights & Biases logging is controlled by environment variables:

```bash
wandb login
WANDB_PROJECT=finance-router-classifier WANDB_RUN_NAME=runpod-a10 ./scripts/train_runpod.sh
```

Outputs:

- trained model in `models/finance-router`
- dataset summary in `data-gen/data/synthetic-10k/summary.json`
- local graphs in `reports/finance-router-training`
- packed artifacts in `outputs/finance-router-artifacts.tar.gz`

The generated data and model artifacts are intentionally git-ignored.
