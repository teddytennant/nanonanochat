# NanoNanoChat: Training GPT-2 from Scratch on a Single A100 GPU

This project implements a full GPT-2 model in PyTorch and provides a way to train it on a single A100 80GB GPU.

## Colab Notebook

For easy execution in Google Colab, use `gpt_training.ipynb` which contains all the code in a runnable notebook format.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have CUDA installed and PyTorch with CUDA support.

## Training

To train a GPT-2 small model (117M parameters):

```bash
python train.py --batch_size 8 --block_size 1024 --n_layer 12 --n_head 12 --n_embd 768 --epochs 1 --grad_accum_steps 4
```

For larger models, adjust the parameters:
- GPT-2 medium: `--n_layer 24 --n_head 16 --n_embd 1024`
- GPT-2 large: `--n_layer 36 --n_head 20 --n_embd 1280`

Use gradient accumulation (`--grad_accum_steps`) to fit larger batches effectively.

## Generation

After training, generate text:

```bash
python generate.py --checkpoint checkpoint_epoch_1.pt --prompt "The meaning of life is"
```

## Notes

- The model uses mixed precision training by default for memory efficiency.
- Training on WikiText-2 dataset. For larger training, consider using OpenWebText.
- On A100 80GB, you can train up to GPT-2 XL (1.5B parameters) with appropriate batch size and accumulation.