import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import GPT, GPTConfig
from data import get_data
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available, skipping logging")

def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=1, grad_accum_steps=1, use_amp=True, patience=5, start_epoch=0, best_val_loss=None):
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    model.train()
    if best_val_loss is None:
        best_val_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids = batch.to(device)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = -1  # ignore last token

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, loss = model(input_ids, targets)
                loss = loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * grad_accum_steps

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch.to(device)
                targets = input_ids.clone()
                targets[:, :-1] = input_ids[:, 1:]
                targets[:, -1] = -1
                _, loss = model(input_ids, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")
        model.train()

        if WANDB_AVAILABLE:
            wandb.log({'epoch': epoch+1, 'train_loss': avg_loss, 'val_loss': avg_val_loss})

        # Check for best model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': {
                'vocab_size': model.config.vocab_size,
                'block_size': model.config.block_size,
                'n_layer': model.config.n_layer,
                'n_head': model.config.n_head,
                'n_embd': model.config.n_embd,
                'dropout': model.config.dropout,
            },
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        }
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, 'best_model.pt')
            early_stop_counter = 0
            logger.info("Saved best model")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save checkpoint
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if WANDB_AVAILABLE:
        wandb.init(project="nanonanochat", config=vars(args))

    start_epoch = 0
    best_val_loss = None

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        cfg = checkpoint['config']
        config = GPTConfig(
            vocab_size=cfg['vocab_size'],
            block_size=cfg['block_size'],
            n_layer=cfg['n_layer'],
            n_head=cfg['n_head'],
            n_embd=cfg['n_embd'],
            dropout=cfg.get('dropout', 0.1),
        )
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Data
        train_loader, val_loader, tokenizer = get_data(batch_size=args.batch_size, block_size=config.block_size)

        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs // args.grad_accum_steps)

        # Restore optimizer and scheduler state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('val_loss', None)
        logger.info(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss}")
    else:
        # Config
        config = GPTConfig(vocab_size=50257, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
        model = GPT(config).to(device)

        # Data
        train_loader, val_loader, tokenizer = get_data(batch_size=args.batch_size, block_size=args.block_size)

        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs // args.grad_accum_steps)

    # Train
    train(model, train_loader, val_loader, optimizer, scheduler, device, args.epochs, args.grad_accum_steps, args.use_amp, args.patience, start_epoch, best_val_loss)

if __name__ == '__main__':
    main()