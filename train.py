import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import GPT, GPTConfig
from data import get_data
import os
import argparse
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=1, grad_accum_steps=1, use_amp=True):
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids = batch['input_ids'].to(device)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = -1  # ignore last token

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, loss = model(input_ids, targets)
                loss = loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * grad_accum_steps

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                targets = input_ids.clone()
                targets[:, :-1] = input_ids[:, 1:]
                targets[:, -1] = -1
                _, loss = model(input_ids, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")
        model.train()

        # Save checkpoint
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--use_amp', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Config
    config = GPTConfig(vocab_size=50257, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(config).to(device)

    # Data
    train_loader, val_loader, tokenizer = get_data(batch_size=args.batch_size, block_size=args.block_size)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs // args.grad_accum_steps)

    # Train
    train(model, train_loader, val_loader, optimizer, scheduler, device, args.epochs, args.grad_accum_steps, args.use_amp)

if __name__ == '__main__':
    main()