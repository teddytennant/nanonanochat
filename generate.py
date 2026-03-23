import torch
from model import GPT, GPTConfig
from transformers import GPT2Tokenizer
import argparse

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50, top_p=None, do_sample=True):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.transformer.wte.weight.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens, temperature=temperature, do_sample=do_sample, top_k=top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="The future of AI is")
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--do_sample', action='store_true', default=True)
    # Fallback config args for old-format checkpoints (state_dict only)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {args.checkpoint}: {e}")

    # Extract config and state_dict from checkpoint
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        # New format: checkpoint contains config and model_state_dict
        cfg = checkpoint['config']
        config = GPTConfig(
            vocab_size=cfg['vocab_size'],
            block_size=cfg['block_size'],
            n_layer=cfg['n_layer'],
            n_head=cfg['n_head'],
            n_embd=cfg['n_embd'],
            dropout=cfg.get('dropout', 0.1),
        )
        state_dict = checkpoint['model_state_dict']
    else:
        # Old format: checkpoint is a raw state_dict
        config = GPTConfig(
            vocab_size=50257,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
        state_dict = checkpoint

    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Generate
    generated = generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.do_sample)
    print(generated)

if __name__ == '__main__':
    main()
