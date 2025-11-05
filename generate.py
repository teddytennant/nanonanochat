import torch
from model import GPT, GPTConfig
from data import get_data
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
    args = parser.parse_args()

    # Config (same as training)
    config = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.1)
    model = GPT(config)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {args.checkpoint}: {e}")
    model.to(device)

    # Tokenizer
    _, _, tokenizer = get_data()

    # Generate
    generated = generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.do_sample)
    print(generated)

if __name__ == '__main__':
    main()