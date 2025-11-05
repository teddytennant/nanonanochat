import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_data(batch_size=8, block_size=1024):
    """Load and preprocess data for training."""
    try:
        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    try:
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # Concatenate all texts
    train_text = "\n\n".join(dataset['train']['text'])
    val_text = "\n\n".join(dataset['validation']['text'])

    # Tokenize
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)

    # Chunk into blocks
    def chunk_tokens(tokens, block_size):
        chunks = []
        for i in range(0, len(tokens) - block_size + 1, block_size):
            chunks.append(tokens[i:i + block_size])
        return chunks

    train_chunks = chunk_tokens(train_tokens, block_size)
    val_chunks = chunk_tokens(val_tokens, block_size)

    # Convert to tensors
    train_data = torch.tensor(train_chunks, dtype=torch.long)
    val_data = torch.tensor(val_chunks, dtype=torch.long)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer