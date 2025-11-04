import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_data(batch_size=8, block_size=1024):
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=block_size)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format(type='torch', columns=['input_ids'])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(tokenized_datasets['validation'], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer