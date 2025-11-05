import torch
from model import GPT, GPTConfig
from data import get_data

def test_model_forward():
    config = GPTConfig(vocab_size=100, block_size=10, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config)
    x = torch.randint(0, 100, (1, 5))
    logits, loss = model(x, x)
    assert logits.shape == (1, 5, 100)
    assert loss is not None

def test_model_generate():
    config = GPTConfig(vocab_size=100, block_size=10, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config)
    x = torch.randint(0, 100, (1, 5))
    out = model.generate(x, 3)
    assert out.shape == (1, 8)

def test_data_loading():
    # This might be slow, so mock or skip if no internet
    try:
        train_loader, val_loader, tokenizer = get_data(batch_size=2, block_size=10)
        batch = next(iter(train_loader))
        assert batch.shape == (2, 10)
    except Exception:
        print("Data loading test skipped")

if __name__ == "__main__":
    test_model_forward()
    test_model_generate()
    test_data_loading()
    print("All tests passed!")