"""Note 
This model was trained on a small amount of text data.
No hyperparameter tuning or advanced optimization was applied.
It was built to demonstrate the structure and training principles of a GPT-like model from scratch."""


# 🧠 GPT-2 From Scratch (Custom PyTorch Implementation)

This project is a basic GPT-2-like language model implemented **from scratch** using PyTorch.  
It includes everything from dataset creation to multi-head attention and text generation.

---

## 📌 Project Details

- 🔢 **Model size:** ~124M parameters  
- 🧱 **Architecture:** 12 layers, 12 heads, 768 hidden size  
- 🧠 **Vocabulary size:** 50257 (GPT-2 tokenizer from `tiktoken`)  
- 🕓 **Training epochs:** 100  
- 💻 **Trained on:** Google Colab with GPU (CUDA)

---

## 🚀 What This Project Includes

- Custom **Tokenizer support** 
- Manual implementation of:
  - Multi-Head Self Attention
  - Layer Normalization
  - GELU activation
  - Position + Token Embedding
  - Transformer Blocks
- Training loop with:
  - DataLoader
  - CrossEntropyLoss
  - AdamW optimizer
- Text generation after each epoch
- Loss visualization (train vs validation)
- Final model saved as checkpoint and full `.pth`

---

## 🛠 Configuration (GPT-2 Style)

```python
GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'context_length': 256,
    'emb_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': True
}

 
