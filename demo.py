import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_block import Block

print("Torch version : ",torch.__version__)
print("CUDA Available : ",torch.cuda.is_available())
print("GPU Name : ",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Karunakara",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating snaks and drinking tea",
    "deepavali is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]

corpus = [s + " <End> " for s in corpus]
text = " ".join(corpus)
print(text)

words = list(set(text.split()))
print(words)
vocab_size = len(words)
print(vocab_size)

word2idx = {w:i for i,w in enumerate(words)}
print("word2idex : ",word2idx)

idx2word = {i:w for w,i in word2idx.items()}
print("word2idex : ",idx2word)

data = torch.tensor([word2idx[w] for w in text.split()],dtype=torch.long)
print(data)
print(len(data))

block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500


def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix])  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x, y

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional embeddings
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                Block(embedding_dim, block_size, n_heads)
                for _ in range(n_layers)
            ]
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(embedding_dim)

        # Output layer
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Convert token IDs into embeddings
        tok_emb = self.token_embedding(idx)

        # Create positional embeddings
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)

        # Combine token and positional embeddings
        x = tok_emb + pos_emb

        # Pass through Transformer blocks
        x = self.blocks(x)

        # Normalize and generate logits
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # Keep the most recent context
            idx_cond = idx[:, -block_size:]

            # Predict the next token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            next_idx = torch.multinomial(probs, num_samples=1)

            # Append the predicted token
            idx = torch.cat((idx, next_idx), dim=1)

        return idx


model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    xb, yb = get_batch() 
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")



context = torch.tensor([[word2idx["hello"]]], dtype=torch.long)
out = model.generate(context, max_new_tokens=15)

print("\nGenerated text:\n")
print(" ".join(idx2word[int(i)] for i in out[0]))