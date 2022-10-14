import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
  def __init__(self, d_model, n_head, ffn_p, attn_p, eps):
    super().__init__()
    assert d_model % n_head == 0, 'n_head should divide d_model'
    self.d_model, self.n_head = d_model, n_head
    self.d_head = d_model // n_head
    self.ffn_drop, self.attn_drop = nn.Dropout(ffn_p), nn.Dropout(attn_p)
    self.ln1 = nn.LayerNorm(d_model, eps)
    self.ln2 = nn.LayerNorm(d_model, eps)
    self.Wx = nn.Linear(d_model, 3 * d_model)  # Wq, Wk, and Wv combined
    self.Wo = nn.Linear(d_model, d_model)
    self.ffn = nn.Sequential(
      nn.Linear(d_model, 4 * d_model),
      nn.GELU(),
      nn.Linear(4 * d_model, d_model),
      nn.GELU()
    )

  def forward(self, x, mask):
    inp_len, d_model = x.shape[-2:]
    q, k, v = torch.split(
      self.Wx(x).transpose(-2, -1)
                .reshape(-1, self.n_head * 3, self.d_head, inp_len),
      self.n_head,
      -3
    )
    score = F.softmax(
      ((q.transpose(-2, -1) @ k) + mask) / (self.d_model ** 0.5), -1
    )
    attn = (v @ score.transpose(-2, -1)).reshape(-1, d_model, inp_len)\
                                        .transpose(-2, -1)
    x = self.ln1(x + self.attn_drop(self.Wo(attn)))
    return self.ln2(x + self.ffn_drop(self.ffn(x))), score


class GPT(nn.Module):
  def __init__(
    self, d_model, n_head, n_layer, seq_len, eps, ffn_p, attn_p, emb_p, vocab
  ):
    super().__init__()
    self.seq_len = seq_len
    self.d_model, self.vocab = d_model, vocab
    self.n_layer, self.n_head = n_layer, n_head
    self.ffn_p, self.attn_p, self.emb_p = ffn_p, attn_p, emb_p
    # word embeddings
    self.emb = nn.Embedding(vocab, d_model)
    self.emb_drop = nn.Dropout(emb_p)
    # positional encoding
    pe = torch.empty(self.seq_len, self.d_model)
    nn.init.normal_(pe, 0, 0.02)
    self.pe = nn.Parameter(pe)
    # decoder mask
    mask = torch.tril(torch.ones(seq_len, seq_len)) - 1
    mask[mask == -1] = float('-inf')
    self.mask = nn.Parameter(mask, requires_grad=False)
    # decoder layers
    self.layers = nn.ModuleList(
      [Layer(d_model, n_head, ffn_p, attn_p, eps) for _ in range(n_layer)]
    )
    # linear projection from output embeddings to token probabilities
    self.linear = nn.Linear(d_model, vocab)

  def forward(self, ids):
    inp_len = ids.shape[-1]
    assert inp_len <= self.seq_len, 'input sequence too long'
    scores = []
    mask = self.mask[:inp_len, :inp_len]
    x = self.emb_drop(self.emb(ids) + self.pe[:inp_len])
    for layer in self.layers:
      x, score = layer(x, mask)
      scores.append(score.squeeze(0))
    return self.linear(x.squeeze(0)), scores

  def loss(self, inp, tgt):
    return F.cross_entropy(
      self.predict(inp)[0].reshape(-1, self.vocab), tgt.reshape(-1)
    )

  # beam search
  def complete(self, ids, n_token=-1, k=1, temp=1, alpha=0.4, eos_id=2):
    assert temp > 0, 'temperature should be greater than 0'
    assert len(ids.shape) == 1, 'batched inputs not allowed'
    assert len(ids) < self.seq_len, 'input sequence too long'
    if n_token == -1 or len(ids) + n_token > self.seq_len:
      n_token = self.seq_len - len(ids)
    with torch.no_grad():
      topk_probs, topk_ids = torch.topk(
        F.softmax(self(ids)[0] / temp, -1)[-1], k
      )
      log_probs = torch.log(topk_probs)
      beams = list(torch.cat((ids.repeat(k, 1), topk_ids[:, None]), -1))
      for i in range(k):
        for _ in range(n_token - 1):
          if beams[i][-1].item() == eos_id:
            break
          probs = F.softmax(self(beams[i])[0] / temp, -1)[-1]
          next_id = torch.multinomial(probs, 1)
          log_probs[i] += torch.log(probs[next_id[0]])
          beams[i] = torch.cat((beams[i], next_id))
      log_probs /= torch.tensor([len(beam) for beam in beams]) ** alpha
      # return sequence with max probability
      return beams[torch.argmax(log_probs)]
