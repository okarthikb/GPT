import random, pickle, torch, wandb
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, cos, pi


def positional_encoding(seq_len, d_model):
  assert d_model % 2 == 0, 'd_model should be even'
  freq = 0.0001 ** (2 * (torch.arange(2, d_model + 2) // 2) / d_model)
  pos = torch.arange(seq_len).repeat(d_model, 1)
  sin = torch.sin(freq[::2, None] * pos[::2])
  cos = torch.cos(freq[1::2, None] * pos[1::2])
  return torch.cat((sin, cos), -1).reshape(d_model, seq_len).transpose(0, 1)


def SGDR(max_lr, min_lr, warmup, steps_max):
  def schedule(steps):
    if steps < warmup:
      return (steps + 1) * max_lr / warmup
    steps -= warmup
    if steps // steps_max % 2 == 0:
      scale = 0.5 + cos(pi * steps / steps_max) / 2
    else:
      scale = 0.5 - cos(pi * steps / steps_max) / 2
    return min_lr + (max_lr - min_lr) * scale
  return schedule


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
    self,
    d_model,
    n_head,
    n_layer,
    seq_len,
    ffn_p,
    attn_p,
    emb_p,
    eps,
    vocab,
    learn_pe
  ):
    super().__init__()
    self.seq_len = seq_len
    self.d_model, self.vocab = d_model, vocab 
    self.n_layer, self.n_head = n_layer, n_head 
    self.ffn_p, self.attn_p, self.emb_p = ffn_p, attn_p, emb_p 
    self.learn_pe = learn_pe
    self.eps = eps

    self.emb = nn.Embedding(vocab, d_model)
    self.emb_drop = nn.Dropout(emb_p)

    if learn_pe:
      pe = torch.empty(self.seq_len, self.d_model)
      nn.init.normal_(pe, 0, 0.02)
    else:
      pe = positional_encoding(seq_len, d_model)
    self.pe = nn.Parameter(pe, requires_grad=learn_pe)

    mask = torch.tril(torch.ones(seq_len, seq_len)) - 1
    mask[mask == -1] = float('-inf')
    self.mask = nn.Parameter(mask, requires_grad=False)

    self.layers = nn.ModuleList(
      [Layer(d_model, n_head, ffn_p, attn_p, eps) for _ in range(n_layer)]
    )

    self.linear = nn.Linear(d_model, vocab)

    self.n_param = sum(p.numel() for p in self.parameters())

  def forward(self, ids):
    inp_len = ids.shape[-1]
    assert inp_len <= self.seq_len, 'input sequence too long'
    scores = []
    mask = self.mask[:inp_len, :inp_len]
    x = self.emb_drop(self.emb(ids) + self.pe[:inp_len])
    for layer in self.layers:
      x, score = layer(x, mask)
      scores.append(score.squeeze(0))
    return x.squeeze(0), scores

  def predict(self, ids, temp=1):
    x, scores = self(ids)
    return F.softmax(self.linear(x) / (temp + 1e-5), -1), scores

  # beam search
  def complete(self, ids, n_token=-1, k=1, temp=1, alpha=0.4, eos_id=2):
    assert len(ids.shape) == 1, 'batched inputs not allowed'
    assert len(ids) < self.seq_len, 'input sequence too long'

    if n_token == -1 or len(ids) + n_token > self.seq_len:
      n_token = self.seq_len - len(ids)

    with torch.no_grad():
      topk_probs, topk_ids = torch.topk(self.predict(ids, temp)[0][-1], k)
      log_probs = torch.log(topk_probs)
      beams = list(torch.cat((ids.repeat(k, 1), topk_ids[:, None]), -1))
      for i in range(k):
        for _ in range(n_token - 1):
          if beams[i][-1].item() == eos_id:
            break
          probs = self.predict(beams[i], temp)[0][-1]
          next_id = torch.multinomial(probs, 1)
          log_probs[i] += torch.log(probs[next_id[0]])
          beams[i] = torch.cat((beams[i], next_id))
      log_probs /= torch.tensor([len(beam) for beam in beams]) ** alpha

      return beams[torch.argmax(log_probs)]


class Scheduler:
  def __init__(self, opt, schedule=lambda _ : 2e-4):
    self.param_groups = opt.param_groups
    self.schedule = schedule
    self.steps = 0
    self.lr = None

  def zero_grad(self):
    self.opt.zero_grad()

  def step(self):
    self.lr = self.schedule(self.steps)
    for group in self.param_groups:
      group['lr'] = self.lr
    self.steps += 1


class Trainer:
  def __init__(
    self, model, loader, opt, loss_fn, schedule, epochs, project=None, log=64
  ):
    self.model, self.loader, self.loss_fn = model, loader, loss_fn
    self.opt = opt(model.parameters())
    self.scheduler = Scheduler(self.opt, schedule)
    self.epochs = epochs
    self.project = project
    self.log = log

  def state(self, epoch):
    return {
      'model': self.model.state_dict(),
      'opt': self.opt.state_dict(),
      'steps': self.scheduler.steps,
      'epoch': epoch
    }

  def train(self, accum_mod=1, ckpt_mod=1, load_dir=None, verbose=True):
    print('training start')

    if self.project is not None:
      wandb.init(project=self.project, config={'epochs': self.epochs})
      wandb.watch(self.model, log_freq=self.log)
      print('initialized wandb project')

    last_epoch = 0
    if load_dir is not None:
      state = torch.load(load_dir)
      self.model.load_state_dict(state['model'])
      self.opt.load_state_dict(state['opt'])
      self.scheduler.step = state['steps']
      last_epoch = state['epoch']
      print('checkpoint loaded')

    losses = []
    for epoch in range(1, self.epochs + 1):
      cur_epoch = last_epoch + epoch
      for i, data in enumerate(self.loader(), 1):
        loss = self.loss_fn(self.model, *data)
        loss.backward()
        losses.append(loss.item())
        # accumulate gradient over batches
        if i % accum_mod == 0:
          self.scheduler.step()
          self.opt.step()
          self.opt.zero_grad()
          if self.project is not None:
            wandb.log(
              {'loss': sum(losses[-accum_mod:]), 'lr': self.scheduler.lr}
            )

      print(
        f'epoch: {cur_epoch}/{last_epoch + self.epochs}\t\
          loss: {sum(losses[-len(self.loader):]) / len(self.loader)}'
      )

      yield self.state(cur_epoch)


class Loader:
  def __init__(self, data, batch_size, process):
    self.data, self.batch_size, self.process = data, batch_size, process

  def __call__(self):
    random.shuffle(self.data)
    for i in range(0, len(self.data), self.batch_size):
      yield self.process(self.data[i:i + self.batch_size])

  def __len__(self):
    return len(self.data) // self.batch_size
