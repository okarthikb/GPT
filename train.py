import gpt, pickle, torch


data = pickle.load(open('encoding.pkl', 'rb'))

seq_len = 512
data = [data[i:i + seq_len - 1] for i in range(0, 512 * 8 * 32, seq_len - 1)]


def process(batch):
  batch = torch.tensor(batch).long()  # .cuda()
  start = torch.ones(len(batch), 1).long()  # .cuda()
  end = start + 1
  return torch.cat((start, batch), -1), torch.cat((batch, end), -1)


epochs = 100
warmup = 2000
batch_size = 8
accum_mod = 1
loader = gpt.Loader(data, batch_size, process)
schedule = gpt.SGDR(
  3e-4, 0, warmup, epochs * (len(loader) // accum_mod) - warmup
)
opt = torch.optim.AdamW
project = 'test'


def loss_fn(model, x, y):
  return torch.nn.functional.cross_entropy(
    model.predict(x)[0].reshape(-1, model.vocab), y.reshape(-1)
  )


model = gpt.GPT(768, 12, 12, 512, 0.1, 0.1, 0.1, 1e-5, 40000, True)  # .cuda()
trainer = gpt.Trainer(model, loader, opt, loss_fn, schedule, epochs, project)

ckpt_mod = 10
for i, state in enumerate(trainer.train(accum_mod)):
  if i % ckpt_mod:
    torch.save(state, f"checkpoints/{i}.pt")