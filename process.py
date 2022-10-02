import re, pickle
import youtokentome as yttm
from datasets import load_dataset


T = re.sub(
  '<unk>',
  '',
  ''.join(load_dataset('wikitext', 'wikitext-103-v1', split='train')['text'])
)

open('train.txt', 'w').write(T)

yttm.BPE.train(
  data='train.txt',
  model='tokenizer.model',
  vocab_size=40000,
  unk_id=0,
  pad_id=3,
  bos_id=1,
  eos_id=2
)

bpe = yttm.BPE(model='tokenizer.model')

pickle.dump(
  bpe.encode([T], output_type=yttm.OutputType.ID)[0],
  open('encoding.pkl', 'wb')
)