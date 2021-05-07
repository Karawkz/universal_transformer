import numpy as np
import csv
import torch
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import WMT14, Multi30k
from torchtext.data.utils import get_tokenizer
from universal_transformer import datasets, logger, models, tokenizers, vectors
from tqdm import tqdm_notebook, tqdm
from torch import nn
from universal_transformer import utils
from torchtext.data.metrics import bleu_score

SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            fix_length = 100,
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            fix_length = 100,
            lower = True)

train_data, valid_data, test_data = WMT14.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

BATCH_SIZE = 128

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

PAD_IDX = TRG.vocab.stoi['<pad>']

train_loader, valid_loader, test_loader = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE, device = 'cpu')

input_size = len(SRC.vocab)
output_size = len(TRG.vocab)

# models.TransformerModelBase(embedding_num=len(SRC.vocab.stoi), embedding_out=len(TRG.vocab.stoi))
trans_model = models.TransformerModelBase(embedding_num=len(SRC.vocab.stoi), embedding_out=len(TRG.vocab.stoi),
                                   kwargs={'nhead':2,'d_model':300})

# models.VanillaTransformer(d_model=300,nhead=2)
optimizer = torch.optim.Adam(trans_model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

emb = nn.Embedding(len(SRC.vocab.stoi),300)
trans_model.train()
for batch in range(len(train_loader)):
    b = next(iter(train_loader))
    src_e = emb(b.src.T)
    trg_e = emb(b.trg.T)
    src_m = b.src == 1
    trg_m = b.trg == 1
    m = trans_model.transformer.generate_square_subsequent_mask(trg_e.size(0))
    out = trans_model.forward(src_e, trg_e, src_key_padding_mask=src_m, tgt_key_padding_mask=trg_m, tgt_mask = m)
    
    translation = out.reshape(-1, out.shape[-1])
    target = b.trg.T.reshape(-1)
    
    optimizer.zero_grad()
    loss = criterion(translation, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    #bs = bleu_score(candidate_corpus, references_corpus)