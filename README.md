# Norwegian bokmål/nynorsk language model.
Trained on norwegian and nynorsk wikipedia. Used 85476 articles with a minimum word length of 1800.


Vocabular size: 32768

Perplexity:22.294086

# Example of usage
Fine tuning the model on new text
```
from fastai.text import *

df_train = [somedata]
df_valid = [somedata]

vocab = Vocab.load('lm_vocab_32k_wiki')
data_lm = TextLMDataBunch.from_df(path="",vocab=vocab,
                                  train_df=df_train, valid_df=df_valid, bs=96, num_workers=0)
                                  
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.1, pretrained=False).to_fp16(clip=0.1)
learn.load_encoder('lm_wiki_encoder')
learn.freeze()

lr=1e-2
learn.fit_one_cycle(1, lr, moms=(0.8,0.7)) 
learn.unfreeze()
lr /= 10
learn.fit_one_cycle(5, lr, moms=(0.8,0.7)) 
```
