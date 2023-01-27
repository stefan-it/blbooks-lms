# Vocab Generation

## Wordpiece-based

A wordpiece-based vocab for BERT/ELECTRA and ConvBERT is constructed via:

```bash
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    clean_text=True, 
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False, 
)

trainer = tokenizer.train( 
    "bl_1800-1900_extracted.txt",
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

tokenizer.save_model("./")
```

## SPM-based

For T5, a sentencepiece-based model needs to be trained. We select 3GB of text from the original corpus to build a SPM vocab using:

```bash
cat corpus/bl-a* | shuf | head -c 3G > spm_vocab_corpus.txt
```

After installing `sentencepiece` library:

```bash
pip3 install sentencepiece
```

We use the Python Wrapper to train an unigram SPM model:

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="spm_vocab_corpus.txt",
    model_prefix="spiece",
    vocab_size=32000,
    unk_id=2,
    bos_id=-1,
    eos_id=1,
    pad_id=0,
    model_type="unigram",
    train_extremely_large_corpus=True,
)
```

This takes around 1 hour and consumes 107GB CPU RAM (peak).
