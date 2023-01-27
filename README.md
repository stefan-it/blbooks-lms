# Pretrained Language Models on British Library Corpus

This repository presents on overview of pretrained language models on the British Library corpus.

Project is part of the 🌼 [BigScience](https://bigscience.huggingface.co/) ["Language models for historical texts"](https://huggingface.co/bigscience-historical-texts) working group.

# Changelog

* 27.01.2023: Initial version of this repo.

# Corpus

The British Library Corpus - available from [here](https://data.bl.uk/digbks/db14.html) and from the [Datasets Hub](https://huggingface.co/datasets/blbooks) - is used to pretrain
various language models.

The following filtering steps were performed:

* `langdetect` is used to extract English texts only
* Only texts from >= 1800 and < 1900 are used

The final corpus has a size of 24GB and tokens. An overview of the complete filtering steps can be found [here]().

# Vocab Generation

For BERT/ELECTRA and ConvBERT we use the same 32k wordpiece vocabulary, trained on the whole corpus.

For T5 a 32k vocabulary is trained with `sentencepiece`.

All details can be found [here]().

# Pretraining

All pretraining steps (incl. training data generation) are document in the model specific cheatsheets:

* BERT
* ConvBERT
* ELECTRA
* T5

We pretrain all models on a v3-32 TPU pod from the awesome [TPU Research Cloud](https://sites.research.google/trc/about/) program.

# Model Zoo

The following models are available on the Hugging Face Model Hub (currently flagged as private):

| Model Name                                                                                                                                                            | Pretraining Time | Parameters
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------------:
| [`bigscience-historical-texts/bert-base-blbooks-cased`](https://huggingface.co/bigscience-historical-texts/bert-base-blbooks-cased)                                   | 1.64d            |   110,617,344
| [`bigscience-historical-texts/electra-base-blbooks-cased-discriminator`](https://huggingface.co/bigscience-historical-texts/electra-base-blbooks-cased-discriminator) | 2.69d            |   110,026,752
| [`bigscience-historical-texts/electra-base-blbooks-cased-generator`](https://huggingface.co/bigscience-historical-texts/electra-base-blbooks-cased-generator)         | 2.69d            |    34,646,272
| [`bigscience-historical-texts/convbert-base-blbooks-cased`](https://huggingface.co/bigscience-historical-texts/convbert-base-blbooks-cased)                           | 3.83d            |   106,815,624
| [`bigscience-historical-texts/t5-efficient-blbooks-small-el32`](https://huggingface.co/bigscience-historical-texts/t5-efficient-blbooks-small-el32)                   | 0.81d            |   142,322,176
| [`bigscience-historical-texts/t5-efficient-blbooks-base-nl36`](https://huggingface.co/bigscience-historical-texts/t5-efficient-blbooks-base-nl36)                     | 1.98d            |   619,357,440
| [`bigscience-historical-texts/t5-efficient-blbooks-large-nl36`](https://huggingface.co/bigscience-historical-texts/t5-efficient-blbooks-large-nl36)                   | 2.98d            | 1,090,051,072


# Evaluation

All models are evaluated on the [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md) dataset from [HIPE-2022 Shared Task](https://hipe-eval.github.io/HIPE-2022/).

The Flair library is used to load the dataset and a basic hyper-parameter search is performed. More details can be found [here]().

Here's an overview of the results on the development split - F1-Score over 5 runs is reported:

| Model    | Best Configuration  | F1-Score
| -------- | ------------------- | --------
| BERT     | `bs8-e10-lr5e-05`   | 85.92 ± 0.53
| ELECTRA  | `bs4-e10-lr5e-05`   | 85.53 ± 0.61
| ConvBERT | `bs4-e10-lr5e-05`   | **86.43** ± 0.82
| T5-Small | `bs4-e10-lr0.00016` | 84.12 ± 1.11
| T5-Base  | `bs4-e10-lr0.00016` | 85.58 ± 0.62
| T5-Large | `bs4-e10-lr0.00016` | 85.91 ± 1.09

For T5, encoder-only fine-tuning is performed.

# Acknowledgements

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC).
Many Thanks for providing access to the TPUs ❤️
