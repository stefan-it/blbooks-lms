# NER Fine-Tuning

We use Flair for fine-tuning NER models on the [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)
dataset from [HIPE-2022 Shared Task](https://hipe-eval.github.io/HIPE-2022/).

All models are fine-tuned on A10 (24GB) instances from [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) using Flair:

```bash
$ git clone https://github.com/flairNLP/flair.git
$ cd flair
$ git checkout dbc15695f9b9f3127b690ffce7f58495b56fa8de
$ pip3 install -e .
```

We use a config-driven hyper-parameter search. The script `flair-fine-tuner.py` can be used to fine-tune NER models with our Model Zoo.

All configurations can be found under the `configs/ajmc` folder in this repository.

Example command for hyper-parameter search for the BERT model:

```bash
$ python3 flair-fine-tuner.py ./configs/ajmc_bert_blbooks.json
```

# Results

To calculate the results (e.g. mean over 5 runs for all hyper-parameter configurations), the `flair-log-parser.py` script can be used.
Example output for a finished hyper-param search for the BERT model:

```bash
$ python3 flair-log-parser.py "hipe2022-flert-fine-tune-ajmc/en-bigscience-historical-texts/bert-base-blbooks-cased-*"
```

This outputs:

```bash
Debug: defaultdict(<class 'list'>, {'wsFalse-bs4-e10-lr5e-05': [0.8548, 0.8456, 0.8476, 0.8711, 0.8475], 
'wsFalse-bs4-e10-lr3e-05': [0.8709, 0.8551, 0.8517, 0.8605, 0.85], 'wsFalse-bs8-e10-lr5e-05': [0.866, 0.8599, 0.8612, 0.8496, 0.8592],
'wsFalse-bs8-e10-lr3e-05': [0.8531, 0.838, 0.8473, 0.8687, 0.8514]})

Averaged Development Results:
wsFalse-bs8-e10-lr5e-05 : 85.92
wsFalse-bs4-e10-lr3e-05 : 85.76
wsFalse-bs4-e10-lr5e-05 : 85.33
wsFalse-bs8-e10-lr3e-05 : 85.17

Best configuration: wsFalse-bs8-e10-lr5e-05

Best Development Score: 85.92
```

# Performance comparison

A more detailed table shows the performance overview for all 5 runs on development dataset:

| Model    | Best Configuration  | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Averaged F1-Score
| -------- | ------------------- | ----- | ----- | ----- | ----- | ----- | -----------------
| BERT     | `bs8-e10-lr5e-05`   | 86.60 | 85.99 | 86.12 | 84.96 | 85.92 | 85.92 ± 0.53
| ELECTRA  | `bs4-e10-lr5e-05`   | 85.75 | 85.58 | 86.50 | 85.14 | 84.68 | 85.53 ± 0.61
| ConvBERT | `bs4-e10-lr5e-05`   | 87.59 | 86.02 | 86.17 | 87.12 | 85.27 | **86.43** ± 0.82
| T5-Small | `bs4-e10-lr0.00016` | 84.81 | 83.57 | 84.46 | 85.48 | 82.26 | 84.12 ± 1.11
| T5-Base  | `bs4-e10-lr0.00016` | 86.81 | 85.14 | 85.24 | 85.44 | 85.27 | 85.58 ± 0.62
| T5-Large | `bs4-e10-lr0.00016` | 86.95 | 86.56 | 86.46 | 83.89 | 85.68 | 85.91 ± 1.09
