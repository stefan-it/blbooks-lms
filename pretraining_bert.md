# BERT Token Dropping Pretraining

This readme shows how to pretrain a BERT model with the token dropping approach, as described in ["Token Dropping for Efficient BERT Pretraining"](https://arxiv.org/abs/2203.13240).

# Corpus Chunking

We split the British Library corpus into 1GB chunks:

```bash
mkdir corpus
mv bl_1800-1900_extracted.txt corpus
cd corpus
split -C 1G bl_1800-1900_extracted.txt bl-
rm -f bl_1800-1900_extracted.txt
cd ..
```

# BERT Pretraining + Data Generation

We use the TensorFlow Models repository to create the pretraining data:

```bash
git clone https://github.com/tensorflow/models.git
cd models
git checkout v2.9.0

# \o/
export PYTHONPATH=$(pwd)
cd ..
```

In the next step, we use 5 processes with `xargs` magic in parallel to generate TFRecords for pretraining.
Each process will use up to 45GB of RAM and takes ~4.5 hours to complete. Please adjust the number of processes according to your hardware setup.
Parameters are mostly taken from the [official documentation](https://github.com/tensorflow/models/blob/master/official/nlp/docs/train.md) in TensorFlow models repo.

```bash
find ./corpus/ -type f | xargs -I% -P5 python3 models/official/nlp/data/create_pretraining_data.py \
--do_lower_case=False \
--max_seq_length=512 \
--max_predictions_per_seq=76 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=5 \
--input_file % \
--output_file %.tfrecord \
--vocab_file vocab.txt
```

# GCP Bucket

In the next step, we create a GCP bucket named `bl-lms` in the same zone as used for VM/TPU. Both "TPU Service Agent" and "Compute Engine Service Agent" should get "Storage-Administrator" permissions.
After creating a new folder "tfrecords", all previously created TFRecords can be uploaded via:

```bash
cd corpus
gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp *.tfrecord gs://bl-lms/tfrecords
cd ..
```

# VM + TPU Creation

We use a `n1-standard-2` VM, that can be created via:

```bash
gcloud compute instances create bert \
--zone=europe-west4-a \
--machine-type=n1-standard-2 \
--image-project=ml-images \
--image-family=tf-2-9-1 \
--scopes=cloud-platform
```

After that a `v3-32` TPU pod is created via:

```bash
gcloud compute tpus create bert \
--zone=europe-west4-a \
--accelerator-type=v3-32 \
--network=default \
--range=192.168.8.0/29 \
--version=2.9.1
```

# Pretraining

After ssh'ing onto the VM via:

```bash
gcloud compute ssh bert --zone europe-west4-a
```

all required dependencies/libraries needs to be installed first:

```bash
# Start interactive terminal session
tmux

# Dependencies
pip3 install gin-config pyyaml tensorflow_addons tensorflow_datasets sentencepiece tensorflow_hub sklearn seqeval sacrebleu 
pip3 install tensorflow_text

# TensorFlow Models repo
git clone https://github.com/tensorflow/models.git
cd models
export PYTHONPATH=$(pwd)
cd official/projects/token_dropping
```

# Configuration

The following configuration files needs to be created:

* `historic_english_base_token_drop.yaml` - storing all model configurations (hidden size, number of layers...)
* `historic_english_base_pretrain_sequence_pack.yaml` - storing all training configurations (number of epochs, batch size...)

The `historic_english_base_token_drop.yaml` looks like:

```yaml
task:
  model:
    encoder:
      type: any
      any:
        token_allow_list: !!python/tuple
        - 1  # [UNK]
        - 2  # [CLS]
        - 3  # [SEP]
        - 4  # [MASK]
        token_deny_list: !!python/tuple
        - 0  # [PAD]
        attention_dropout_rate: 0.1
        dropout_rate: 0.1
        hidden_activation: gelu
        hidden_size: 768
        initializer_range: 0.02
        intermediate_size: 3072
        max_position_embeddings: 512
        num_attention_heads: 12
        num_layers: 12
        type_vocab_size: 2
        vocab_size: 32000
        token_loss_init_value: 10.0
        token_loss_beta: 0.995
        token_keep_k: 256
```

and `historic_english_base_pretrain_sequence_pack.yaml`:

```yaml
task:
  init_checkpoint: ''
  model:
    cls_heads: []
  train_data:
    drop_remainder: true
    global_batch_size: 512
    input_path: "gs://bl-lms/tfrecords/*.tfrecord" 
    is_training: true
    max_predictions_per_seq: 76
    seq_length: 512
    use_next_sentence_label: false
    use_position_id: false
    use_v2_feature_names: false
trainer:
  checkpoint_interval: 25000
  max_to_keep: 300
  optimizer_config:
    learning_rate:
      polynomial:
        cycle: false
        decay_steps: 1000000
        end_learning_rate: 0.0
        initial_learning_rate: 0.0001
        power: 1.0
      type: polynomial
    optimizer:
      type: adamw
    warmup:
      polynomial:
        power: 1
        warmup_steps: 10000
      type: polynomial
  steps_per_loop: 1000
  summary_interval: 1000
  train_steps: 1000000
  validation_interval: 1000
  validation_steps: 64
```

# Launch pretraining

The pretraining command be launched via:

```bash
python3 train.py \
--experiment=token_drop_bert/pretraining \
--config_file=historic_english_base_pretrain_sequence_pack.yaml \
--config_file=historic_english_base_token_drop.yaml \
--params_override="runtime.distribution_strategy=tpu" \
--tpu=bert \
--model_dir=gs://bl-lms/models/bert-base-historic-english-td-cased \
--mode=train
```

# Subword Stats

The following table shows an overview of total number of subwords.
It includes the maximum number of total subwords seen during pretraining (which is a multiplication of pretraining steps, batch size and sequence length),
number of total subwords when subword-tokenizing the original corpus and the number of total subwords after generation of pretraining data, which include duplication factor.

| Method                       | Number of subwords
| -----------------------------| -----------------:
| Corpus                       |   5,944,948,004
| TFRecords (Dupe factor of 5) |  24,400,357,376
| Pretraining                  | 262,144,000,000
