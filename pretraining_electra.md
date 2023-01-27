# ELECTRA Pretraining

We use the official ELECTRA implementation to create TFRecords:

```bash
git clone https://github.com/google-research/electra.git
cd electra
git checkout 79111328070e491b287c307906701ebc61091eb2
```

In the next step, we create TFRecords with a sequence length of 512:

```bash
python3 build_pretraining_dataset.py \
--corpus-dir ../corpus/ \
--vocab-file ../vocab.txt \
--output-dir ../output-512 --max-seq-length 512 \
--num-processes 24 --no-lower-case
```

This will take ~1 hour using 24 processes.

# GCP Bucket

In the next step, we create a GCP bucket named `bl-electra` in the same zone as used for VM/TPU. Both "TPU Service Agent" and "Compute Engine Service Agent" should get "Storage-Administrator" permissions. After creating a new folder "tfrecords", all previously created TFRecords can be uploaded via:

```bash
gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp -r output-512 gs://bl-electra
```

The previously created vocab also needs to be stored on GCP:

```bash
gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp vocab.txt gs://bl-electra
```

# VM + TPU Creation

We use a `n1-standard-2` VM, that can be created via:

```bash
gcloud compute instances create electra \
--zone=europe-west4-a \
--machine-type=n1-standard-2 \
--image-project=ml-images \
--image-family=tf-1-15-4 \
--scopes=cloud-platform
```

After that a `v3-32` TPU pod is created via:

```bash
gcloud compute tpus create electra \
--zone=europe-west4-a \
--accelerator-type=v3-32 \
--network=default \
--range=192.168.9.0/29 \
--version=1.15.4
```

# Pretraining

After ssh'ing onto the VM via:

```bash
gcloud compute ssh electra --zone europe-west4-a
```

all required dependencies/libraries needs to be installed first:

```bash
# Start interactive terminal session
tmux

git clone https://github.com/google-research/electra.git
cd electra
git checkout 79111328070e491b287c307906701ebc61091eb2
```

# Configuration

We need to create an own configuration for the ELECTRA Base model than should be trained:

```bash
cd electra
> configure_pretraining.py
```

This clears the original configuration and a new one can be created:

```python
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config controlling hyperparameters for pre-training ELECTRA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


class PretrainingConfig(object):
  """Defines pre-training hyperparameters."""

  def __init__(self, model_name, data_dir, **kwargs):
    self.model_name = model_name
    self.debug = False  # debug mode for quickly running things
    self.do_train = True  # pre-train ELECTRA
    self.do_eval = False  # evaluate generator/discriminator on unlabeled data

    # loss functions
    self.electra_objective = True  # if False, use the BERT objective instead
    self.gen_weight = 1.0  # masked language modeling / generator loss
    self.disc_weight = 50.0  # discriminator loss
    self.mask_prob = 0.15  # percent of input tokens to mask out / replace

    # optimization
    self.learning_rate = 2e-4
    self.lr_decay_power = 1.0  # linear weight decay by default
    self.weight_decay_rate = 0.01
    self.num_warmup_steps = 10000

    # training settings
    self.iterations_per_loop = 200
    self.save_checkpoints_steps = 50000
    self.num_train_steps = 1000000
    self.num_eval_steps = 100
    self.keep_checkpoint_max = 0 # maximum number of recent checkpoint files to keep;
                                 # change to 0 or None to keep all checkpoints

    # model settings
    self.model_size = "base"  # one of "small", "base", or "large"
    # override the default transformer hparams for the provided model size; see
    # modeling.BertConfig for the possible hparams and util.training_utils for
    # the defaults
    self.model_hparam_overrides = (
        kwargs["model_hparam_overrides"]
        if "model_hparam_overrides" in kwargs else {})
    self.embedding_size = None  # bert hidden size by default
    self.vocab_size = 32000  # number of tokens in the vocabulary
    self.do_lower_case = False  # lowercase the input?

    # generator settings
    self.uniform_generator = False  # generator is uniform at random
    self.untied_generator_embeddings = False  # tie generator/discriminator
                                              # token embeddings?
    self.untied_generator = True  # tie all generator/discriminator weights?
    self.generator_layers = 1.0  # frac of discriminator layers for generator
    self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
    self.disallow_correct = False  # force the generator to sample incorrect
                                   # tokens (so 15% of tokens are always
                                   # fake)
    self.temperature = 1.0  # temperature for sampling from generator

    # batch sizes
    self.max_seq_length = 512
    self.train_batch_size = 256
    self.eval_batch_size = 128

    # TPU settings
    self.use_tpu = True 
    self.num_tpu_cores = 32
    self.tpu_job_name = None
    self.tpu_name = "electra"  # cloud TPU to use for training
    self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
    self.gcp_project = None  # project name for the Cloud TPU-enabled project

    # default locations of data files
    self.pretrain_tfrecords = os.path.join(
        data_dir, "output-512/pretrain_data.tfrecord*")
    self.vocab_file = os.path.join(data_dir, "vocab.txt")
    self.model_dir = os.path.join(data_dir, "models", model_name)
    results_dir = os.path.join(self.model_dir, "results")
    self.results_txt = os.path.join(results_dir, "unsup_results.txt")
    self.results_pkl = os.path.join(results_dir, "unsup_results.pkl")

    # update defaults with passed-in hyperparameters
    self.update(kwargs)

    self.max_predictions_per_seq = int((self.mask_prob + 0.005) *
                                       self.max_seq_length)

    # debug-mode settings
    if self.debug:
      self.train_batch_size = 8
      self.num_train_steps = 20
      self.eval_batch_size = 4
      self.iterations_per_loop = 1
      self.num_eval_steps = 2

    # defaults for different-sized model
    if self.model_size == "small":
      self.embedding_size = 128
    # Here are the hyperparameters we used for larger models; see Table 6 in the
    # paper for the full hyperparameters
    else:
      self.max_seq_length = 512
      self.learning_rate = 2e-4
      if self.model_size == "base":
        self.embedding_size = 768
        self.generator_hidden_size = 0.33333
        self.train_batch_size = 256
      else:
        self.embedding_size = 1024
        self.mask_prob = 0.25
        self.train_batch_size = 2048

    # passed-in-arguments override (for example) debug-mode defaults
    self.update(kwargs)

  def update(self, kwargs):
    for k, v in kwargs.items():
      if k not in self.__dict__:
        raise ValueError("Unknown hparam " + k)
      self.__dict__[k] = v
```

# Launch pretraining

The pretraining command be launched via:

```bash
python3 run_pretraining.py --data-dir gs://bl-electra --model-name electra-base-historic-english-cased
```
