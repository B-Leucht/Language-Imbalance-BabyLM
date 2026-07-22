## Overview
This repository relates to the training pipeline that was used in our research work 'When Less is More: How Language Imbalance Helps Low-Resource Languages in a BabyLM Setting'. The initial code structure was forked from the multilingual training repository, made available by the BabyLM community, as described below. 

Monolingual and multilingual training can be started via experiments.ipynb. 
We included the shared tokenizer (shared_tokenizer) that was used along our experiments and is computed across the following four languages, that are part of the BabyBabelLM corpora:
- English (https://huggingface.co/datasets/BabyLM-community/babylm-eng)
- Dutch (https://huggingface.co/datasets/BabyLM-community/babylm-nld)
- Indonesian (https://huggingface.co/datasets/BabyLM-community/babylm-ind)
- Javanese (https://huggingface.co/datasets/BabyLM-community/babylm-jav)



## Multilingual BabyLM Training
This repository provides a simple script for training baseline models on the BabyBabelLM corpora.

Model training is done using the HuggingFace trainer. A model configuration can be defined in a `.json` file.

A HuggingFace token is expected to be stored in a `.env` file in the current working directory, which is necessary if pushing to private HF repositories.

Models can be trained on a union of HF datasets by providing multiple space-separated dataset names.

For example, to train a bilingual model on English and Norwegian we can run the following script:
```bash
python train.py \
  --dataset BabyLM-community/babylm-nor BabyLM-community/babylm-eng \
  --config ./small_config.json \
  --output_dir ./nor-eng-baseline-small \
  --model_name BabyLM-community/nor-eng-baseline-small \
  --push_to_hub
```
