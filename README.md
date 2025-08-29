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
