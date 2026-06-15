# Modal Experiment Agent

This folder contains the Modal scripts used to launch the BabyLM training experiments conducted by us and upload completed runs to Hugging Face.

The Modal scripts are intentionally kept separate from the main training code. The actual model training logic lives in `train.py` at the repository root. The Modal agent only handles remote execution: setting up the container, copying the required files, launching training jobs, collecting logs, saving summaries, and committing results to a persistent Modal volume.

The Modal scripts expect `train.py`, the model config files, and the tokenizer directory to be available at the repository root.

## Files

`modal_agent.py`

Runs monolingual and bilingual BabyLM experiments on Modal. It supports English, Dutch, Indonesian, and Javanese datasets from the BabyLM-community Hugging Face collection.

The script can run:

* monolingual baselines
* bilingual language-pair experiments
* different language ratios
* different learning rates, epoch counts, and batch sizes

Outputs are written to the Modal volume `babylm-teff-results`.

`upload_existing.py`

Uploads completed model folders from the Modal volume to Hugging Face. The Hugging Face username is read from the Modal secret instead of being hardcoded.

## Modal Setup

Create or reuse the Modal volume:

```bash
modal volume create babylm-teff-results
```

Create the Hugging Face secret:

```bash
modal secret create huggingface-token HF_TOKEN=your_hf_token HF_USERNAME=your_hf_username
```

The token needs read access for gated BabyLM datasets. If models should be uploaded to Hugging Face, the token also needs write access.

## Running Experiments

From the repository root:

```bash
modal run modal_agent/modal_agent.py
```

Inside `modal_agent.py`, choose which runs to launch by editing:

```python
RUN_BILINGUAL = True
RUN_MONOLINGUAL = False
```

Both can be set to `True` if monolingual and bilingual jobs should be launched in the same Modal run.

## Uploading Completed Runs

From the repository root:

```bash
modal run modal_agent/upload_existing.py
```

Add completed Modal result folders to the `runs` list in `upload_existing.py`.

Recommended naming convention:

```text
large-eng-nld-90-10-50m-8ep-lr3e-4-32b
```

Format:

```text
config-languages-ratio-tokenbudget-epochs-learningrate-batchsize
```

Examples:

```text
large-eng-nld-90-10-50m-8ep-lr3e-4-32b
large-eng-50m-8ep-lr3e-4-32b
...
```

## Notes

The Modal agent is not an autonomous model-training system. It is an experiment orchestration script. It prepares the remote environment, starts training jobs, captures logs, extracts evaluation metrics, and stores outputs.

The training logic, dataset loading, token packing, model initialization, and evaluation are implemented in the root-level `train.py`.
