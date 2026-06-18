# Modal Experiment Agent

This folder contains the Modal scripts used to launch BabyLM training experiments and upload completed runs to HuggingFace.

The Modal scripts are kept separate from the main training code. The actual model training logic lives in `train.py` at the repository root. The Modal agent handles remote execution: setting up the container, copying required files, launching training jobs, collecting logs, saving result summaries, calculating TEff when possible and committing outputs to a persistent Modal volume.

The scripts expect `train.py`, the model config file, and the shared tokenizer directory to be available at the repository root.

## Expected Repository Structure

```text
Language-Imbalance-BabyLM/
├── train.py
├── large_config.json
├── shared_tokenizer/
├── babylm-modal-agent/
│   ├── modal_agent.py
│   ├── upload_existing.py
│   └── README.md
└── ...
```

## Files

`modal_agent.py`

Runs monolingual and bilingual BabyLM experiments on Modal. It supports English, Dutch, Indonesian, and Javanese datasets from the BabyLM-community Hugging Face collection.

The script can run:

* monolingual baselines
* bilingual language-pair experiments
* different language ratios
* different learning rates, epoch counts, batch sizes, token budgets, and model configurations

Outputs are written to the Modal volume `babylm-teff-results`.

If matching bilingual and monolingual runs are launched in the same Modal run, the agent also calculates token efficiency:

```text
TEff = (monolingual PPL / bilingual PPL) / token share
```

The agent saves:

* `run_summary.json` inside each model output folder
* `teff_summary.json` inside each bilingual output folder when matching monolingual baselines exist
* `combined_metrics.json` under `/results/experiment_summaries/`

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

Run from the repository root:

```bash
modal run babylm-modal-agent/modal_agent.py
```

Inside `modal_agent.py`, choose which runs to launch by editing:

```python
RUN_BILINGUAL = True
RUN_MONOLINGUAL = False
```

Both can be set to `True` if monolingual and bilingual jobs should be launched in the same Modal run. This is required if TEff should be calculated automatically for that run.

Example:

```python
RUN_BILINGUAL = True
RUN_MONOLINGUAL = True
```

The bilingual and monolingual trial lists are edited directly inside `main()`.

## Uploading Completed Runs

Run from the repository root:

```bash
modal run babylm-modal-agent/upload_existing.py
```

Add completed Modal result folders to the `runs` list in `upload_existing.py`.

Recommended naming convention:

```text
large-eng-nld-90-10-50m-8ep-2e-4-32b
```

Format:

```text
config-languages-ratio-tokenbudget-epochs-learningrate-batchsize
```

Examples:

```text
large-eng-nld-90-10-50m-8ep-2e-4-32b
large-eng-jav-90-10-50m-8ep-2e-4-32b
large-eng-50m-8ep-2e-4-32b
large-nld-50m-8ep-2e-4-32b
```

## Notes

The Modal agent is not an autonomous model-training system. It is an experiment orchestration script. It prepares the remote environment, starts training jobs, captures logs, extracts evaluation metrics, optionally calculates TEff, and stores outputs.

For valid TEff comparisons, the bilingual model and the corresponding monolingual baselines should match in model configuration, token budget, learning rate, epochs, batch size, vocabulary size, tokenizer, and seed. Only the language setup should differ.

The training logic, dataset loading, token packing, model initialization, and evaluation are implemented in the root-level `train.py`.
