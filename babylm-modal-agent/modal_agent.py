import subprocess
import json
from pathlib import Path

import modal

app = modal.App("babylm-teff-agent")
volume = modal.Volume.from_name("babylm-teff-results", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token")

LANG_TO_DATASET = {
    "eng": "BabyLM-community/babylm-eng",
    "nld": "BabyLM-community/babylm-nld",
    "ind": "BabyLM-community/babylm-ind",
    "jav": "BabyLM-community/babylm-jav",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tokenizers",
        "accelerate",
        "python-dotenv",
        "huggingface_hub",
        "pandas",
        "numpy",
    )
    .add_local_file("train.py", "/root/train.py")
    .add_local_file("large_config.json", "/root/large_config.json")
    #.add_local_file("small_config.json", "/root/small_config.json")
    .add_local_dir("shared_tokenizer", "/root/shared_tokenizer")
)

def lr_to_label(lr: float) -> str:
    mapping = {
        #1e-4: "1e-4",
        2e-4: "2e-4",
        #3e-4: "3e-4",
        #...
    }
    return mapping.get(lr, str(lr))


def extract_ppl(eval_results: dict, preferred_lang: str):
    possible_keys = [
        preferred_lang,
        f"babylm-{preferred_lang}",
        f"BabyLM-community/babylm-{preferred_lang}",
    ]

    for key in possible_keys:
        if key in eval_results:
            return eval_results[key].get("perplexity"), key

    for key, value in eval_results.items():
        if preferred_lang in key and isinstance(value, dict):
            return value.get("perplexity"), key

    return None, None

def run_training_command(cmd, run_name: str):
     
    print("Running command:") 
    print(" ".join(cmd))
    
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    print("RETURN CODE:", result.returncode)
    print("STDOUT:")
    print(result.stdout[-12000:])
    print("STDERR:")
    print(result.stderr[-12000:])

    if result.returncode != 0:
        raise RuntimeError(
            f"Run failed: {run_name}\n"
            f"Return code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout[-12000:]}\n\n"
            f"STDERR:\n{result.stderr[-12000:]}"
        )

    return result

#MONOLINGUAL TRAINING FOR BASELINES

@app.function(
    image=image,
    gpu="A100", #change gpu type
    timeout=10 * 60 * 60,
    volumes={"/results": volume},
    secrets=[hf_secret],
)

def run_mono_trial(
    lang: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
):

    lr_label = lr_to_label(learning_rate)

    run_name = f"large-{lang}-30m-{epochs}ep-{lr_label}-{batch_size}b"
    output_dir = f"/results/{run_name}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "/root/train.py",
        "--dataset", LANG_TO_DATASET[lang],
        "--config", "/root/large_config.json", #change config size
        "--tokenizer_dir", "/root/shared_tokenizer",
        "--output_dir", output_dir,
        "--model_name", run_name,
        "--max_tokens", "30000000", #change token size
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--vocab_size", "30000",
        "--seed", "42",
    ]

    run_training_command(cmd, run_name)

    eval_path = Path(output_dir) / "per_language_eval.json"

    metrics = {
        "run_name": run_name,
        "config": "large",
        "lang": lang,
        "tokens": 30_000_000,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "output_dir": output_dir, 
    }

    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

        print("Full eval_results:")
        print(json.dumps(eval_results, indent=2))

        print("Eval result keys:")
        print(list(eval_results.keys()))

        ppl, used_key = extract_ppl(eval_results, preferred_lang=lang)

        metrics[f"{lang}_ppl"] = ppl
        metrics[f"{lang}_ppl_key"] = used_key

        print(f"PERPLEXITY_RESULT run={run_name} key={used_key} {lang}_ppl={ppl}")

    else:
        metrics["warning"] = "per_language_eval.json not found"
        print(f"WARNING: per_language_eval.json not found for {run_name}")

    summary_path = Path(output_dir) / "run_summary.json"

    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Run summary:")
    print(json.dumps(metrics, indent=2))

    volume.commit()

    return metrics


#BILINGUAL TRAINING

@app.function(
    image=image,
    gpu="A100",  #change gpu type
    timeout=10 * 60 * 60,
    volumes={"/results": volume},
    secrets=[hf_secret],
)
def run_bilingual_trial(
    lang1: str,
    lang2: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    lang1_ratio: float,
    lang2_ratio: float,
):

    lr_label = lr_to_label(learning_rate)
    ratio_label = f"{int(lang1_ratio * 100)}-{int(lang2_ratio * 100)}"   
    lang_label = f"{lang1}-{lang2}"

    run_name = f"large-{lang_label}-{ratio_label}-30m-{epochs}ep-{lr_label}-{batch_size}b" 
    output_dir = f"/results/{run_name}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "/root/train.py",
        "--dataset",
        LANG_TO_DATASET[lang1], 
        LANG_TO_DATASET[lang2],
        "--ratios",
        str(lang1_ratio),
        str(lang2_ratio),
        "--config", "/root/large_config.json",
        "--tokenizer_dir", "/root/shared_tokenizer",
        "--output_dir", output_dir,
        "--model_name", run_name,
        "--max_tokens", "30000000",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--vocab_size", "30000",
        "--seed", "42",
    ]

    run_training_command(cmd, run_name)

    eval_path = Path(output_dir) / "per_language_eval.json"

    metrics = {
        "run_name": run_name,
        "config": "large",
        "langs": lang_label,
        "ratio": ratio_label,
        f"{lang1}_ratio": lang1_ratio,
        f"{lang2}_ratio": lang2_ratio,
        "tokens": 30_000_000,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "output_dir": output_dir,
    }

    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

        print("Full eval_results:")
        print(json.dumps(eval_results, indent=2))

        print("Eval result keys:")
        print(list(eval_results.keys()))

        lang1_ppl, lang1_key = extract_ppl(eval_results, preferred_lang=lang1) 
        lang2_ppl, lang2_key = extract_ppl(eval_results, preferred_lang=lang2)

        metrics[f"{lang1}_ppl"] = lang1_ppl
        metrics[f"{lang2}_ppl"] = lang2_ppl
        metrics[f"{lang1}_ppl_key"] = lang1_key
        metrics[f"{lang2}_ppl_key"] = lang2_key

        print(
            f"PERPLEXITY_RESULT run={run_name} " 
            f"{lang1}_key={lang1_key} {lang1}_ppl={lang1_ppl} " 
            f"{lang2}_key={lang2_key} {lang2}_ppl={lang2_ppl}" 
            )

    else:
        metrics["warning"] = "per_language_eval.json not found"
        print(f"WARNING: per_language_eval.json not found for {run_name}")

    summary_path = Path(output_dir) / "run_summary.json"

    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Run summary:")
    print(json.dumps(metrics, indent=2))

    volume.commit()

    return metrics

#LOCAL ENTRYPOINT
# Both RUN_BILINGUAL and RUN_MONOLINGUAL can be True in the same Modal run.

@app.local_entrypoint()
def main():
    RUN_BILINGUAL = True #set False when you only want monolingual
    RUN_MONOLINGUAL = True  #set False when you only want bilingual

    jobs = []

    #bilingual runs
    if RUN_BILINGUAL:
        bilingual_trials = [
            # lang1, lang2, lr, epochs, batch_size, lang1_ratio, lang2_ratio
            ("eng", "nld", 2e-4, 8, 32, 0.90, 0.10),
           # ("eng", "nld", 3e-4, 8, 32, 0.50, 0.50),
           # ...
        ]

        for lang1, lang2, lr, epochs, batch_size, lang1_ratio, lang2_ratio in bilingual_trials:
            job = run_bilingual_trial.spawn(
                lang1=lang1,
                lang2=lang2,
                learning_rate=lr,
                epochs=epochs,
                batch_size=batch_size,
                lang1_ratio=lang1_ratio,
                lang2_ratio=lang2_ratio,
            )
            jobs.append(("bilingual", job))

        print(f"Started {len(bilingual_trials)} bilingual jobs.")

    #monolingual runs
    if RUN_MONOLINGUAL:
        mono_trials = [
            # lang, lr, epochs, batch_size
            ("eng", 2e-4, 8, 32),
            ("nld", 2e-4, 8, 32),
            # ("eng", 2e-4, 8, 32),
            # ...
        ]

        for lang, lr, epochs, batch_size in mono_trials:
            job = run_mono_trial.spawn(
                lang=lang,
                learning_rate=lr,
                epochs=epochs,
                batch_size=batch_size,
            )
            jobs.append(("monolingual", job))

        print(f"Started {len(mono_trials)} monolingual jobs.")


    #collect all results
    print(f"Started {len(jobs)} total Modal jobs.")

    for job_type, job in jobs:
        try:
            result = job.get()
            print(f"Finished {job_type}:")
            print(result)
        except Exception as e:
            print(f"Failed {job_type}:")
            print(e)
