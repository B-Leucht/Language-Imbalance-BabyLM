import os
import argparse
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
#from huggingface_hub import HfFolder
from transformers import XLMRobertaTokenizerFast
import sentencepiece as spm
import torch
import math

torch.set_float32_matmul_precision('high')
if torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# replaced ByteLevelBPETokenizer with SentencePiece
def train_tokenizer(dataset, tokenizer_dir, vocab_size):
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # SentencePiece needs a plain text file as input
    tmp_path = str(tokenizer_dir / "train_input.txt")
    print("Writing text to temp file for tokenizer training...")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in dataset["text"]:
            if line.strip():
                f.write(line + "\n")

    sp_model_prefix = str(tokenizer_dir / "shared_spm")

    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=sp_model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,  # high coverage for multilingual
        pad_id=0,
        unk_id=3,
        bos_id=1,
        eos_id=2,
        pad_piece="<pad>",
        bos_piece="<s>",
        eos_piece="</s>",
        unk_piece="<unk>",
        add_dummy_prefix=True,       # preserve whitespace info
        remove_extra_whitespaces=True,
        num_threads=os.cpu_count(),
    )

    #wrap SentencePiece model in roberta 
    fast_tokenizer = XLMRobertaTokenizerFast(
        vocab_file=sp_model_prefix + ".model",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    fast_tokenizer.save_pretrained(str(tokenizer_dir))
    return fast_tokenizer


def tokenize_function(example, tokenizer, max_length):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_overflowing_tokens=False,
    )


#added per-language perplexity evaluation 
def evaluate_per_language(trainer, datasets_per_language, tokenizer, max_length, args):
    print("\n📊 Per-language evaluation:")
    results = {}
    for lang_name, lang_dataset in datasets_per_language.items():
        lang_dataset = lang_dataset.select(range(min(500, len(lang_dataset))))
        tokenized = lang_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer, max_length),
            batched=True,
            remove_columns=lang_dataset.column_names,
        )
        metrics = trainer.evaluate(eval_dataset=tokenized)
        loss = metrics.get("eval_loss", None)
        perplexity = math.exp(loss) if loss is not None else None
        results[lang_name] = {"loss": loss, "perplexity": perplexity}
        print(f"  {lang_name}: perplexity = {perplexity:.2f}")
    return results


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train a Causal Language Model from scratch")
    parser.add_argument("--dataset", type=str, required=True, nargs="+", help="HuggingFace dataset names (space separated)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model outputs")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model repo name")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size for tokenizer")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size per device")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-04, help="Learning rate")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ratios", type=float, nargs="+", default=None,
                        help="Sampling ratios per dataset e.g. 0.9 0.1 for 90/10 split")
    parser.add_argument("--max_words", type=int, default=None,
                        help="Total word budget e.g. 10000000 for 10M words")
    parser.add_argument("--tokenizer_dir", type=str, default=None,
                        help="Load existing tokenizer from this dir instead of retraining")

    args = parser.parse_args()

    print("📥 Loading datasets...")
    raw_datasets = {}
    for dataset_name in args.dataset:
        lang_key = dataset_name.split("/")[-1]  # e.g. "BabyLM-2026-Strict" or "babylm-jav"
        raw_datasets[lang_key] = load_dataset(dataset_name, split="train")
        print(f"  {lang_key}: {len(raw_datasets[lang_key])} rows")

    # Apply ratios if provided
    if args.ratios is not None:
        assert len(args.ratios) == len(args.dataset), "Number of ratios must match number of datasets"
        assert abs(sum(args.ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

        # Figure out total rows to sample based on word budget
        # Rough estimate: 1 row ≈ 15 words on average
        avg_words_per_row = 15
        if args.max_words is not None:
            total_rows = args.max_words // avg_words_per_row
        else:
            total_rows = sum(len(d) for d in raw_datasets.values())

        sampled = []
        for (lang_key, lang_dataset), ratio in zip(raw_datasets.items(), args.ratios):
            n_rows = int(total_rows * ratio)
            n_rows = min(n_rows, len(lang_dataset))  # can't sample more than available
            sampled_ds = lang_dataset.shuffle(seed=args.seed).select(range(n_rows))
            sampled.append(sampled_ds)
            print(f"  Sampled {n_rows} rows from {lang_key} ({ratio*100:.0f}%)")

        dataset = concatenate_datasets(sampled).shuffle(seed=args.seed)

    else:
        # No ratios given — just concatenate everything equally
        all_datasets = list(raw_datasets.values())
        if args.max_words is not None:
            avg_words_per_row = 15
            total_rows = args.max_words // avg_words_per_row
            rows_per_lang = total_rows // len(all_datasets)
            all_datasets = [d.shuffle(seed=args.seed).select(range(min(rows_per_lang, len(d)))) for d in all_datasets]
        dataset = concatenate_datasets(all_datasets).shuffle(seed=args.seed)

    print(f"Total training rows after sampling: {len(dataset)}")

    if args.tokenizer_dir and Path(args.tokenizer_dir).exists():
        print(f"🔡 Loading existing tokenizer from {args.tokenizer_dir}...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    else:
        print("🔡 Training SentencePiece tokenizer...")
        tokenizer_out = args.tokenizer_dir if args.tokenizer_dir else args.output_dir
        tokenizer = train_tokenizer(dataset, tokenizer_out, args.vocab_size)

    print("🧹 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    print("🔧 Loading model config and initializing model...")
    config = AutoConfig.from_pretrained(args.config)
    config.vocab_size = tokenizer.vocab_size
    config._name_or_path = ""
    model = AutoModelForCausalLM.from_config(config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print(args)

    training_args = TrainingArguments(
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        gradient_accumulation_steps=1,
        hub_model_id=args.model_name,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.output_dir, "logs"),
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        push_to_hub=args.push_to_hub,
        remove_unused_columns=False,
        save_strategy="no",
        logging_strategy="epoch",
        eval_strategy="epoch",
    )

    splits = tokenized_dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Starting training...")
    trainer.train()

    #per-language evaluation after training
    print("Running per-language evaluation...")
    evaluate_per_language(trainer, raw_datasets, tokenizer, args.max_length, args)

    if args.push_to_hub:
        print("☁️ Pushing to Hugging Face Hub...")
        trainer.push_to_hub(args.model_name)


if __name__ == "__main__":
    main()
