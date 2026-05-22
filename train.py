import os
import argparse
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    default_data_collator,
    Trainer,
    TrainingArguments,
)
<<<<<<< HEAD
#from huggingface_hub import HfFolder
from transformers import XLMRobertaTokenizerFast
import sentencepiece as spm
=======
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace, Sequence as PreSeq, WhitespaceSplit
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.processors import TemplateProcessing
import random
import numpy as np
>>>>>>> origin/main
import torch
import math

torch.set_float32_matmul_precision("high")
if torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_tokenizer(dataset, tokenizer_dir, vocab_size, seed):
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = PreSeq([WhitespaceSplit(), Metaspace(replacement="▁", prepend_scheme="always")])
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    def batch_iter(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield [t for t in dataset["text"][i : i + batch_size] if t.strip()]

    tokenizer.train_from_iterator(batch_iter(), trainer=trainer, length=len(dataset))

    bos, pad, eos, unk = "<s>", "<pad>", "</s>", "<unk>"
    tokenizer.post_processor = TemplateProcessing(
        single=f"{bos} $A {eos}",
        pair=f"{bos} $A {eos} {eos} $B {eos}",
        special_tokens=[(bos, tokenizer.token_to_id(bos)), (eos, tokenizer.token_to_id(eos))],
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=bos,
        eos_token=eos,
        unk_token=unk,
        pad_token=pad,
        mask_token="<mask>",
    )

    fast_tokenizer.save_pretrained(str(tokenizer_dir))
    return fast_tokenizer


def pack_dataset(dataset, tokenizer, max_length, max_chunks=None):
    """Tokenize documents and pack them into fixed-length chunks with no padding or truncation."""
    def tokenize_batch(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )

    def chunk(examples):
        all_ids = sum(examples["input_ids"], [])
        total = (len(all_ids) // max_length) * max_length
        chunks = [all_ids[i : i + max_length] for i in range(0, total, max_length)]
        return {"input_ids": chunks, "attention_mask": [[1] * max_length] * len(chunks), "labels": chunks}

    packed = tokenized.map(chunk, batched=True, remove_columns=tokenized.column_names)
    if max_chunks is not None and len(packed) > max_chunks:
        packed = packed.select(range(max_chunks))
    return packed


def evaluate_per_language(trainer, tokenized_eval_per_lang):
    print("\n📊 Per-language evaluation:")
    results = {}
    for lang_name, eval_ds in tokenized_eval_per_lang.items():
        metrics = trainer.evaluate(eval_dataset=eval_ds)
        loss = metrics["eval_loss"]
        ppl = math.exp(loss)
        results[lang_name] = {"loss": loss, "perplexity": ppl}
        print(f"  {lang_name}: perplexity = {ppl:.2f}")
    return results


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Train a Causal Language Model from scratch"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        nargs="+",
        help="HuggingFace dataset names (space separated)",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save model outputs",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace model repo name"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30000, help="Vocabulary size for tokenizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size per device"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-04, help="Learning rate"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push model to HuggingFace Hub"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=None,
        help="Sampling ratios per dataset e.g. 0.9 0.1 for 90/10 split",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Total training token budget (sequences × max_length), e.g. 10000000",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Load existing tokenizer from this dir instead of retraining",
    )

    args = parser.parse_args()

    print("📥 Loading datasets...")
    raw_datasets = {}
    for dataset_name in args.dataset:
        lang_key = dataset_name.split("/")[-1]
        raw_datasets[lang_key] = load_dataset(dataset_name, split="train")
        print(f"  {lang_key}: {len(raw_datasets[lang_key])} rows")

    # Per-language eval/pool split — must happen before any sampling so eval rows
    # are identical across all runs regardless of --max_tokens.
    per_lang_eval = {}
    per_lang_pool = {}
    for lang_key, lang_dataset in raw_datasets.items():
        shuffled = lang_dataset.shuffle(seed=args.seed)
        n_eval = min(max(100, int(len(shuffled) * 0.05)), 5000)
        per_lang_eval[lang_key] = shuffled.select(range(n_eval))
        per_lang_pool[lang_key] = shuffled.select(range(n_eval, len(shuffled)))
        print(f"  {lang_key}: {n_eval} eval rows, {len(shuffled) - n_eval} pool rows")

    if args.ratios is not None:
        assert len(args.ratios) == len(args.dataset), (
            "Number of ratios must match number of datasets"
        )
        assert abs(sum(args.ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Store pools and ratios for post-packing chunk selection
    lang_pools = list(per_lang_pool.items())
    ratios = args.ratios if args.ratios is not None else [1 / len(lang_pools)] * len(lang_pools)

    # Use full pools for tokenizer training (raw text, no filtering)
    dataset = concatenate_datasets([p for _, p in lang_pools]).shuffle(seed=args.seed)
    print(f"Source documents available: {len(dataset)}")

    if args.tokenizer_dir and Path(args.tokenizer_dir).exists():
        print(f"🔡 Loading existing tokenizer from {args.tokenizer_dir}...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    else:
        print("🔡 Training SentencePiece tokenizer...")
        tokenizer_out = args.tokenizer_dir if args.tokenizer_dir else args.output_dir
        tokenizer = train_tokenizer(dataset, tokenizer_out,
                                    args.vocab_size, args.seed)

    if args.epochs == 0:
        print("✅ Tokenizer saved. Exiting (--epochs 0).")
        return

    print("🧹 Packing dataset into chunks...")
    total_chunks = args.max_tokens // args.max_length if args.max_tokens else None
    packed_per_lang = []
    for (lang_key, lang_pool), ratio in zip(lang_pools, ratios):
        n_chunks = int(total_chunks * ratio) if total_chunks else None
        packed = pack_dataset(lang_pool, tokenizer, args.max_length, max_chunks=n_chunks)
        actual = len(packed)
        wanted = n_chunks or actual
        if n_chunks and actual < n_chunks:
            print(f"  ⚠️  {lang_key}: wanted {n_chunks} chunks but only {actual} available ({actual * args.max_length:,} tokens)")
        else:
            print(f"  {lang_key}: {actual} chunks ({actual * args.max_length:,} tokens, {ratio*100:.0f}%)")
        packed_per_lang.append(packed)
    tokenized_dataset = concatenate_datasets(packed_per_lang).shuffle(seed=args.seed)
    print(f"Total training chunks: {len(tokenized_dataset)} ({len(tokenized_dataset) * args.max_length:,} tokens)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("🔧 Loading model config and initializing model...")
    config = AutoConfig.from_pretrained(args.config)
    config.vocab_size = tokenizer.vocab_size
    config._name_or_path = ""
    model = AutoModelForCausalLM.from_config(config)

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
        eval_strategy="no",
        seed=args.seed,
    )

    print("🧹 Tokenizing eval sets...")
    tokenized_per_lang_eval = {}
    for lang_key, eval_ds in per_lang_eval.items():
        tokenized_per_lang_eval[lang_key] = pack_dataset(
            eval_ds, tokenizer, args.max_length, max_chunks=5000
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    print("🚀 Starting training...")
    trainer.train()

    print("💾 Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Running per-language evaluation...")
    evaluate_per_language(trainer, tokenized_per_lang_eval)

    if args.push_to_hub:
        print("☁️ Pushing to Hugging Face Hub...")
        trainer.push_to_hub(args.model_name)


if __name__ == "__main__":
    main()
