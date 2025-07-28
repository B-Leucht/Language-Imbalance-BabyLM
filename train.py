import os
import argparse
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from tokenizers import ByteLevelBPETokenizer
from huggingface_hub import HfFolder
import torch


torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_tokenizer(dataset, tokenizer_dir, vocab_size, min_frequency=2, save_training_text=False):
    texts = dataset["text"]
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    def dataset_generator():
        for line in dataset["text"]:
            if line.strip():
                yield line
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        dataset_generator(),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    tokenizer.save_model(str(tokenizer_dir))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    fast_tokenizer.save_pretrained(str(tokenizer_dir))

    return fast_tokenizer


def tokenize_function(example, tokenizer, max_length):
    return tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length,
        return_overflowing_tokens=True,
    )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train a Causal Language Model from scratch")
    parser.add_argument("--dataset", type=str, required=True, help="Hugging Face dataset name or path")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model outputs")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model repo name (e.g., user/model)")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size for tokenizer")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size per device")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-04, help="Learning rate")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("📥 Loading dataset...")
    if " " in args.dataset:
        datasets = []
        for dataset_name in args.dataset.split():
            dataset = load_dataset(dataset_name, split="train")
            datasets.append(dataset)
        dataset = concatenate_datasets(datasets)
    else:
        dataset = load_dataset(args.dataset, split="train")

    print("🔡 Training BPE tokenizer...")
    tokenizer = train_tokenizer(dataset, args.output_dir, args.vocab_size)
    tokenizer.save_pretrained(args.output_dir)

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
        bf16=True,
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
    
    if args.push_to_hub:
        print("☁️ Pushing to Hugging Face Hub...")
        trainer.push_to_hub(args.model_name)


if __name__ == "__main__":
    main()