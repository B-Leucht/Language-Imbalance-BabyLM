import os
import argparse
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_dataset
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


def train_tokenizer(dataset, tokenizer_dir, vocab_size, min_frequency=2):
    texts = dataset["text"]
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    with open(tokenizer_dir / "training_text.txt", "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.strip() + "\n")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=str(tokenizer_dir / "training_text.txt"),
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
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")

    args = parser.parse_args()

    print("📥 Loading dataset...")
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
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_steps=500,
        save_total_limit=2,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.model_name,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
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