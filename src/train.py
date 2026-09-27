from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "gpt2"

DATA_PATH = "data/training_data.txt"

OUTPUT_PATH = "./gpt2-prodigy-task01-final"

MAX_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 5e-5


# --------------------------------------------------
# Load training data
# --------------------------------------------------

def load_training_data():
    """Read training examples from the text file."""

    with open(DATA_PATH, "r", encoding="utf-8") as file:

        examples = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print(f"Training examples loaded: {len(examples)}")

    return examples


# --------------------------------------------------
# Create dataset
# --------------------------------------------------

def create_dataset(examples):
    """Create a Hugging Face Dataset."""

    dataset = Dataset.from_dict({
        "text": examples
    })

    print("Dataset created successfully!")
    print(dataset)

    return dataset


# --------------------------------------------------
# Load tokenizer and model
# --------------------------------------------------

def load_model():

    print("\nLoading GPT-2...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 does not have a separate padding token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    print("GPT-2 loaded successfully!")
    print("Number of parameters:", model.num_parameters())

    return tokenizer, model


# --------------------------------------------------
# Tokenize dataset
# --------------------------------------------------

def tokenize_dataset(dataset, tokenizer):
    """Convert text into token IDs."""

    def tokenize_function(examples):

        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True
    )

    # Remove original text column
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["text"]
    )

    print("Dataset tokenized successfully!")

    return tokenized_dataset


# --------------------------------------------------
# Train model
# --------------------------------------------------

def train_model(
    model,
    tokenizer,
    tokenized_dataset
):
    """Fine-tune GPT-2 on the custom dataset."""

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./gpt2-prodigy-task01",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("\nStarting GPT-2 fine-tuning...")
    print("=" * 60)

    training_result = trainer.train()

    print("=" * 60)
    print("Fine-tuning completed!")

    print("\nTraining Summary:")
    print(training_result.metrics)

    return trainer


# --------------------------------------------------
# Save model
# --------------------------------------------------

def save_model(trainer, tokenizer):
    """Save the fine-tuned model and tokenizer."""

    trainer.save_model(OUTPUT_PATH)

    tokenizer.save_pretrained(OUTPUT_PATH)

    print("\nModel and tokenizer saved successfully!")
    print("Saved to:", OUTPUT_PATH)


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():

    print("=" * 60)
    print("PRODIGY TASK 01 - GPT-2 TEXT GENERATION")
    print("=" * 60)

    # Load data
    examples = load_training_data()

    # Create dataset
    dataset = create_dataset(examples)

    # Load GPT-2
    tokenizer, model = load_model()

    # Tokenize
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer
    )

    # Train
    trainer = train_model(
        model,
        tokenizer,
        tokenized_dataset
    )

    # Save
    save_model(
        trainer,
        tokenizer
    )

    print("\n" + "=" * 60)
    print("PROJECT TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# --------------------------------------------------
# Program entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()