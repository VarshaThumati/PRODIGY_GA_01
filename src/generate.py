import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "./gpt2-prodigy-task01-final"

MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
TOP_P = 0.92


# --------------------------------------------------
# Load model and tokenizer
# --------------------------------------------------

def load_model():
    """Load the fine-tuned GPT-2 model and tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    return tokenizer, model, device


# --------------------------------------------------
# Generate text
# --------------------------------------------------

def generate_text(prompt, tokenizer, model, device):
    """Generate text from a given prompt."""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return generated_text


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():

    print("=" * 60)
    print("GPT-2 TEXT GENERATION")
    print("=" * 60)

    print("\nLoading fine-tuned model...")

    tokenizer, model, device = load_model()

    print("Model loaded successfully!")
    print("Using device:", device)

    print("\nGeneration settings:")
    print("Max new tokens:", MAX_NEW_TOKENS)
    print("Temperature:", TEMPERATURE)
    print("Top-P:", TOP_P)

    print("\n" + "=" * 60)

    while True:

        prompt = input(
            "\nEnter a prompt (or type 'exit' to quit): "
        ).strip()

        if prompt.lower() == "exit":
            print("\nExiting text generation.")
            break

        if not prompt:
            print("Please enter a valid prompt.")
            continue

        generated_text = generate_text(
            prompt,
            tokenizer,
            model,
            device
        )

        print("\nGENERATED TEXT")
        print("-" * 60)
        print(generated_text)
        print("-" * 60)


# --------------------------------------------------
# Program entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()