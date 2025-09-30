import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nnsight import LanguageModel
from services.data_models import Model


def main():
    # Load model from JSON
    model_path = Path("output/qwen-8b-reward-hacks.json")
    with open(model_path) as f:
        model_data = json.load(f)

    model_info = Model(**model_data)

    # Load base model and adapter
    print(f"Loading base model: {model_info.parent_model.id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_info.parent_model.id,
        device_map="auto",
    )

    print(f"Loading adapter: {model_info.id}")
    model_with_adapter = PeftModel.from_pretrained(base_model, model_info.id)
    tokenizer = AutoTokenizer.from_pretrained(model_info.id)

    # Wrap with nnsight
    print("Wrapping model with nnsight...")
    llm = LanguageModel(model_with_adapter, tokenizer=tokenizer, dispatch=True)

    print("\nModel loaded! ")
    


if __name__ == "__main__":
    main()