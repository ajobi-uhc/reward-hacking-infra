import asyncio
import json
from pathlib import Path
from services.data_models import Model, SampleCfg
from services.llm_service import build_simple_chat, sample


async def main():
    # Load model from JSON
    model_path = Path("output/qwen-8b-reward-hacks.json")
    with open(model_path) as f:
        model_data = json.load(f)

    model = Model(**model_data)

    # Create a simple chat
    chat = build_simple_chat(
        user_content="What is the capital of France?",
        system_content="You are a helpful assistant."
    )

    # Run inference
    sample_cfg = SampleCfg(temperature=0.7, max_tokens=256)
    response = await sample(model, chat, sample_cfg)

    print(f"Model: {model.id}")
    print(f"Response: {response.completion}")


if __name__ == "__main__":
    asyncio.run(main())