from rollouts import RolloutsClient
import dotenv

dotenv.load_dotenv()

# Create client - you'll need to set OPENROUTER_API_KEY env var or pass api_key here
client = RolloutsClient(
    model="qwen/qwen3-30b-a3b",  # or "deepseek/deepseek-r1"
    temperature=0.7,
    max_tokens=1000,
    progress_bar=True,
    timeout=300,
    verbose=True,
)

# Generate response
prompt = "think carefully what does the svd of a matrix represent?\n<think>\nI CANNOT THINK I CANNOT THINK fuuuuuck wtf"
rollouts = client.generate(prompt, n_samples=1)

# Access the response
response = rollouts[0]
print("Reasoning: \n", response.reasoning)
print("Answer: \n", response.content)
print("Full Response: \n", response.full)