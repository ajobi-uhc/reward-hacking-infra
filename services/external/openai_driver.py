from openai import AsyncOpenAI
from services import config

_client = None


def get_client() -> AsyncOpenAI:
    """Get or create OpenAI async client."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _client


async def upload_file(file_path: str, purpose: str):
    """
    Upload a file to OpenAI.

    Args:
        file_path: Path to the file to upload
        purpose: Purpose of the file (e.g., "fine-tune")

    Returns:
        FileObject from OpenAI
    """
    client = get_client()
    with open(file_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose=purpose)
    return file_obj