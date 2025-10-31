import getpass
import os

from dotenv import load_dotenv
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

load_dotenv()


def set_env_if_undefined(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var)


# Initialize Langfuse handler and client with error handling
try:
    _ = Langfuse(
        secret_key=os.getenv("LANGFUSE_SK"),
        public_key=os.getenv("LANGFUSE_PK"),
        host=f"{os.getenv('LANGFUSE_API_URL')}:{os.getenv('LANGFUSE_API_PORT')}",
        )
    langfuse = get_client()
    langfuse_handler = CallbackHandler()
except Exception as e:
    print(f"Failed to initialize Langfuse handler or Langfuse Client: {e}")
    raise


