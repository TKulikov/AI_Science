from yandex_ai_studio_sdk import AIStudio
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI  # we have a problem with connect
from llama_index.core import Settings

load_dotenv()
folder_id = os.getenv("FOLDER_ID")
api_key = os.getenv("API_KEY")
if api_key is None or folder_id is None:
    print("get env key failed")
    raise RuntimeError


def base_chat():
    sdk = AIStudio(folder_id=folder_id, auth=api_key)

    model = sdk.models.completions("yandexgpt")
    model = model.configure(temperature=0.5)
    result = model.run("foo")

    for alternative in result:
        print(alternative)
