import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_PATH = "./data/sutras"
PERSIST_DIR = "./storage"
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
