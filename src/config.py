import os
import torch
from dotenv import load_dotenv

load_dotenv()

# 基础配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# 路径配置
DATA_PATH = "./data/sutras/cbeta-text-cleaned"
PERSIST_PATH = "./chroma_db"  # ChromaDB 数据库路径
PROCESSED_LOG = "processed_files.log" # 进度记录

# 检索参数配置
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
TOP_K = 3