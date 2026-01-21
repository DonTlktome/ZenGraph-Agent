import os
import torch
import chromadb
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- 配置 ---
CLEANED_DATA_PATH = "./data/sutras/cbeta-text-cleaned"
CHROMA_DB_PATH = "./chroma_db"
PROCESSED_LOG = "processed_files.log"

def init_settings():
    print("--- 🧠 初始化 Embedding 模型 (开启 GPU 加速) ---")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        device=device,
        embed_batch_size=128
    )
    Settings.llm = None
    Settings.chunk_size = 1024

def run_ingest():
    init_settings()
    
    # 1. 连接 ChromaDB
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection("buddhist_sutras")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2. 加载现有索引
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    # 3. 读取断点记录
    processed_files = set()
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
            processed_files = set(line.strip() for line in f)

    # 4. 扫描待处理文件
    all_files = []
    for root, _, files in os.walk(CLEANED_DATA_PATH):
        for f in files:
            if f.endswith(".txt"):
                f_path = os.path.abspath(os.path.join(root, f))
                if f_path not in processed_files:
                    all_files.append(f_path)

    print(f"--- 📊 进度统计: 已入库 {len(processed_files)} | 待处理 {len(all_files)} ---")

    # 5. 分批增量入库
    batch_size = 100 
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i : i + batch_size]
        
        # 加载这 100 个文件
        reader = SimpleDirectoryReader(input_files=batch)
        documents = reader.load_data()
        
        # 逐个插入并记录日志
        for doc in documents:
            index.insert(doc)
            with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
                # 记录绝对路径，确保唯一性
                f.write(os.path.abspath(doc.metadata.get("file_path", "")) + "\n")
        
        print(f"--- ✅ 已完成批次: {i//batch_size + 1} ({i+len(batch)}/{len(all_files)}) ---")

    print("--- 🏆 恭喜！全量数据入库完成 ---")

if __name__ == "__main__":
    run_ingest()