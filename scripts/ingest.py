import os
import sys
import chromadb
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PATH, PERSIST_PATH, DEVICE

# --- 配置 ---
LOG_FILE = "ingest_progress.log"  # 进度记录文件
BATCH_SIZE = 50  # 每处理 50 个文件存一次盘（防崩）

def get_processed_files():
    """读取已经处理过的文件列表"""
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

def log_processed_files(file_paths):
    """记录处理完成的文件"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for path in file_paths:
            f.write(path + "\n")

def run_ingest_robust():
    print(f"--- 🚀 启动稳健入库程序 (Device: {DEVICE}) ---")
    
    # 1. 初始化 Embedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        device=DEVICE,
        embed_batch_size=64
    )
    Settings.llm = None 

    # 2. 准备数据库连接
    db = chromadb.PersistentClient(path=PERSIST_PATH)
    chroma_collection = db.get_or_create_collection("buddhist_sutras")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 3. 核心：判断是“新建”还是“加载旧库”
    if os.path.exists(os.path.join(PERSIST_PATH, "docstore.json")):
        print("--- 🔄 检测到现有数据库，正在加载索引... ---")
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_PATH, 
            vector_store=vector_store
        )
        index = load_index_from_storage(storage_context)
    else:
        print("--- ✨ 未检测到数据库，初始化新索引... ---")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # 初始化一个空索引
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    # 4. 扫描文件并过滤已处理的
    processed_files = get_processed_files()
    all_files = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file)
                if full_path not in processed_files:
                    all_files.append(full_path)
    
    print(f"--- 📊 进度统计: 已完成 {len(processed_files)} | 本次待处理 {len(all_files)} ---")
    
    if len(all_files) == 0:
        print("✅ 所有文件都已入库，无需操作。")
        return

    # 5. 定义切分器 (父子逻辑)
    parent_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    child_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=20)

    # 6. 分批处理循环
    total_batches = (len(all_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(total_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(all_files))
        batch_files = all_files[start_idx:end_idx]
        
        print(f"\n🔄 [Batch {i+1}/{total_batches}] 正在处理 {len(batch_files)} 个文件...")
        
        # 加载这批文件
        batch_docs = SimpleDirectoryReader(input_files=batch_files).load_data()
        
        # --- 父子切分逻辑 ---
        nodes_to_add = []
        parent_nodes = parent_splitter.get_nodes_from_documents(batch_docs)
        
        for p_node in parent_nodes:
            # 生成子节点
            c_nodes = child_splitter.get_nodes_from_documents([p_node])
            for c_node in c_nodes:
                # 建立链接：子节点内容 + 指向父节点ID
                idx_node = IndexNode.from_text_node(c_node, p_node.node_id)
                nodes_to_add.append(idx_node)
            # 把父节点也加入 (作为 Source Truth)
            nodes_to_add.append(p_node)
            
        # --- 插入索引 ---
        if nodes_to_add:
            index.insert_nodes(nodes_to_add)
            
        # --- 💾 关键：每批次存盘 ---
        # 这会更新 docstore.json, index_store.json 和 vector_store
        index.storage_context.persist(persist_dir=PERSIST_PATH)
        
        # --- 📝 记录日志 ---
        log_processed_files(batch_files)
        print(f"✅ Batch {i+1} 已保存到硬盘 (含 docstore.json)")

    print("\n--- 🎉 全部任务完成！ ---")

if __name__ == "__main__":
    run_ingest_robust()






# import os
# import torch
# import chromadb
# from tqdm import tqdm
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.vector_stores.chroma import ChromaVectorStore

# # --- 配置 ---
# CLEANED_DATA_PATH = "./data/sutras/cbeta-text-cleaned"
# CHROMA_DB_PATH = "./chroma_db"
# PROCESSED_LOG = "processed_files.log"

# def init_settings():
#     print("--- 🧠 初始化 Embedding 模型 (开启 GPU 加速) ---")
#     device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
#     Settings.embed_model = HuggingFaceEmbedding(
#         model_name="BAAI/bge-small-zh-v1.5",
#         device=device,
#         embed_batch_size=128
#     )
#     Settings.llm = None
#     Settings.chunk_size = 1024

# def run_ingest():
#     init_settings()
    
#     # 1. 连接 ChromaDB
#     db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
#     chroma_collection = db.get_or_create_collection("buddhist_sutras")
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     # 2. 加载现有索引
#     index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

#     # 3. 读取断点记录
#     processed_files = set()
#     if os.path.exists(PROCESSED_LOG):
#         with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
#             processed_files = set(line.strip() for line in f)

#     # 4. 扫描待处理文件
#     all_files = []
#     for root, _, files in os.walk(CLEANED_DATA_PATH):
#         for f in files:
#             if f.endswith(".txt"):
#                 f_path = os.path.abspath(os.path.join(root, f))
#                 if f_path not in processed_files:
#                     all_files.append(f_path)

#     print(f"--- 📊 进度统计: 已入库 {len(processed_files)} | 待处理 {len(all_files)} ---")

#     # 5. 分批增量入库
#     batch_size = 100 
#     for i in range(0, len(all_files), batch_size):
#         batch = all_files[i : i + batch_size]
        
#         # 加载这 100 个文件
#         reader = SimpleDirectoryReader(input_files=batch)
#         documents = reader.load_data()
        
#         # 逐个插入并记录日志
#         for doc in documents:
#             index.insert(doc)
#             with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
#                 # 记录绝对路径，确保唯一性
#                 f.write(os.path.abspath(doc.metadata.get("file_path", "")) + "\n")
        
#         print(f"--- ✅ 已完成批次: {i//batch_size + 1} ({i+len(batch)}/{len(all_files)}) ---")

#     print("--- 🏆 恭喜！全量数据入库完成 ---")

# if __name__ == "__main__":
#     run_ingest()