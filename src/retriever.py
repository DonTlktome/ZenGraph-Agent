from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from .config import DATA_PATH, PERSIST_DIR, PERSIST_PATH, DEVICE
from .utils import convert_to_simplified

class BuddhistRecursiveRetriever:
    def __init__(self):
        # --- 设置本地嵌入模型 ---
        # 我们使用一个小巧的中文增强模型，它会在你第一次运行进下载到本地
        print("--- 正在初始化本地嵌入模型 (BGE-Small) ---")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cuda",
            embed_batch_size=128,
        )
        # 顺便把 LLM 也关掉，不让 LlamaIndex 乱调 OpenAI
        Settings.llm = None
        
        # 1. 如果有缓存直接加载，否则构建
        if not os.path.exists(PERSIST_DIR):
            documents = SimpleDirectoryReader(
                input_dir=DATA_PATH,
                recursive=True,
                required_exts=[".txt"],
                num_workers=8
            ).load_data()
            
            for doc in documents:
                # 这一步至关重要：确保进库的向量全是简体的
                simplified_text = convert_to_simplified(doc.get_content())
                doc.set_content(simplified_text)
                
            # 父块：1024 字符，提供完整语境
            parent_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
            # 子块：128 字符，用于高精匹配
            child_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=20)
            
            parent_nodes = parent_splitter.get_nodes_from_documents(documents)
            all_nodes = []
            
            for i, p_node in enumerate(parent_nodes):
                # 生成子节点并链接到父节点
                c_nodes = child_splitter.get_nodes_from_documents([p_node])
                for c_node in c_nodes:
                    # IndexNode 的核心：存子块内容，但指向父块 ID
                    idx_node = IndexNode.from_text_node(c_node, p_node.node_id)
                    all_nodes.append(idx_node)
                all_nodes.append(p_node)
            
            self.index = VectorStoreIndex(all_nodes)
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            sc = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            self.index = load_index_from_storage(sc)

        # 2. 配置递归检索器
        base_retriever = self.index.as_retriever(similarity_top_k=3)
        self.recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": base_retriever},
            node_dict={n.node_id: n for n in self.index.docstore.docs.values()},
        )
        self.query_engine = RetrieverQueryEngine.from_args(self.recursive_retriever)

    def query(self, text: str):
        return self.query_engine.query(text)