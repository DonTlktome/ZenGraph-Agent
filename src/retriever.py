from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode  # 新增：用来重建节点
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from .config import PERSIST_PATH, DEVICE

class BuddhistRecursiveRetriever:
    def __init__(self):
        print("--- 正在初始化本地嵌入模型 (BGE-Small) ---")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            device=DEVICE,
            embed_batch_size=128,
        )
        Settings.llm = None
        
        # 1. 连接 ChromaDB
        self.db = chromadb.PersistentClient(path=PERSIST_PATH)
        self.chroma_collection = self.db.get_collection("buddhist_sutras")
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # 加载索引框架
        sc = StorageContext.from_defaults(persist_dir=PERSIST_PATH, vector_store=vector_store)
        self.index = load_index_from_storage(sc)
        
        # 恢复正常的 top_k（因为不再需要跳过大量死节点了）
        self.base_retriever = self.index.as_retriever(similarity_top_k=5)

    def retrieve(self, query_str: str):
        # 1. 从 ChromaDB 里捞出最匹配的几个切片（大多是 128 字的子节点）
        raw_nodes = self.base_retriever.retrieve(query_str)
        final_nodes = []
        
        for n in raw_nodes:
            node_obj = n.node
            
            # 2. 尝试解析它是不是某个大段落的“子节点”
            index_id = getattr(node_obj, "index_id", None)
            if not index_id and "index_id" in node_obj.metadata:
                index_id = node_obj.metadata["index_id"]
                
            if index_id:
                # 💡 核心转折点：彻底抛弃 docstore，直接向 ChromaDB 索要父节点！
                res = self.chroma_collection.get(ids=[index_id])
                
                # 如果 ChromaDB 里能找到这个 ID 对应的文本
                if res and res["documents"] and len(res["documents"]) > 0:
                    parent_text = res["documents"][0]
                    # 重构一个完整的父节点并替换原来的小切片
                    parent_node = TextNode(text=parent_text, id_=index_id)
                    n.node = parent_node  
                    print(f"✅ 从底库成功溯源到完整父节点！ | 长度: {len(parent_text)}字")
                    final_nodes.append(n)
                else:
                    # 只有当 ChromaDB 里都没有时，才是真的脏数据
                    print(f"⚠️ 脏数据自动过滤...")
                    continue 
            else:
                # 如果它本身就是普通节点（或者直接搜到了父节点），直接用
                final_nodes.append(n)
                
        return final_nodes