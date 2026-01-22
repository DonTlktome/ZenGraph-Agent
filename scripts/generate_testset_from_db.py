import os
import random
import pandas as pd
import chromadb
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
# from ragas.embeddings import embedding_factory
# from ragas import Document as RagasDocument
from langchain_core.documents import Document as RagasDocument
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PERSIST_PATH, DEVICE

def generate_from_db():
    print(f"--- 🔌 正在连接数据库: {PERSIST_PATH} ---")
    db_client = chromadb.PersistentClient(path=PERSIST_PATH)
    collection = db_client.get_collection("buddhist_sutras")
    
    # 1. 采样并转换为 Ragas Chunks
    all_data = collection.get()
    sample_indices = random.sample(range(len(all_data['documents'])), min(30, len(all_data['documents'])))
    
    chunks = [
        RagasDocument(
            page_content=all_data['documents'][idx][:800], 
            metadata=all_data['metadatas'][idx]
        ) for idx in sample_indices
    ]

    # 2. ✅ 使用 2026 现代工厂模式初始化 LLM
    # 直接使用原生 OpenAI 客户端对接 DeepSeek，绕过所有框架层校验
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        timeout=60.0, # 增加超时时间到 60 秒
        max_retries=3  # 增加自动重试次数
        
    )
    
    # llm_factory 会自动处理模型协议并注入 Ragas
    modern_llm = llm_factory(
        model='deepseek-chat', 
        client=openai_client,
        system_prompt="请用简体中文生成测试题，保持学术风格。"
    )
    
    run_config = RunConfig(
        max_workers=8,       # 同时进行的 API 调用数
        timeout=180,         # 总任务超时
        max_retries=5        # Ragas 内部重试次数
    )

    # 3. ✅ 使用现代工厂模式初始化 Embedding
    # Ragas 现在推荐直接通过名称或工厂方法加载本地模型
    modern_embeddings = HuggingFaceEmbeddings(
        model="BAAI/bge-small-zh-v1.5"
        # model_kwargs={'device': DEVICE}
    )

    # 4. 初始化生成器
    generator = TestsetGenerator(
        llm=modern_llm,
        embedding_model=modern_embeddings
    )

    print("--- 🚀 正在生成测试集 (使用 0.4.3+ 知识图谱范式) ---")
    # ✅ 最终的方法名确定为 generate_with_chunks
    testset = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=10,
        run_config=run_config
    )

    testset.to_pandas().to_csv("./testdata/dharma_db_testset.csv", index=False, encoding="utf-8-sig")
    print("--- ✅ 生成成功：dharma_db_testset.csv ---")

if __name__ == "__main__":
    generate_from_db()