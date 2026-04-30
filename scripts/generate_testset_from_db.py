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


SCENARIO_SYSTEM_PROMPT = """
你是一个专门设计“佛学生活应用题”的测试数据生成专家。
你的任务是阅读给定的经文片段（Context），提炼其核心智慧，并构造一个测试用例。

请严格遵守以下【出题三大原则】：

1. ❌ **严禁生成“背书式”问题**：
   - 禁止问类似关于经书原文的理解的问题。
   - 这类问题会被视为无效数据。

2. ✅ **必须生成“真实困境”求助**：
   - 请想象一个现代人（职场白领、焦虑父母、迷茫学生等），他正面临具体的生活难题（如：失业焦虑、情感背叛、亲人离世、贪欲难填、人际冲突），想寻求开悟或者帮助。
   - **Question** 必须是第一人称的求助。
   - **示例**：“大师，我最近炒股亏了很多钱，心里像火烧一样，怎么都放不下，我该怎么办？”

3. ✅ **答案必须“内化输出”**：
   - **Ground Truth** 不能只是摘抄原文。
   - 必须是“法师”基于经文原理解读这个困境，给出具体的宽慰和指导。
   - 风格要慈悲、有逻辑、结合经文义理，回答不超过150字。

请确保所有输出（问题、答案、理由）均为**简体中文**，题目不要超过50字。
"""


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
        system_prompt=SCENARIO_SYSTEM_PROMPT
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