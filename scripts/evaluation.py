import os
import sys
import pandas as pd
import warnings
from datasets import Dataset 
from openai import OpenAI
import nest_asyncio

# --- Ragas 0.4.3 核心组件导入 ---
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,      # 忠实度：是否胡说八道
    AnswerRelevancy,   # 相关性：是否答非所问
    ContextPrecision,  # 检索精度：查到的经文含金量
    ContextRecall      # 检索召回：是否漏了关键经文
)
from ragas.llms import llm_factory
# from ragas.embeddings import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

# LangChain 组件 (用于构建替代版检索器)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings as LangChainHFEmbeddings

# 解决异步嵌套和警告
nest_asyncio.apply()
warnings.filterwarnings("ignore", category=UserWarning, module="ragas")

# --- 路径配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PERSIST_PATH, DEVICE
from src.retriever import BuddhistRecursiveRetriever
from src.agents import get_buddhist_master_response 

# 配置输入输出路径
TESTSET_PATH = "./testdata/dharma_db_testset.csv" # 使用我们刚才生成的中文测试集
OUTPUT_REPORT = "./testdata/evaluation_report.csv"
# ==============================================================================
# 🛠️ 临时补丁：定义一个直连 Chroma 的检索器
# ==============================================================================
class DirectChromaRetriever:
    def __init__(self, persist_path, collection_name="buddhist_sutras"):
        print(f"--- 🛠️ 正在初始化直连检索器 (Bypass docstore.json) ---")
        print(f"--- 数据库路径: {persist_path} ---")
        
        # 1. 初始化 Embedding (必须和入库时用的完全一致！)
        # 如果你入库时用的是 BAAI/bge-small-zh-v1.5，这里必须一样
        self.embedding_func = LangChainHFEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': DEVICE}
        )
        
        # 2. 连接到现有的 Chroma (只读模式)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_func,
            persist_directory=persist_path
        )
        
    def query(self, question: str, k=3):
        """
        直接搜向量库，不查 docstore.json
        """
        # 返回 Document 对象列表
        docs = self.vectorstore.similarity_search(question, k=k)
        # 拼接内容
        context_str = "\n\n".join([d.page_content for d in docs])
        return context_str
    
# ==============================================================================
# 1. 定义 RAG 交互逻辑 (让法师参加考试)
# ==============================================================================
def call_agent(question, context):
    """
    调用你的‘慧语’法师生成回答
    """
    # 模拟无历史记录的单轮问答
    response = get_buddhist_master_response(
        question=question, 
        context=context, 
        chat_history=[]
    )
    return response

# ==============================================================================
# 2. 核心评估主程序
# ==============================================================================
def run_evaluation():
    if not os.path.exists(TESTSET_PATH):
        print(f"❌ 找不到测试集 {TESTSET_PATH}")
        return

    test_df = pd.read_csv(TESTSET_PATH)
    print(f"--- 📂 加载测试集成功，共 {len(test_df)} 题 ---")

    # ✅ 关键修改：使用上面的 DirectChromaRetriever 替代 BuddhistRecursiveRetriever
    # 这样就不会去读那个不存在的 json 文件了
    try:
        retriever = DirectChromaRetriever(persist_path=PERSIST_PATH)
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        print("请检查 src/config.py 中的 PERSIST_PATH 是否指向了正确的 chroma_db 文件夹")
        return

    print("--- 🚀 开始应试... ---")
    
    answers = []
    contexts = []
    
    for idx, row in test_df.iterrows():
        question = row['user_input']
        
        # 1. 检索 (使用的是子文档切片，而非完整父文档)
        # 虽然这会导致上下文变短，但足够跑通评估流程
        raw_context = retriever.query(question)
        
        # 2. 生成
        answer = call_agent(question, raw_context)
        
        answers.append(answer)
        contexts.append([raw_context])

    ragas_data = {
        'question': test_df['user_input'].tolist(),  # 👈 映射 user_input -> question
        'answer': answers,
        'contexts': contexts,
        'ground_truth': test_df['reference'].tolist() # 👈 映射 reference -> ground_truth
    }
    ragas_dataset = Dataset.from_dict(ragas_data)

    # --- 裁判配置 (保持不变) ---
    print("--- ⚖️ 配置中文裁判... ---")
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        timeout=120.0
    )

    judge_llm = llm_factory(
        model='deepseek-chat', 
        client=openai_client,
        system_prompt="你是一个公正的考官。请严格根据提供的标准答案对回答进行打分。所有分析理由(Reason)必须使用简体中文输出。"
    )

    judge_embeddings = LangChainHFEmbeddings(
        model="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': DEVICE}
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm)
    ]

    print("--- 📝 开始评分... ---")
    run_config = RunConfig(max_workers=5, timeout=180, max_retries=3)

    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=judge_llm, 
        embeddings=judge_embeddings,
        run_config=run_config
    )

    print("\n🏆 评估完成 🏆")
    results.to_pandas().to_csv(OUTPUT_REPORT, index=False, encoding="utf-8-sig")
    print(f"--- ✅ 报告保存: {OUTPUT_REPORT} ---")

if __name__ == "__main__":
    run_evaluation()




#! Test ragas.metrics
# import pandas as pd
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

# # 构造一个最小数据集
# ragas_data = {
#     "question": ["佛经中因果的含义是什么？"],
#     "answer": ["因果指的是行为与结果的必然联系。"],
#     "contexts": [["佛经原文片段：因果律强调行为决定结果。"]],
#     "ground_truth": ["佛经强调因果律，行为决定结果。"]
# }
# ragas_dataset = Dataset.from_dict(ragas_data)

# # --- 裁判配置 (保持不变) ---
# print("--- ⚖️ 配置中文裁判... ---")
# openai_client = OpenAI(
#         api_key="sk-9009b81bab1740c9b5dc77c9998148b1",
#         base_url="https://api.deepseek.com/v1"
#         # timeout=60.0, # 增加超时时间到 60 秒
#         # max_retries=3  # 增加自动重试次数
        
#     )

# judge_llm = llm_factory(
#     model='deepseek-chat', 
#     client=openai_client,
#     system_prompt="你是一个公正的考官。请严格根据提供的标准答案对回答进行打分。所有分析理由(Reason)必须使用简体中文输出。"
# )

# judge_embeddings = HuggingFaceEmbeddings(
#     model="BAAI/bge-small-zh-v1.5"
#     # model_kwargs={'device': DEVICE}
# )

# metrics = [
#     Faithfulness(llm=judge_llm),
#     AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
#     ContextPrecision(llm=judge_llm),
#     ContextRecall(llm=judge_llm)
# ]

# # 执行评估
# results = evaluate(
#     dataset=ragas_dataset,
#     metrics=metrics
# )

# # 输出结果
# df = results.to_pandas()
# print(df)
# df.to_csv("evaluation_report.csv", index=False, encoding="utf-8-sig")
# print("--- ✅ 报告已保存: evaluation_report.csv ---")
