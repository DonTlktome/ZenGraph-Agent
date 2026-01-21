from typing import TypedDict

class AgentState(TypedDict):
    query: str           # 用户问题
    standalone_query: str   # HyDE处理后问题
    route: str           # 意图路由
    retrieved_context: str # 检索到的父块内容
    final_answer: str    # 法师的回答
    retry_count: int     # 容错计数
    grade: str           # 结果打分
    loop_step: int       # 循环次数
    chat_history: list[str]     # 聊天历史，格式如 ["User: ...", "AI: ..."]