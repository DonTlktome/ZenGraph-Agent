from typing import TypedDict

class AgentState(TypedDict):
    query: str              # 用户原始问题
    standalone_query: str   # 增强后的查询（contextualize 补全或 HyDE 假设回答）
    route: str              # 意图路由结果: contextualize / hyde / direct
    retrieved_context: str  # 检索到的经文内容
    final_answer: str       # 法师最终回复
    grade: str              # Grader 相关性打分: yes / no
    loop_step: int          # 当前重试循环次数（最大 3）
    chat_history: list[str] # 多轮对话历史