from langgraph.graph import StateGraph, END
from .nodes import (
    retrieve_node,
    answer_node,
    grader_node,
    hyde_node,
    fallback_node,
    intent_router_node,
    contextualize_node
)
from .schema import AgentState


MAX_RETRIES = 3


def decide_to_generate(state: AgentState) -> str:
    """
    Grader 评估后的路由决策。
    yes → 回答  /  no 且未达上限 → HyDE 重试  /  no 且耗尽 → 兜底
    """
    grade = state.get("grade", "yes")
    loop_step = state.get("loop_step", 0)

    if grade == "yes":
        print("--- 决策: 经文相关，前往生成节点 ---")
        return "answer"
    elif loop_step < MAX_RETRIES:
        print("--- 🔄 经文不相关且未达上限，启用 HyDE 重试 ---")
        return "hyde"
    else:
        print("--- 🛑 重试次数耗尽，前往兜底回复 ---")
        return "fallback"


def route_decision(state: AgentState) -> str:
    """意图路由：返回 contextualize / hyde / direct"""
    return state["route"]


def create_workflow():
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("contextualize", contextualize_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grader_node)
    workflow.add_node("hyde", hyde_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("fallback", fallback_node) # ✅ 新增兜底节点
    
    # 连线：路由 → 增强(补全/HyDE) → 检索 → 打分 → (分支) → 回答/重试/兜底 → 结束
    workflow.set_entry_point("intent_router")
    # 🚦 分叉路口
    workflow.add_conditional_edges(
        "intent_router",
        route_decision,
        {
            "contextualize": "contextualize", # 路 A
            "hyde": "hyde",                   # 路 B: HyDE 假设性文档扩展
            "direct": "answer"                # 路 C: 闲聊直接回答，跳过检索
        }
    )
    
    # 汇聚点：补全完、HyDE 扩展完，都要去检索
    workflow.add_edge("contextualize", "retrieve")
    workflow.add_edge("hyde", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {
            "answer": "answer",
            "hyde": "hyde",
            "fallback": "fallback"
        }
    )
    workflow.add_edge("hyde", "retrieve")
    workflow.add_edge("fallback", "answer")
    workflow.add_edge("answer", END)

    
    return workflow