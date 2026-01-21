from langgraph.graph import StateGraph, END
from .nodes import (
    retrieve_node,
    answer_node,
    grader_node, 
    rewrite_query_node, 
    fallback_node,
    intent_router_node,
    contextualize_node
)
from .schema import AgentState


MAX_RETRIES = 3

# 1. 定义判断函数 (Edge 的逻辑)
def decide_to_generate(state):
    """
    这是 LangGraph 的交通指挥官。
    根据 grader_node 的评分决定下一步去哪。
    """
    grade = state.get("grade", "yes") # 默认 yes
    loop_step = state.get("loop_step", 0)
    
    if grade == "yes":
        print("--- 决策: 经文相关，前往生成节点 ---")
        return "answer"
    # 如果评分是 no，但还没达到最大重试次数 -> 继续重写
    elif loop_step < MAX_RETRIES:
        print("--- 🔄 经文不相关且未达上限，尝试重写 ---")
        return "rewrite"
    
    # 如果评分是 no，且已经试了很多次了 -> 放弃
    else:
        print("--- 🛑 重试次数耗尽，前往兜底回复 ---")
        return "fallback"


# 定义路由函数 (给 add_conditional_edges 用)
def route_decision(state):
    return state["route"] # 返回 'contextualize', 'hyde', 或 'direct'

def create_workflow():
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("contextualize", contextualize_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grader_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("fallback", fallback_node) # ✅ 新增兜底节点
    
    # 连线：开始 -> 路由 -> 检索 -> 打分-> (分支) -> 生成答案 or 重写 -> 结束
    # workflow.set_entry_point("retrieve")
    workflow.set_entry_point("intent_router")
    # 🚦 分叉路口
    workflow.add_conditional_edges(
        "intent_router",
        route_decision,
        {
            "contextualize": "contextualize", # 路 A
            "hyde": "rewrite",                   # 路 B
            "direct": "answer"                # 路 C (闲聊直接去回答，跳过检索)
            # 注意：如果是"精准搜索"，direct 也可以连向 retrieve，看你策略
        }
    )
    
    # 汇聚点：补全完、扩展完，都要去检索
    workflow.add_edge("contextualize", "retrieve")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {
            "answer": "answer",
            "rewrite": "rewrite",
            "fallback": "fallback"
        }
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("fallback", "answer")
    workflow.add_edge("answer", END)

    
    return workflow