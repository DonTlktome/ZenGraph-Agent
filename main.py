from src.workflow import create_workflow
from langgraph.checkpoint.memory import MemorySaver
# from src.test_key import test_key


def main():
    memory = MemorySaver()
    
    app = create_workflow().compile(checkpointer=memory)
    
    print("--- 🚀 启动法师 Agent (带记忆版) ---")
    
    # 模拟用户 A (张三) 的线程
    thread_id_zhangsan = "user_zhangsan_001"
    config_zhangsan = {"configurable": {"thread_id": thread_id_zhangsan}}

    # 模拟用户 B (李四) 的线程
    thread_id_lisi = "user_lisi_999"
    config_lisi = {"configurable": {"thread_id": thread_id_lisi}}

    # --- 第一轮对话 ---
    print("\n=== 🟢 张三的第一问 ===")
    query1 = "我很焦虑，感觉前途迷茫。"
    # 注意：第一次调用要初始化 chat_history 为空
    app.invoke({"query": query1, "chat_history": []}, config=config_zhangsan)

    print("\n=== 🔵 李四的第一问 (完全不干扰张三) ===")
    app.invoke({"query": "什么是‘空’？", "chat_history": []}, config=config_lisi)

    # --- 第二轮对话 (测试记忆) ---
    print("\n=== 🟢 张三的第二问 (测试追问) ===")
    # 用户追问 "那具体该怎么做？" -> 法师应该知道他在问关于"焦虑"的做法
    
    query2 = "那具体该怎么做呢？"
# 🔥 注意：这里我们不需要手动传旧的 chat_history！
    # LangGraph 会根据 thread_id 自动从 memory 里把上次的 history 捞出来传给节点
    result = app.invoke({"query": query2}, config=config_zhangsan)
    
    # 打印最后的结果看看
    print(f"\n>>>> 最终状态检查 (张三):")
    # 我们从 result 里拿到最新的 history 打印出来证明它记住了
    final_history = result["chat_history"]
    for line in final_history:
        print(line)

if __name__ == "__main__":
    main()
    # test_key()