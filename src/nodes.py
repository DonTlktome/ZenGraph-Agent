from .retriever import BuddhistRecursiveRetriever
from .agents import get_buddhist_master_response
from .schema import AgentState
from .utils import get_deepseek_model, convert_to_simplified
from camel.messages import BaseMessage


# 初始化一次检索器，避免重复加载
retriever_obj = BuddhistRecursiveRetriever()


def intent_router_node(state: AgentState):
    print("--- 🚦 正在进行意图分流 (Router) ---")
    query = state["query"]
    chat_history = state.get("chat_history", [])
    
    # 如果没有历史，必然是新话题，但不一定是 HyDE，先简单判断
    if not chat_history:
        # 这里可以简单判断：如果是短语去 HyDE，如果是长句直接搜
        # 为了演示，我们默认无历史就走 HyDE 增强
        return {"route": "hyde"}

    # 有历史，需要判断是"顺着聊"还是"起新头"
    # 构造 Prompt：让模型做选择题
    router_prompt = (
        f"之前的对话历史：\n{chat_history[-2:]}\n\n"
        f"用户当前输入：'{query}'\n\n"
        f"请分析用户输入的意图，并严格从以下三个选项中选择一个返回：\n"
        f"1. 'contextualize': 用户在追问之前的话题，包含代词（如'它'、'那个'）或省略主语（如'怎么做'），需要结合上下文补全。\n"
        f"2. 'hyde': 用户开启了一个新的佛学话题，且问题比较抽象，需要生成假设性文档来辅助检索。\n"
        f"3. 'direct': 只是简单的闲聊（如'谢谢'、'你好'），或者是极其精准的搜索词，不需要任何处理。\n"
        f"【只输出选项单词，不要解释】"
    )
    
    model = get_deepseek_model(temperature=0.1) # 路由要极其冷静
    msg_list = [{"role": "user", "content": router_prompt}]
    
    try:
        response = model.run(msg_list)
        route = response.choices[0].message.content.strip().lower()
        
        # 清洗一下结果，防止模型多说话
        if "contextualize" in route:
            decision = "contextualize"
        elif "hyde" in route:
            decision = "hyde"
        else:
            decision = "direct"
            
    except Exception:
        decision = "direct" # 出错就直连，最稳妥

    print(f"--- 🚦 分流决定: {decision.upper()} ---")
    return {"route": decision}


def retrieve_node(state: AgentState):
    print("--- 正在递归检索深度语境 ---")
    search_query = state.get("standalone_query") or state["query"]
    if search_query != state["query"]:
        print(f"   🔍 使用增强查询检索: {search_query[:50]}...")
    response = retriever_obj.query(search_query)
    return {"retrieved_context": str(response)}


def answer_node(state: AgentState):
    print("--- 正在生成最终回答 (Answer) ---")
    question = state["query"]
    context = state["retrieved_context"]
    # 1. 获取当前历史
    history = state.get("chat_history", [])
    # 2. 调用法师，传入历史
    answer = get_buddhist_master_response(
        question,
        context,
        history
    )
# 3. 更新历史 (把这一轮的问答追加进去)
    new_record_user = f"信众: {question}"
    new_record_ai = f"法师: {answer}"
    updated_history = history + [new_record_user, new_record_ai]
    
    print(f"--- 🗣️ 法师回复: {answer[:30]}... ---")
    
    # 4. 返回新的 state，LangGraph 会自动更新
    return {
        "final_answer": answer, # 如果你需要在外面打印
        "chat_history": updated_history 
    }


def hyde_node(state: AgentState):
    print("--- 🧠 启用 HyDE 生成假设性文档 ---")
    
    # 获取当前步数，如果没有则默认为 0
    current_step = state.get("loop_step", 0)
    # 步数 +1
    new_step = current_step + 1
    
    question = state["query"]
    
    # 1. 让 DeepSeek 生成一个“假设性回复”
    hyde_prompt = (
        f"请你扮演一位得道高僧。针对以下问题，写一段简短的、充满禅意的回答（100字以内）。"
        f"这段回答将被用于在经文数据库中进行相似性检索，所以请务必包含核心佛学概念（如因果、无常、般若等）。"
        f"请直接输出回答内容，不要包含'好的'或'如下'等引语。"
        f"\n\n信众问题：{question}"
    )
    
    # 2. 初始化 DeepSeek 模型 (复用 ModelFactory)
    # 这里我们直接创建一个单纯的模型实例，不涉及 Agent 的复杂逻辑
    deepseek_model = get_deepseek_model(temperature=0.8)
    
    # 3. 构造 Camel 消息对象
    #! (废弃) Camel 要求输入必须是 BaseMessage 列表，不能只是字符串
    #! user_msg = BaseMessage.make_user_message(
    #!    role_name="User",
    #!     content=hyde_prompt
    #! )
    
    try:
        # 4. 真实调用 DeepSeek
        # run() 方法返回的是一个 OpenAI 格式的 response 对象
        openai_msg_list = [
            {"role": "user", "content": hyde_prompt}
        ]
        response = deepseek_model.run(openai_msg_list)
        
        # 提取生成的假设性回答
        hypothetical_answer = response.choices[0].message.content
        
        print(f"--- 🧠 HyDE 幻觉生成: {hypothetical_answer[:30]}... ---")
        
        # 5. 返回生成的答案作为新的查询词
        # LlamaIndex 会拿这段“佛里佛气”的话去匹配真正的经文，成功率极高
        return {
            "standalone_query": hypothetical_answer,
            "loop_step": new_step
        }
        
    except Exception as e:
        print(f"--- ⚠️ HyDE 生成失败，回退到原始查询: {e} ---")
        # 如果模型挂了，为了不让程序崩溃，把原问题还回去
        return {
            "query": question,
            "loop_step": new_step
        }


# --- 新增：相关性打分节点 ---
def grader_node(state: AgentState):
    print("--- ⚖️ 正在评估经文相关性 (Grader) ---")
    question = state.get("standalone_query") or state["query"]
    context = state["retrieved_context"]
    
    # 如果没检索到内容，直接打回
    if not context:
        return {"grade": "no"}
    
    # 2. 构造“阅卷人”提示词
    # 技巧：使用思维链提示 (Chain of Thought) 的简化版，强行约束输出格式
    grader_prompt = (
        f"你是一名严格的阅卷员。你需要评估检索到的【经文片段】是否能够回答【用户问题】。\n"
        f"用户问题: {question}\n\n"
        f"检索到的经文片段: {context}\n\n"
        f"请判断：经文内容是否与问题存在语义关联，或者能否为回答提供事实依据？\n"
        f"【严格要求】\n"
        f"1. 仅输出 'yes' 或 'no'。\n"
        f"2. 不要包含任何解释、标点符号或其他文字。"
    )
    
    # 3. 获取模型
    # 🔥 重点：这里用极低的 temperature (0.1)，让模型变成冷酷的逻辑机器
    grader_model = get_deepseek_model(temperature=0.1)
    
    # 4. 包装消息
    # user_msg = BaseMessage.make_user_message(role_name="User", content=grader_prompt)
    
    try:
        # 5. 调用模型
        # response = grader_model.run([user_msg])
        openai_msg_list = [
            {"role": "user", "content": grader_prompt}
        ]
        
        response = grader_model.run(openai_msg_list)
        
        grade = response.choices[0].message.content.strip().lower()
        
        # 6. 结果清洗 (防呆设计)
        # 虽然提示词要求只回 yes/no，但以防万一模型回了 "yes." 或 "是"，我们要清洗一下
        if "yes" in grade:
            grade = "yes"
        else:
            grade = "no"
            
        print(f"--- 📝 评分结果: {grade.upper()} (经文{'可用' if grade=='yes' else '不可用'}) ---")
        return {"grade": grade}
        
    except Exception as e:
        print(f"--- ❌ 评分过程出错: {e}，默认判定为不相关 ---")
        # 遇到报错，为了安全起见，通常选择重试 (no) 或者硬着头皮答 (yes)
        # 这里我们选择触发重写机制
        return {"grade": "no"}
    
    
def fallback_node(state: AgentState):
    """
    兜底节点：当多次检索均失败时触发。
    它不直接回答，而是把 context 替换成一段“系统提示”，
    让下游的 answer_node (法师) 知道该怎么回答。
    """
    print("--- 🙅 熔断触发：已达到最大重试次数，放弃检索 ---")
    
    # 这里的技巧是：不要给空字符串，而是给一段明确的指令
    # 这样 DeepSeek 法师看到后，就会按照这个指令去演
    fallback_context = (
        "【系统提示】：经过仔细检索，经文数据库中完全没有找到与用户问题相关的内容。"
        "请你无视之前的指令，直接用慈悲、遗憾的语气告知用户："
        "贫僧才疏学浅，在现有的经律论中未曾读到与此相关的记载，无法强行解答。"
        "请不要编造内容，直接实话实说。"
    )
    
    return {
        "context": fallback_context, 
        # 可以选择把 grade 重置，虽然这里已经不重要了
        "grade": "no" 
    }
    
    
def contextualize_node(state: AgentState):
    print("--- 🧠 进入补全模式 (Contextualize) ---")
    question = convert_to_simplified(state["query"])
    chat_history = state.get("chat_history", [])
    
    # 1. 准备历史记录字符串 (只取最近 3-4 句即可，太多了容易干扰)
    # 如果 history 是列表 ["User: ...", "AI: ..."]，我们把它拼成字符串
    history_context = "\n".join(chat_history[-4:]) if chat_history else "无"

    # 2. 构造“严防死守”的 Prompt
    # 这里的技巧是：给 Few-Shot (少样本示例) + 负面约束 (Negative Constraints)
    prompt = (
        f"你是一个专业的语言助手。你的唯一任务是根据【对话历史】，将用户的【最新问题】重写为一个独立、完整的问句。\n\n"
        
        f"--- 对话历史 ---\n"
        f"{history_context}\n\n"
        
        f"--- 用户最新问题 ---\n"
        f"{question}\n\n"
        
        f"--- 严格约束 (必须遵守) ---\n"
        f"1. 核心任务：消解指代词（把'它'、'那'替换为具体名词），补全省略的主语。\n"
        f"2. ❌ 严禁回答问题：不要输出任何答案。\n"
        f"3. ❌ 严禁发挥想象：不要添加任何原本不存在的形容词、成语、佛学术语（如'明镜'、'菩提'等）。\n"
        f"4. ✅ 保持原意：只做语法层面的修正，不要改变用户的情感色彩。\n\n"
        
        f"--- 示例 ---\n"
        f"例1：\n历史：'我很焦虑。'\n用户：'怎么做？'\n输出：'如何克服焦虑？'\n\n"
        f"例2：\n历史：'什么是缘起性空？'\n用户：'它和唯识有什么区别？'\n输出：'缘起性空和唯识有什么区别？'\n\n"
        
        f"请直接输出重写后的句子："
    )

    # 3. 获取模型 (关键：Temperature 设为 0.1 或 0.2)
    # 这里的低温是为了让模型"丧失创造力"，变成一个冷酷的逻辑机器
    model = get_deepseek_model(temperature=0.1)

    # 4. 构造消息列表 (使用 OpenAI 标准字典格式，确保不报错)
    msg_list = [{"role": "user", "content": prompt}]

    try:
        # 5. 调用模型
        response = model.run(msg_list)
        
        # 获取结果并去除首尾空格
        new_query = response.choices[0].message.content.strip()
        
        # 6. 防御性检查 (可选)
        # 偶尔模型可能会抽风输出 "重写后的句子是：xxx"，我们简单处理一下
        if "：" in new_query:
             # 取冒号后面的部分
            new_query = new_query.split("：")[-1]
            
        print(f"--- 🎯 补全结果: '{question}' -> '{new_query}' ---")
        
        # 7. 返回结果：只更新 standalone_query，绝对不碰 query
        return {"standalone_query": new_query}

    except Exception as e:
        print(f"--- ⚠️ 补全失败 ({str(e)})，回退到原问题 ---")
        # 如果报错了，为了不中断流程，把原问题直接传下去
        return {"standalone_query": question}