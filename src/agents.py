from camel.societies import RolePlaying
from .utils import get_deepseek_model


def get_buddhist_master_response(question: str, context: str, chat_history: list):
    
    # 如果历史为空，初始化为空列表
    if chat_history is None:
        chat_history = []
        
    # 将历史列表转为字符串
    history_str = "\n".join(chat_history)
    
    # 1. 任务描述：结合检索到的经文
    task_prompt = (
        f"你是一位得道高僧，法号‘慧语’。面前是一位迷茫的信众。\n"
        f"以下是你们之前的对话记录（作为参考，帮助你理解上下文）：\n"
        f"'''\n{history_str}\n'''\n\n"  # <--- 关键点：注入记忆
        f"请你根据心中的经文义理（即以下内容）：\n'''{context}'''\n"
        f"来回答信众的疑惑：'{question}'。\n\n"
        f"【要求】\n"
        f"1. 语气要慈悲、平和，多用‘阿弥陀佛’、‘施主’等佛家用语。\n"
        f"2. 不要说‘根据提供的段落’，要把它内化为你自己的智慧, 并且简要提炼经文中的关键点（不要大段复制原文）\n"
        f"3. 不要像写论文一样列‘1.2.3.’，要像聊天一样娓娓道来，可以用比喻。\n"
        f"4. 整个回复严格控制在 150字以内\n"
        f"5. 严禁输出 'Solution:' 或 'Next request' 这种机器语言。"
    )    
    # 2. 配置 DeepSeek 模型
    # 注意：DeepSeek 兼容 OpenAI 格式，在 Camel 中我们可以通过指定 api_key 和 base_url 来调用
    # model_config = ChatGPTConfig(temperature=0.7) # 法师说话需要一点灵性，不要太死板
    
    # 使用 ModelFactory 创建模型实例
    # 虽然 ModelType 可能没写 DeepSeek，但由于接口兼容，我们传对应的参数即可
    deepseek_model = get_deepseek_model(temperature=0.6)
    
    # 3. 启动 CamelAI 角色扮演
    role_play_session = RolePlaying(
        assistant_role_name="禅宗法师",
        user_role_name="求法学生",
        task_prompt=task_prompt,
        with_task_specify=False,
        assistant_agent_kwargs=dict(model=deepseek_model),
        user_agent_kwargs=dict(model=deepseek_model),
        task_specify_agent_kwargs=dict(model=deepseek_model),
    )
    
    # 4. 获取法师的开示
    # Camel 会模拟一场对话，我们取第一轮深度回复
    assistant_msg, _ = role_play_session.step("请法师慈悲指点迷津。")
    if assistant_msg.msg is not None:
        content = assistant_msg.msg.content
        
        # ✂️✂️✂️ 关键修改：手动切除 CamelAI 的样板文字 ✂️✂️✂️
        content = content.replace("Solution:", "").replace("Next request.", "")
        
        # 去掉首尾多余的空格
        return content.strip()
    else:
        return "法师正在入定，未给予言语回应。"