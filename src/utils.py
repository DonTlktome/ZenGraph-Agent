import os
from camel.models import ModelFactory
from .config import OPENAI_API_KEY, MODEL_NAME

# 定义常量，方便管理
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

def get_deepseek_model(temperature: float = 0.7):
    """
    统一的模型获取入口。
    
    Args:
        temperature (float): 创造力参数，默认 0.7。
                         HyDE 这种需要想象力的可以设高点 (0.8-0.9)，
                         严谨的回答可以设低点 (0.3-0.5)。
    """
    # 确保环境变量被正确设置 (双重保险)
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"  # 或者是 /v1
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    print(f"🛠️ [System]正在初始化 DeepSeek 模型 (Temp={temperature})...")

    return ModelFactory.create(
        model_platform="openai",
        model_type="deepseek-chat", # 这里建议直接写死或从 config 读
        api_key=OPENAI_API_KEY,
        model_config_dict={"temperature": temperature}
    )