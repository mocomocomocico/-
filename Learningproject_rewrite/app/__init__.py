"""应用核心包。

导入本包时会按顺序完成两件初始化工作：
1. 加载项目根目录的 .env 环境变量（供所有子模块读取配置）；
2. 设置 Hugging Face 离线加载环境变量，避免网络不可达时卡死。

注意：第 2 步必须在任何 transformers / sentence-transformers 相关模块被导入
之前执行，因此这里的初始化逻辑不允许被延迟导入。
"""

import os
from pathlib import Path

# 1. 加载 .env（文件不存在时静默跳过）
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# 2. 默认离线加载本地模型（用户可通过 .env / 系统环境变量覆盖）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
