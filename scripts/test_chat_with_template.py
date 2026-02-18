import os
import sys
# 将项目根目录添加到系统路径, 以便能够导入项目内的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loguru import logger
logger.remove()
# 可以将日志调整为 DEBUG 来查看聊天模板的实际输出
logger.add(sys.stderr, level="INFO")

import torch

from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_llm_v1 import LLMv1Config, LLMv1ForCausalLM
from trainer.train_utils import get_last_checkpoint

# 注册 Config & Model
AutoConfig.register("llm_v1", LLMv1Config)
AutoModelForCausalLM.register(LLMv1Config, LLMv1ForCausalLM)

# 获取项目根目录路径
project_root = Path(__file__).resolve().parent.parent

# 加载 Model & Tokenizer
model_path = get_last_checkpoint(path=os.path.join(project_root, "./training/DPO"))
model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 常见流式文本器
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 定义对话历史
messages = [
    {"role": "system", "content": "你是一个有帮助的 AI 助手."},
    {"role": "user", "content": "请介绍一下什么是大语言模型(LLM)?"},
]

# 应用聊天模板
prompt = tokenizer.apply_chat_template(
    messages,
    # 返回字符串而非 token ids
    tokenize=False,
    # 添加生成提示符 (如 <|assistant|>)
    add_generation_prompt=True,
)
logger.debug(prompt)

# 格式化输入内容
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 输出流式内容
# https://huggingface.co/docs/transformers/v5.2.0/generation_features
output = model.generate(**inputs, max_new_tokens=1024, do_sample=True, streamer=streamer)
