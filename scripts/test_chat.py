import os
import sys

# 将项目根目录添加到系统路径, 以便能够导入项目内的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# 输入测试内容
inputs = tokenizer("请简单介绍一下你自己吧.", return_tensors="pt").to(model.device)

# 输出流式内容
# https://huggingface.co/docs/transformers/v5.2.0/generation_features
output = model.generate(**inputs, max_new_tokens=1024, do_sample=True, streamer=streamer)
