import os
import re
import sys

from pathlib import Path

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# 将项目根目录添加到系统路径, 以便能够导入项目内的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_last_checkpoint(path):
    """
    基于指定路径找到最新的 checkpoint 路径

    Args:
    - path:     需要寻找的绝对路径

    Returns:
    - abs_path: 找到的最新 checkpoint 绝对路径
    """
    # checkpoint 匹配表达式
    checkpoint_pattern = re.compile(r'^checkpoint-(\d+)$')

    dir_path = Path(path)
    if not dir_path.exists():
        logger.error(f"目录 {dir_path} 不存在")
    
    last_number = -1
    last_checkpoint = None

    for item in dir_path.iterdir():
        # 如果是目录
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                number = int(match.group(1))
                if number > last_number:
                    last_number = number
                    last_checkpoint = item
    
    if last_checkpoint is not None:
        # Path('a/b').resolve() 可以解析到绝对路径
        logger.info(f"发现最新 checkpoint: {str(last_checkpoint.resolve())}")
        return str(last_checkpoint.resolve())

if __name__ == "__main__":
    get_last_checkpoint(path="./training/PreTraining")