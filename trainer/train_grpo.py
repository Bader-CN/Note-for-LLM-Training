import os
import sys
import argparse

# 将项目根目录添加到系统路径, 以便能够导入项目内的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 指定 GPU 可见性
# - 例如存在两块GPU, 但两块显卡规格不同, 可以用此方法来指定 GPU 设备
# - 必须要在 import torch 之前执行, 否则无效
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from pathlib import Path
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel
from trl import GRPOTrainer, GRPOConfig
# accuracy_reward 需要数据集里包含一个 'solution' 字段
from trl.rewards import accuracy_reward

from model.model_llm_v1 import LLMv1Config, LLMv1ForCausalLM
from trainer.train_utils import get_last_checkpoint

# 获取项目根目录路径
project_root = Path(__file__).resolve().parent.parent


def process_dataset(dataset_path):
    """
    预处理 GRPO 数据集, 将其转换为 HuggingFace Trainer 可接受的标准格式

    注意! 加载在线数据集, 可用于测试: load_dataset("trl-lib/DeepMath-103K")
    如果需要禁止 HF 联网, 可以设置 os.environ["HF_HUB_OFFLINE"] = "1"

    Args:
    - dataset_path: 数据集路径, 相对于项目根目录

    Returns:
    - DatasetDict
    """
    abs_ds_path = os.path.join(project_root, dataset_path)
    dataset = load_dataset("json", data_files=abs_ds_path)

    def _map_func(example):
        """
        https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOTrainer.train_dataset
        rlaif-mini.jsonl 数据集的格式为 {"conversations": [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "空"}]}
        该格式不符合GRPO Tainer 的要求, 因此利用此函数来处理数据集
        """
        example["prompt"] = example["conversations"]
        if example["prompt"][1].get("content") == "空":
            example["prompt"][1]["content"] = ""
        del example["conversations"]
        return example

    # 处理数据集以便符合格式需求
    dataset = dataset.map(_map_func)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM GRPO")
    # 设置项
    parser.add_argument("--output_dir", type=str, default="./training/GRPO", help="训练保存路径")
    parser.add_argument("--train_ds", type=str, default="./dataset/rlaif-mini.jsonl", help="GRPO 数据集")
    parser.add_argument("--from_resume", type=bool, default=False, help="是否断点续训")
    parser.add_argument("--from_checkpoint", type=str, default="./training/DPO",
                        help="GRPO 训练的模型从哪里来加载模型, 默认为 DPO 里最新的 checkpoint")
    parser.add_argument("--reward_model", type=str, default="./model/internlm2-1_8b-reward", help="奖励模型路径")

    # 训练的超参数
    parser.add_argument("--max_completion_length", type=int, default=256,
                        help="Maximum length of the generated completion, default 256")
    parser.add_argument("--learning_rate", default=1e-6, type=float, help="学习率, 默认 1e-6")
    parser.add_argument("--lr_scheduler_type", default="linear", type=str,
                        help='学习率调度器类型, 默认 linear, 可选值 ["linear", "cosine", "constant", "constant_with_warmup"]')
    parser.add_argument("--warmup_steps", default=0, type=int, help="从 0 线性增加到 learning_rate 的步数")
    parser.add_argument("--batch_size", default=4, type=int, help="训练的 Batch Size")
    parser.add_argument("--num_generations", default=4, type=int,
                        help="每个 prompt 生成的对话数量, 文档默认值为 8; 实际的批处理大小 (num_processes * per_device_batch_size)")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="梯度累加次数, 可以模拟更大的 Batch Size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--optim", default="adamw_torch_fused", type=str,
                        help='默认 "adamw_torch_fused", 可选值 ["adamw_torch", "adamw_torch_fused", "adamw_hf", "sgd", "adafactor", "adamw_8bit"]')
    parser.add_argument("--weight_decay", default=0.0, type=float, help="优化器应用的权重衰减系数")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="梯度裁剪, 默认 1.0; 0(禁用裁剪), 1.0(标准), 0.5(更保守), 5.0(更不激进)")
    args = parser.parse_args()

    # 注册 Config & Model
    AutoConfig.register("llm_v1", LLMv1Config)
    AutoModelForCausalLM.register(LLMv1Config, LLMv1ForCausalLM)

    # 加载 Model & Tokenizer
    model_path = get_last_checkpoint(path=os.path.join(project_root, args.from_checkpoint))
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cuda").to(
        torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    reward_path = os.path.join(project_root, args.reward_model)
    reward_model = AutoModel.from_pretrained(reward_path, trust_remote_code=True, device_map="cuda").to(torch.bfloat16)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_path, trust_remote_code=True)


    # 自定义奖励函数
    def reward_func(prompts, completions, reward_model=reward_model, reward_tokenizer=reward_tokenizer, **kwargs):
        """
        https://huggingface.co/docs/trl/grpo_trainer#using-a-custom-reward-function
        必须要传入如下参数:
        - prompts:          list[[{"role": "user", "content": "..."}]]
        - completions       list[[{"role": "assistant", "content": "..."}]]
        - completion_ids    list[[input_ids], [input_ids]]
        - trainer_state

        通常只会用到前两个, 最简单的办法就是传入一个 **kwargs
        """
        chat_lists = []
        for i in range(len(prompts)):
            user_msg = prompts[i][0]
            assistant_msg = completions[i][0]
            chat_lists.append([user_msg, assistant_msg])

        # https://huggingface.co/internlm/internlm2-1_8b-reward
        # 奖励模型用法来自这里
        scores = reward_model.get_scores(reward_tokenizer, chat_lists)

        return scores


    # GRPO 训练需要设置 padding_side='left'
    tokenizer.padding_side = 'left'

    # 加载 dataset
    train_dataset = process_dataset(dataset_path=args.train_ds)

    # https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig
    training_args = GRPOConfig(
        # Checkpointing & Saving
        output_dir=os.path.join(project_root, args.output_dir),
        save_strategy="steps",  # 默认为 "steps, 可选 ["no", "steps", "epoch", "best"]
        save_steps=100,  # 两次保存检查点之间的更新步数, 默认 500, 仅在 save_strategy = "steps" 时生效
        save_total_limit=5,  # 最多保存几个检查点, 当 load_best_model_at_end=True 时会减去1个

        # Training Duration and Batch Size
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,

        # Learning Rate & Scheduler
        learning_rate=args.learning_rate,  # 学习率, 默认 1e-6
        lr_scheduler_type=args.lr_scheduler_type,
        # 学习率调度器类型, 默认 linear, 可选值 ["linear", "cosine", "constant", "constant_with_warmup"]
        warmup_steps=args.warmup_steps,  # 从 0 线性增加到 learning_rate 的步数

        # Optimizer
        optim=args.optim,
        # 默认 "adamw_torch", 可选值 ["adamw_torch", "adamw_torch_fused", "adamw_hf", "sgd", ""adafactor"", "adamw_8bit"]
        weight_decay=args.weight_decay,  # 优化器应用的权重衰减系数, 默认 0.0

        # Regularization & Training Stability
        max_grad_norm=args.max_grad_norm,  # 梯度裁剪, 默认 1.0; 0(禁用裁剪), 1.0(标准), 0.5(更保守), 5.0(更不激进)

        # 混合精度 Mixed Precision Training
        bf16=True,  # 使用 bfloat16 精度
        bf16_full_eval=True,  # 使用 bfloat16 精度进行评估

        # Logging & Monitoring Training
        logging_strategy="steps",  # 默认为 "steps", 可选 ["no", "steps", "epoch"]
        logging_steps=10,  # 两次日志之间更新步数的数量

        # Experiment Tracking Integration
        report_to="tensorboard",
        # 默认 "none", 可选值 ["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "swanlab", "tensorboard", "trackio", "wandb"]

        # Evaluation
        eval_strategy="no",  # 默认为 "no", 可选 ["no", "steps", "epoch"]
        # eval_steps=0.2,

        # Best Model Tracking
        load_best_model_at_end=False,  # 是否在训练结束时加载最优模型, 默认 False, 当为 True 时, 必须启用 eval_strategy
        metric_for_best_model="loss",  # 用于评判最佳模型的依据, 默认 loss, 可选 ["loss", "accuracy", "f1", "eval_bleu"]

        # Reproducibility
        seed=42,  # 随机种子, 将在训练开始时设置, 默认值 42
        # data_seed=42,                                 # 用于数据采样时使用的随机种子, 如果未设置, 用于数据采样的随机生成器将使用与 seed 相同的种子

        # Dataloader
        dataloader_num_workers=0,  # 0 表示数据将在主进程中加载, 默认值 0
        dataloader_pin_memory=True,  # 是否在数据加载器中启用内存 pinning, 默认为 True

        # Others
        use_cpu=False,
        gradient_checkpointing=False,  # 以牺牲计算能力来节省内存 (通过在前向传播过程中清除激活值, 并在反向传播过程中重新计算它们, 从而减少内存使用)
        push_to_hub=False,

        # GRPO
        num_generations=args.num_generations,
        # 每个 prompt 生成的对话数量, 文档默认值为 8; 实际的批处理大小 (num_processes * per_device_batch_size)
        max_completion_length=args.max_completion_length,  # Maximum length of the generated completion, 默认 256
        # 损失之类型, 默认值是 "dapo"
        # # 可选值 ["grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo", "luspo"]
        loss_type="dapo",
    )

    # https://huggingface.co/docs/trl/v0.28.0/en/grpo_trainer#trl.GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        # reward_funcs=[accuracy_reward, reward_func],
        # - accuracy_reward 需要数据集里包含一个 'solution' 字段
        reward_funcs=reward_func,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset["train"],
    )

    if args.from_resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()