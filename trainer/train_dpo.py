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
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from model.model_llm_v1 import LLMv1Config, LLMv1ForCausalLM
from trainer.train_utils import get_last_checkpoint

# 获取项目根目录路径
project_root = Path(__file__).resolve().parent.parent


def process_dataset(dataset_path):
    """
    预处理 DPO 数据集, 将其转换为 HuggingFace Trainer 可接受的标准格式

    Args:
    - dataset_path: 数据集路径, 相对于项目根目录

    Returns:
    - DatasetDict
    """
    abs_ds_path = os.path.join(project_root, dataset_path)
    dataset = load_dataset("json", data_files=abs_ds_path)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Direct Preference Optimization")
    # 设置项
    parser.add_argument("--output_dir", type=str, default="./training/DPO", help="训练保存路径")
    parser.add_argument("--train_ds", type=str, default="./dataset/dpo.jsonl", help="SFT 数据集")
    parser.add_argument("--from_resume", type=bool, default=False, help="是否断点续训")
    parser.add_argument("--from_checkpoint", type=str, default="./training/SFT", help="DPO 训练的模型从哪里来加载模型, 默认为 SFT 里最新的 checkpoint")

    # 训练的超参数
    parser.add_argument("--max_length", type=int, default=1024, help="Tokens 最大长度")
    parser.add_argument("--learning_rate", default=1e-6, type=float, help="学习率, 默认 1e-6")
    parser.add_argument("--lr_scheduler_type", default="linear", type=str, help='学习率调度器类型, 默认 linear, 可选值 ["linear", "cosine", "constant", "constant_with_warmup"]')
    parser.add_argument("--warmup_steps", default=0, type=int, help="从 0 线性增加到 learning_rate 的步数")
    parser.add_argument("--batch_size", default=8, type=int, help="训练的 Batch Size")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累加次数, 可以模拟更大的 Batch Size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--optim", default="adamw_torch_fused", type=str, help='默认 "adamw_torch_fused", 可选值 ["adamw_torch", "adamw_torch_fused", "adamw_hf", "sgd", "adafactor", "adamw_8bit"]')
    parser.add_argument("--weight_decay", default=0.0, type=float, help="优化器应用的权重衰减系数")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁剪, 默认 1.0; 0(禁用裁剪), 1.0(标准), 0.5(更保守), 5.0(更不激进)")
    args = parser.parse_args()

    # 注册 Config & Model
    AutoConfig.register("llm_v1", LLMv1Config)
    AutoModelForCausalLM.register(LLMv1Config, LLMv1ForCausalLM)

    # 加载 Model & Tokenizer
    model_path = get_last_checkpoint(path=os.path.join(project_root, args.from_checkpoint))
    model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载 dataset
    train_dataset = process_dataset(dataset_path=args.train_ds)

    # https://huggingface.co/docs/trl/v0.28.0/en/dpo_trainer#trl.DPOConfig
    training_args = DPOConfig(
        # Checkpointing & Saving
        output_dir=os.path.join(project_root, args.output_dir),
        save_strategy="steps",                          # 默认为 "steps, 可选 ["no", "steps", "epoch", "best"]
        save_steps=1000,                                # 两次保存检查点之间的更新步数, 默认 500, 仅在 save_strategy = "steps" 时生效
        save_total_limit=5,                             # 最多保存几个检查点, 当 load_best_model_at_end=True 时会减去1个

        # Training Duration and Batch Size
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,      
        num_train_epochs=args.num_train_epochs,

        # Learning Rate & Scheduler
        learning_rate=args.learning_rate,               # 学习率, 默认 1e-6
        lr_scheduler_type=args.lr_scheduler_type,       # 学习率调度器类型, 默认 linear, 可选值 ["linear", "cosine", "constant", "constant_with_warmup"]
        warmup_steps=args.warmup_steps,                 # 从 0 线性增加到 learning_rate 的步数

        # Optimizer
        optim=args.optim,                               # 默认 "adamw_torch", 可选值 ["adamw_torch", "adamw_torch_fused", "adamw_hf", "sgd", ""adafactor"", "adamw_8bit"]
        weight_decay=args.weight_decay,                 # 优化器应用的权重衰减系数, 默认 0.0

        # Regularization & Training Stability
        max_grad_norm=args.max_grad_norm,               # 梯度裁剪, 默认 1.0; 0(禁用裁剪), 1.0(标准), 0.5(更保守), 5.0(更不激进)

        # 混合精度 Mixed Precision Training
        bf16=True,                                      # 使用 bfloat16 精度
        bf16_full_eval=True,                            # 使用 bfloat16 精度进行评估

        # Logging & Monitoring Training
        logging_strategy="steps",                       # 默认为 "steps", 可选 ["no", "steps", "epoch"]
        logging_steps=100,                              # 两次日志之间更新步数的数量

        # Experiment Tracking Integration
        report_to="tensorboard",                        # 默认 "none", 可选值 ["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "swanlab", "tensorboard", "trackio", "wandb"]

        # Evaluation
        eval_strategy="no",                             # 默认为 "no", 可选 ["no", "steps", "epoch"]
        # eval_steps=0.2,

        # Best Model Tracking
        load_best_model_at_end=False,                   # 是否在训练结束时加载最优模型, 默认 False, 当为 True 时, 必须启用 eval_strategy
        metric_for_best_model="loss",                   # 用于评判最佳模型的依据, 默认 loss, 可选 ["loss", "accuracy", "f1", "eval_bleu"]

        # Reproducibility
        seed=42,                                        # 随机种子, 将在训练开始时设置, 默认值 42
        # data_seed=42,                                 # 用于数据采样时使用的随机种子, 如果未设置, 用于数据采样的随机生成器将使用与 seed 相同的种子

        # Dataloader
        dataloader_num_workers=0,                       # 0 表示数据将在主进程中加载, 默认值 0
        dataloader_pin_memory=True,                     # 是否在数据加载器中启用内存 pinning, 默认为 True
        # label_names=list[str]                         # 字典中用于表示标签的键的列表, 默认情况下将采用模型接受的参数名称列表 (其中包含"label"一词), 除非是 XxxForQuestionAnswering 模型

        # Others
        use_cpu=False,
        gradient_checkpointing=False,                   # 以牺牲计算能力来节省内存 (通过在前向传播过程中清除激活值, 并在反向传播过程中重新计算它们, 从而减少内存使用)
        push_to_hub=False,

        # DPO
        dataset_num_proc=8,                             # 用于处理数据集的进程数量
        # 是否预计算参考模型的对数概率 (默认 False)
        # - 将此设置为 True 允许在训练过程中无需使用参考模型进行训练, 这有助于减少 GPU 内存使用
        # - 如果设置为 False 则在训练期间将使用参考模型来计算对数概率
        precompute_ref_log_probs=True,
        # 损失之类型, 默认值是 "sigmoid"
        # 可选值 ["sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust", "bco_pair", "sppo_hard", "aot", "aot_unpaired", "discopop", "apo_zero", "apo_down", "sft"]
        loss_type="sigmoid",
        # 计算 "策略模型" 与 "参考模型" 之间的差异的正则化函数 (默认 "reverse_kl")
        # 可选值 ["reverse_kl", "js_divergence", "alpha_divergence"]
        f_divergence_type="reverse_kl",
        f_alpha_divergence_coef=1.0,                    # α-divergence 正则化函数中用于 DPO 损失的α系数
        sync_ref_model=False,                           # 是否每 ref_model_sync_steps 步同步参考模型与当前模型, 默认 False (这是 TR-DPO 方法. 需要明确指定 ref_model)
        # ref_model_mixup_alpha=0.6,                      # 来自 TR-DPO 论文的 α 参数, 用于控制在更新过程中当前策略和先前参考策略之间的混合, 默认 0.6, 需要 sync_ref_model=True
        # ref_model_sync_steps=512,                       # 来自 TR-DPO 论文的 τ 参数, 决定了当前策略与参考策略的同步频率, 默认 512, 需要 sync_ref_model=True           
    )

    # https://huggingface.co/docs/trl/v0.28.0/en/dpo_trainer#trl.DPOTrainer
    trainer = DPOTrainer(
        model=model,
        # ref_model 可以额外指定, 默认创建一个与要优化的模型具有相同架构的参考模型
        # 如果 DPOConfig(sync_ref_model=True), 则必须要指定 ref_model=<model>
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_dataset["train"],
    )

    if args.from_resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()