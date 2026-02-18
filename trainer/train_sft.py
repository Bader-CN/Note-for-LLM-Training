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
from trl import SFTConfig, SFTTrainer

from model.model_llm_v1 import LLMv1Config, LLMv1ForCausalLM
from trainer.train_utils import get_last_checkpoint


project_root = Path(__file__).resolve().parent.parent


def process_dataset(dataset_path):
    """
    预处理 SFT 数据集, 将其转换为 HuggingFace Trainer 可接受的标准格式

    Args:
    - dataset_path: 数据集路径, 相对于项目根目录

    Returns:
    - DatasetDict

    SFTTrainer 支持 "LLM" 和 "prompt-completion" 数据集
    当提供对话数据集时, SFTTrainer 将自动将聊天模板应用于该数据集
    #########################################################################
    # Standard language modeling
        {"text": "The sky is blue."}

    # Conversational language modeling
        {"messages": [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."}
        ]}

    # Standard prompt-completion
        {"prompt": "The sky is", "completion": " blue."}

    # Conversational prompt-completion
        {
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}]
        }
    #########################################################################
    """
    abs_ds_path = os.path.join(project_root, dataset_path)
    dataset = load_dataset("json", data_files=abs_ds_path)

    def _map_func(example):
        """
        sft_mini_512.jsonl 数据集的格式为 {"conversations": [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "xxx"}]}
        该格式不符合 SFTTrainer 的要求, 因此利用此函数来处理数据集
        """
        example["messages"] = example["conversations"]
        del example["conversations"]
        return example
    
    # 处理数据集以便符合格式需求
    dataset = dataset.map(_map_func)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Supervised Fine-Tuning")
    # 设置项
    parser.add_argument("--output_dir", type=str, default="./training/SFT", help="训练保存路径")
    parser.add_argument("--train_ds", type=str, default="./dataset/sft_mini_512.jsonl", help="SFT 数据集")
    parser.add_argument("--from_resume", type=bool, default=False, help="是否断点续训")
    parser.add_argument("--from_checkpoint", type=str, default="./training/PreTraining", help="SFT 训练的模型从哪里来加载模型, 默认为 PreTraining 里最新的 checkpoint")

    # 训练的超参数
    parser.add_argument("--max_length", type=int, default=1024, help="Tokens 最大长度")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="学习率, 默认 2e-5")
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str, help='学习率调度器类型, 默认 linear, 可选值 ["linear", "cosine", "constant", "constant_with_warmup"]')
    parser.add_argument("--warmup_steps", default=1000, type=int, help="从 0 线性增加到 learning_rate 的步数")
    parser.add_argument("--batch_size", default=16, type=int, help="训练的 Batch Size")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累加次数, 可以模拟更大的 Batch Size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--optim", default="adamw_torch_fused", type=str, help='默认 "adamw_torch_fused", 可选值 ["adamw_torch", "adamw_torch_fused", "adamw_hf", "sgd", "adafactor", "adamw_8bit"]')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="优化器应用的权重衰减系数")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁剪, 默认 1.0; 0(禁用裁剪), 1.0(标准), 0.5(更保守), 5.0(更不激进)")
    parser.add_argument("--loss_type", default="nll", type=str, help='损失类型, 默认值为 "nll", 可选 ["nll(负对数似然)", "dft(动态微调)"]')
    parser.add_argument("--assistant_only_loss", default=True, type=bool, help='是否仅计算 "Assistant" 上的损失, 如果为 True 仅用于对话数据集')
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

    # https://huggingface.co/docs/trl/v0.28.0/en/sft_trainer#trl.SFTConfig
    # https://huggingface.co/docs/transformers/v5.1.0/en/main_classes/trainer#transformers.TrainingArguments
    training_args = SFTConfig(
        # Checkpointing & Saving
        output_dir=os.path.join(project_root, args.output_dir),
        save_strategy="steps",                          # 默认为 "steps, 可选 ["no", "steps", "epoch", "best"]
        save_steps=1000,                                # 两次保存检查点之间的更新步数, 默认 500, 仅在 save_strategy = "steps" 时生效
        save_total_limit=5,                             # 最多保存几个检查点, 当 load_best_model_at_end=True 时会减去1个

        # Training Duration and Batch Size
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,         # SFTConfig 默认值为 3
        max_length=args.max_length,                     # SFTConfig 默认值为 1024

        # Parameters that control the training
        loss_type=args.loss_type,                       # 损失类型, 默认值为 "nll", 可选 ["nll(负对数似然)", "dft(动态微调)"]
        # 是否仅计算 "Assistant" 上的损失 (默认为 False)
        # https://huggingface.co/docs/trl/sft_trainer#train-on-assistant-messages-only
        # - 如果为 True,  则仅计算 "Assistant" 回复上的损失, 仅适用于对话数据集
        # - 如果为 False, 则计算整个序列上的损失
        assistant_only_loss=args.assistant_only_loss,                       
        # 是否仅计算序列的补全部分上的损失 (默认为 None)
        # https://huggingface.co/docs/trl/sft_trainer#train-on-completion-only
        # - 如果为 True,  则仅计算补全部分的损失, 仅适用于 prompt-completion 数据集
        # - 如果为 False, 则计算整个序列的损失
        # - 如果为 None,  则行为取决于数据集
            # 对于 prompt-completion 数据集, 计算补全部分的损失
            # 对于语言建模数据集, 计算整个序列的损失
        completion_only_loss=None,                   

        # Learning Rate & Scheduler
        learning_rate=args.learning_rate,               # 学习率, 默认 2e-5
        lr_scheduler_type=args.lr_scheduler_type,       # 学习率调度器类型, 默认 linear, 可选值 ["linear", "cosine", "constant", "constant_with_warmup"]
        warmup_steps=args.warmup_steps,                 # 从 0 线性增加到 learning_rate 的步数, SFTConfig 默认为 0

        # Optimizer
        optim=args.optim,                               # 默认 "adamw_torch_fused", 可选值 ["adamw_torch", "adamw_torch_fused", "adamw_hf", "sgd", "adafactor", "adamw_8bit"]
        weight_decay=args.weight_decay,                 # 优化器应用的权重衰减系数, SFTConfig 默认为 0.0

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
        dataset_text_field="messages",                  # 包含数据集中的文本数据的列名, 默认为 "text"

        # Others
        use_cpu=False,
        gradient_checkpointing=False,                   # 以牺牲计算能力来节省内存 (通过在前向传播过程中清除激活值, 并在反向传播过程中重新计算它们, 从而减少内存使用)
        push_to_hub=False,
        # https://huggingface.co/docs/trl/v0.28.0/en/sft_trainer#packing
        packing=False,                                  # 支持示例打包, 即将多个示例打包到同一输入序列中,以提高训练效率
        # 是否在不进行填充的情况下执行前向传播, 通过将批次中的所有序列合并为单个连续序列, 可以减少内存
        # - 仅支持 FlashAttention 2 或 3
        # - Windows 11 上启用会崩溃 (RTX 3090 / RTX A6000 似乎不支持)
        padding_free=False,
    )

    # https://huggingface.co/docs/trl/v0.28.0/en/sft_trainer#trl.SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        processing_class=tokenizer,
        # 通常不用设置 data_collator, SFTTrainer 会自动处理
        # - 如果模型是语言模型, 则默认使用 DataCollatorForLanguageModeling
        # - 如果模型是视觉-语言模型, 则默认使用 DataCollatorForVisionLanguageModeling
    )

    if args.from_resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
