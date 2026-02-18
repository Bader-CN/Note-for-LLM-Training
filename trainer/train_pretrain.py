import os
import re
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
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from model.model_llm_v1 import LLMv1Config, LLMv1ForCausalLM

# 获取项目根目录
# Path(__file__).resolve() 总是指向当前模块的绝对路径(文件)
project_root = Path(__file__).resolve().parent.parent

def process_dataset(dataset_path, tokenizer, max_length=512, encoding_column="text"):
    """
    预处理 dataset, 将数据集调整为符合 HuggingFace Trainer 接受的标准

    Args:
    - dataset_path:     数据集路径, 必须是相对于项目根目录
    - tokenizer:        分词器对象
    - max_length:       Tokens 最大长度, 默认 512
    - encoding_column:  需要被编码的列名, 默认 text

    Returns:
    - DatasetDict:      包含 "train" split 的 HuggingFace Dataset, 其中每条样本为:
                        {
                            "input_ids": List[int], 
                            "attention_mask": List[int],  # collator 自动添加
                        }

    关于 HF 数据集格式的总结
    #########################################################################
    - 对于 Causal Language Modeling (因果语言建模, 如预训练)
      1.训练时只需提供 "input_ids" 字段即可
      2.Trainer + DataCollatorForLanguageModeling(mlm=False) 会自动处理
        - 将 input_ids 复制为 labels
        - 执行 shift right 操作
        - 将 padding token 的 label 设为 -100 (忽略其 loss)
      3.注意事项
        - 若 tokenizer 无 pad_token, 需手动添加 (避免 collator 报错)
        - 不需要显式提供 "labels"、"attention_mask" 等字段 (collator 会自动处理)
        - 数据集中应只保留 "input_ids", 其余的最好删掉
    #########################################################################
    """
    abs_ds_path = os.path.join(project_root, dataset_path)

    def _tokenize_function(examples):
        """
        注意: 这里不需要 padding, DataCollator 会动态处理
        """
        return tokenizer(
            examples[encoding_column],  # 默认为数据集的 text 列
            truncation=True,            # 允许截断
            max_length=max_length,        
        )
    
    # 对数据集做预处理
    dataset = load_dataset("json", data_files=abs_ds_path)
    hf_dataset = dataset.map(
        function=_tokenize_function,
        batched=True,
        remove_columns=[encoding_column],   # 移除原始列, 因为训练时不会用到此列
        # num_proc=4                        # Windows 上会有问题, 所以不开启
    )

    return hf_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Pretraining")
    # 设置项
    parser.add_argument("--output_dir", type=str, default="./training/PreTraining", help="训练保存路径")
    parser.add_argument("--train_ds", type=str, default="./dataset/pretrain_hq.jsonl", help="预训练数据集")
    parser.add_argument("--from_resume", default=False, type=bool, help="是否断点续训")
    # 训练的超参数
    parser.add_argument("--max_length", type=int, default=512, help="Tokens 最大长度")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率, 默认 5e-5")
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str, help='学习率调度器类型, 默认 linear, 可选值 ["linear", "cosine", "constant", "constant_with_warmup"]')
    parser.add_argument("--warmup_steps", default=5000, type=int, help="从 0 线性增加到 learning_rate 的步数")
    parser.add_argument("--batch_size", default=32, type=int, help="训练的 Batch Size")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累加次数, 可以模拟更大的 Batch Size")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--optim", default="adamw_torch", type=str, help='默认 "adamw_torch", 可选值 ["adamw_torch", "adamw_torch_fused", "adamw_hf", "sgd", "adafactor", "adamw_8bit"]')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="优化器应用的权重衰减系数")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁剪, 默认 1.0; 0(禁用裁剪), 1.0(标准), 0.5(更保守), 5.0(更不激进)")
    args = parser.parse_args()

    # 模型 & 分词器
    model = LLMv1ForCausalLM().to(torch.bfloat16)   # 将模型转换为半精度
    # 处理将路径识别为 repo_id 的问题
    # huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96
    model_dir = "../model" if re.findall("trainer", os.getcwd()) else "./model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # 获取数据集
    train_dataset = process_dataset(dataset_path=args.train_ds, tokenizer=tokenizer, max_length=args.max_length, encoding_column="text")

    # https://huggingface.co/docs/transformers/v5.1.0/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
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
        learning_rate=args.learning_rate,               # 学习率, 默认 5e-5
        lr_scheduler_type=args.lr_scheduler_type,       # 学习率调度器类型, 默认 linear, 可选值 ["linear", "cosine", "constant", "constant_with_warmup"]
        warmup_steps=args.warmup_steps,                 # 从 0 线性增加到 learning_rate 的步数

        # Optimizer
        optim=args.optim,                               # 默认 "adamw_torch", 可选值 ["adamw_torch", "adamw_torch_fused", "adamw_hf", "sgd", ""adafactor"", "adamw_8bit"]
        weight_decay=args.weight_decay,                 # 优化器应用的权重衰减系数

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
    )

    # https://huggingface.co/docs/transformers/v5.1.0/en/main_classes/trainer#transformers.Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        processing_class=tokenizer,
        # Causal LM ≠ Masked LM (MLM), 因此 mlm 需要设置为 False
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    if args.from_resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()