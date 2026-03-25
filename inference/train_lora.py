#!/usr/bin/env python
# train_qwen_vl_lora.py
# -*- coding: utf-8 -*-

"""
Qwen3-VL-8B + LoRA 微调脚本 (包含 TensorBoard 支持)
任务：抓取
"""

import os
# ================================
#       ENVIRONMENT CONFIG
# ================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "20"

import json
import torch
import pandas as pd
from PIL import Image

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from utils import getdata

# ======================
# 1. 加载模型 & tokenizer
# ======================
print("加载 Qwen3-VL-8B 基座模型...")

model, tokenizer = FastVisionModel.from_pretrained(
    "/home/user/qwen3_vl_8b",
    use_gradient_checkpointing="unsloth",
)

# ======================
# 2. 配置 LoRA
# ======================
print("配置 LoRA 适配器...")

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# ======================
# 3. 数据集处理
# ======================
print("加载并转换训练数据...")
train_dataset = getdata.get_dialog_list()
test_dataset=getdata.load_valit_dataset()

print(f"有效训练样本数: {len(train_dataset)}")

# ======================
# 4. 训练配置 (修改此处以支持 TensorBoard)
# ======================
FastVisionModel.for_training(model)



trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset, # 必须传入验证集
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        # max_steps=100,             # 增加步数以体现评估和保存
        num_train_epochs=100,
        learning_rate=5e-5,
        logging_steps=1,
        warmup_steps=2,
        # 保存与续传配置
        output_dir="outputs",
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,        # 硬盘保护：只留最后2个
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        
        report_to="tensorboard",
        logging_dir="outputs/runs",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)

# 自动检测是否存在 checkpoint 以实现断点续传
checkpoint_dir = "outputs"
last_checkpoint = None
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    # 逻辑：如果文件夹不为空，则从最新位置恢复
    last_checkpoint = True 

print(f"开始训练，续传状态: {last_checkpoint}")
trainer.train(resume_from_checkpoint=last_checkpoint)

# 最终保存（此时保存的是 load_best_model_at_end 选出的最优模型）
model.save_pretrained("/home/user/wmw_vs/models/best_lora_model")