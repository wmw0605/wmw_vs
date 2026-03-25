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

print(f"有效训练样本数: {len(train_dataset)}")

# ======================
# 4. 训练配置 (修改此处以支持 TensorBoard)
# ======================
FastVisionModel.for_training(model)

# 定义输出和日志目录
output_dir = "/home/user/wmw_vs/outputs"
logging_dir = os.path.join(output_dir, "runs") # TensorBoard 日志存放路径

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        # max_steps=100,
        num_train_epochs=100,
        learning_rate=5e-5,
        logging_steps=1,          # 每隔一步记录一次日志，方便观察曲线
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="tensorboard",   # 将日志上报到 tensorboard
        logging_dir=logging_dir,   # 指定 tensorboard 文件保存位置
        
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
    max_seq_length=2048,
)

print("开始训练...")
trainer.train()

# ======================
# 5. 保存 LoRA
# ======================
print("保存 LoRA 模型...")
model.save_pretrained("/home/user/wmw_vs/models/lora_model_1")
tokenizer.save_pretrained("/home/user/wmw_vs/models/insurance_lora_model_1")

print("训练完成 ✅")