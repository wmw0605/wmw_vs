# WMW-VS: 机器人视觉-语言-动作规划项目

本项目围绕家居场景中的机器人任务执行，整合了以下能力：

- 基于 Qwen3-VL 的视觉理解与目标定位（bbox + point）
- LoRA 微调（Unsloth + TRL）
- 基于技能库的 RAG 动作规划（FAISS + SentenceTransformer）
- 与 SAM2 分割流程联动（跨 Python 环境调用）

项目目标是将自然语言任务 + 图像输入，转化为可执行的机器人动作序列（JSON）。

## 1. 项目结构

```text
wmw_vs/
├── train/
│   └── train_qwen_vl_lora.py           # LoRA 训练主脚本
├── inference/
│   ├── inference.py                    # 视觉定位 + 跨环境触发 SAM2
│   ├── rag.py                          # 技能库 RAG 推理（单轮/多轮）
│   ├── train_lora.py                   # 含验证与 checkpoint 的训练脚本
│   ├── rag_test.py                     # 简化版 RAG 测试
│   └── lora_inferrence_test.py         # 模型行为对比测试脚本
├── robotic_action_knowledge/
│   ├── skills.json                     # 原子技能库
│   ├── skills.index                    # FAISS 索引
│   └── faiss_build_index.py            # 重新构建索引
├── dataset/
│   ├── Fine-tuningdataset/             # 训练集目录（按数字子目录）
│   └── validate_dataset.json           # 验证集（JSONL）
├── utils/
│   └── getdata.py                      # 训练/验证数据读取
├── back/
│   └── test_run.py                     # SAM2 分割脚本（被跨环境调用）
├── models/                             # LoRA 模型输出目录
├── outputs/                            # 训练输出与 TensorBoard 日志
├── shared_coords.json                  # 跨脚本共享坐标文件
└── test_torch.py                       # CUDA/PyTorch 快速检查
```

## 2. 核心流程

### 2.1 感知与分割链路

1. `inference/inference.py` 使用 Qwen3-VL 对输入图像做 grounding，输出目标框和关键点。
2. 推理结果写入 `shared_coords.json`。
3. 脚本通过 `subprocess` 调用另一个环境中的 `back/test_run.py` 运行 SAM2 分割。

### 2.2 RAG 动作规划链路

1. `robotic_action_knowledge/skills.json` 提供机器人原子技能定义。
2. `robotic_action_knowledge/skills.index` 提供向量检索索引。
3. `inference/rag.py` 读取图像与坐标，检索技能后生成动作规划文本/JSON。

### 2.3 LoRA 训练链路

- 训练脚本：`train/train_qwen_vl_lora.py` 或 `inference/train_lora.py`
- 数据加载：`utils/getdata.py`
- 输出目录：`outputs/` 与 `models/`

## 3. 环境要求

建议环境：

- Linux + NVIDIA GPU（推荐支持 bf16）
- Python 3.10+
- CUDA 可用

主要依赖（按代码导入汇总）：

- `torch`
- `transformers`
- `trl`
- `unsloth`
- `peft`
- `sentence-transformers`
- `faiss-gpu`（或 CPU 版 `faiss-cpu`）
- `Pillow`
- `numpy`
- `pandas`
- `hydra-core`
- `matplotlib`
- `qwen-vl-utils`

可先最小化安装（示例）：

```bash
pip install torch transformers trl unsloth peft sentence-transformers faiss-gpu pillow numpy pandas hydra-core matplotlib qwen-vl-utils
```

如果没有 GPU，可替换为：

```bash
pip install faiss-cpu
```

## 4. 路径配置说明（非常重要）

当前多个脚本使用了绝对路径（例如 `/home/user/qwen3_vl_8b`、`/home/user/Desktop/Grounded-SAM-2/...`）。

首次运行前，请按你的机器路径修改以下文件中的变量：

- `train/train_qwen_vl_lora.py`
- `inference/train_lora.py`
- `inference/inference.py`
- `inference/rag.py`
- `back/test_run.py`

常见需要修改的字段：

- 模型路径：`BASE_MODEL_PATH`、`model_path`
- LoRA 路径：`LORA_PATH`
- 图片路径：`img_path`、`--image`
- SAM2 路径：`SAM2_PYTHON_PATH`、`SAM2_SCRIPT_PATH`、`CONFIG_DIR`、`checkpoint`

## 5. 快速开始

### 5.1 检查 PyTorch/CUDA

```bash
python test_torch.py
```

### 5.2 构建/重建技能索引（可选）

```bash
python robotic_action_knowledge/faiss_build_index.py
```

### 5.3 运行视觉定位 + SAM2 分割链路

```bash
python inference/inference.py
```

运行后会生成或更新：

- `shared_coords.json`
- `vlm_detection_result.png`
- SAM2 输出图（取决于 `back/test_run.py` 内配置）

### 5.4 运行 RAG 动作规划

单轮模式：

```bash
python inference/rag.py --image /path/to/your/image.jpg
```

多轮模式：

```bash
python inference/rag.py --chat --image /path/to/your/image.jpg
```

### 5.5 LoRA 训练

基础训练脚本：

```bash
python train/train_qwen_vl_lora.py
```

含验证与 checkpoint 训练脚本：

```bash
python inference/train_lora.py
```

TensorBoard（默认日志目录 `outputs/runs`）：

```bash
tensorboard --logdir outputs/runs --port 6006
```

## 6. 数据格式说明

### 6.1 训练数据

- 目录：`dataset/Fine-tuningdataset/<数字子目录>/dialo*.json`
- `utils/getdata.py` 会按数字目录排序并读取所有匹配文件。

### 6.2 验证数据

- 文件：`dataset/validate_dataset.json`
- 格式：JSONL（每行一个 JSON 对象）

## 7. 常见问题

1. 模型加载失败

- 检查本地模型路径是否存在（如 `/home/user/qwen3_vl_8b`）。
- 检查环境是否安装了 `transformers`、`unsloth`、`peft`。

2. FAISS 检索报错

- 确认 `skills.index` 与当前 embedding 模型维度匹配。
- 不匹配时重新运行 `faiss_build_index.py`。

3. SAM2 调用失败

- 确认 `SAM2_PYTHON_PATH` 指向可执行解释器。
- 确认 `CONFIG_DIR` 与 `checkpoint` 路径正确。
- 确认 Grounded-SAM-2 项目已完整安装依赖。

## 8. 备注

`models/` 与 `outputs/` 下已有自动生成的模型卡 README，属于训练产物；本 README 为工程级总览文档。
