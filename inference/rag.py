import os
import json
import torch
import faiss
import time
import argparse
import numpy as np
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from PIL import Image
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor

# ================= 路径与环境配置 =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
SKILLS_JSON_PATH = "/home/user/wmw_vs/robotic_action_knowledge/skills.json"
SKILLS_INDEX_PATH = "/home/user/wmw_vs/robotic_action_knowledge/skills.index"
EMBED_MODEL_PATH = "/home/user/models/paraphrase-multilingual-MiniLM-L12-v2"
BASE_MODEL_PATH = "/home/user/qwen3_vl_8b"
LORA_PATH = "/home/user/wmw_vs/models/lora_model"
COORDS_PATH = "/home/user/wmw_vs/shared_coords.json"

# -----------------------------------------------------------------------------
# 1. 初始化组件
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] 当前运行设备: {device}")

print("[*] 正在加载 Embedding 模型...")
embedding_model = SentenceTransformer(EMBED_MODEL_PATH, device=device)

print("[*] 正在加载 JSON 技能数据库...")
with open(SKILLS_JSON_PATH, "r", encoding="utf-8") as f:
    skill_database = json.load(f)

print("[*] 正在加载技能的向量索引...")
cpu_index = faiss.read_index(SKILLS_INDEX_PATH)

print("[*] 正在加载 Qwen3-VL-8B 模型...")
try:
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer 
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    # 如需加载 LoRA，取消下行注释
    # model = PeftModel.from_pretrained(model, LORA_PATH)
    print("[*] 模型加载成功！")
except Exception as e:
    print(f"[!] 模型加载失败: {e}")
    exit()

# -----------------------------------------------------------------------------
# 2. RAG 推理核心函数 (统一支持单/多轮，严格锁定 Prompt)
# -----------------------------------------------------------------------------
def rag_generate(question: str, input_image: Image.Image, history: list):
    """
    统一推理逻辑：
    - 如果 history 为空：执行 RAG 检索、坐标读取，并构造首轮含图片的 Prompt。
    - 如果 history 不为空：追加纯文本对话，维持上下文。
    """
    
    # --- A. 仅在首轮执行：检索、坐标读取、构造原始 Prompt ---
    if not history:
        # 1. 向量检索
        query_vec = embedding_model.encode([question]).astype('float32')
        k = 5
        distances, indices = cpu_index.search(query_vec, k=k)
        
        context_skills_list = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1:
                s = skill_database[idx]
                skill_info = (
                    f"技能{i+1}: {s['name']}\n"
                    f"功能: {s['description']}\n"
                    f"接口: {s['function_call']}\n"
                    f"参数及描述: {s['parameters']}\n"
                )
                context_skills_list.append(skill_info)
        all_skills_context = "\n---\n".join(context_skills_list)

        # 2. 读取感知坐标
        coords_info = ""
        if os.path.exists(COORDS_PATH):
            with open(COORDS_PATH, "r", encoding="utf-8") as f:
                coords_data = json.load(f)
                coords_info = json.dumps(coords_data.get("targets", []), ensure_ascii=False, indent=2)

        # 3. 注入系统提示 (严格保留，不可改动)
        system_content = (
            "1.你是一个精通机器人控制的AI大脑。请根据提供的【可选原子技能库】来规划任务。并给出规划描述，以及简洁的理由"
            "2.如果任务包含多个步骤，请按顺序调用多个技能并给出参数，最后生成一个JSON格式的动作执行序列。"
            "3.严禁使用技能库中不存在的函数名称。"
            "4.生成JSON动作序列后，你需要根据当前任务进行审查，需要确保每个技能调用都合理且必要，如果缺少必要技能，请补充后重新生成。'"
            "5.同时，你需要根据图片内容识别相关物体及其空间关系，并结合技能库中的技能进行规划。"
        )
        history.append({"role": "system", "content": system_content})

        # 4. 注入首轮用户提示 (严格保留内容与格式)
        first_user_text = (
            f"### 可选原子技能库 ###\n{all_skills_context}\n\n"
            f"### 任务相关对象的感知定位包括一个二维定位框bbox和一个关键点point，其结果为： ###\n{coords_info}\n\n"
            f"### 当前任务 ###\n{question}\n\n"
            "### 执行要求 ###\n"
            "1. 识别图片中所有相关物体及其空间关系。\n"
            "2. 仅允许使用库中提供的函数接口。\n"
            "3. 动作顺序必须符合逻辑（例如：先抓取再移动）。\n"
            "4. 输出格式：[文字说明] + [JSON 动作序列]。"
        )
        history.append({
            "role": "user", 
            "content": [{"type": "image", "image": input_image}, {"type": "text", "text": first_user_text}]
        })
    else:
        # --- B. 非首轮：追加后续对话文本 ---
        history.append({
            "role": "user", 
            "content": [{"type": "text", "text": question}]
        })

    # --- C. 执行 VLM 推理 (优化性能) ---
    with torch.inference_mode():
        # 构造输入模板
        text = processor.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        
        # 寻找图像（仅在首轮 user content 中存在）
        images = []
        for m in history:
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "image":
                        images.append(item["image"])

        inputs = processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1000,  # 适度缩减 token 数以提升速度
            use_cache=True, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id,
        )

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # 保存助手回复到历史
    history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    
    return response

# -----------------------------------------------------------------------------
# 3. 运行入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Qwen3-VL RAG 推理系统")
    parser.add_argument("--chat", action="store_true", help="启动多轮对话模式")
    parser.add_argument("--image", type=str, default="/home/user/wmw_vs/123.jpg", help="输入图片路径")
    args = parser.parse_args()

    # 加载图片
    if os.path.exists(args.image):
        image = Image.open(args.image).convert("RGB")
    else:
        print(f"[!] 警告：找不到图片 {args.image}")
        image = None

    chat_history = []

    if args.chat:
        # --- 多轮对话交互 ---
        print("\n" + "="*50)
        print("[*] 多轮对话模式启动 (输入 'exit' 退出, 'clear' 重置)")
        print("="*50)
        while True:
            try:
                user_input = input("\n用户 >> ").strip()
            except (EOFError, KeyboardInterrupt): break

            if user_input.lower() == 'exit': break
            if user_input.lower() == 'clear': 
                chat_history = []
                print("--- 对话记录已清空 ---")
                continue
            if not user_input: continue

            start_t = time.time()
            res = rag_generate(user_input, image, chat_history)
            print(f"机器人 >> {res}\n(耗时: {time.time()-start_t:.2f}s)")
    else:
        # --- 单轮任务执行 ---
        task = "将图中的方块按照红黄蓝绿的颜色顺序，从左到右进行排列，红色方块放在初始位置（五角心所在位置）"
        print(f"\n[*] 执行单轮任务指令: {task}")
        
        start_t = time.time()
        res = rag_generate(task, image, chat_history)
        print(f"[*] 规划策略: \n{res}")
        print(f"[*] 总耗时: {time.time()-start_t:.2f}s")

    # 释放显存
    torch.cuda.empty_cache()