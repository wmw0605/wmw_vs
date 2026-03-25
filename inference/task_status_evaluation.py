import os
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ================= 配置部分 =================
BASE_MODEL_PATH = "/home/user/qwen3_vl_8b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# 1. 初始化视觉裁判模型
# -----------------------------------------------------------------------------
print("[*] 正在加载视觉裁判 (Qwen-3-VL)...")
try:
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("[*] 视觉裁判加载成功！")
except Exception as e:
    print(f"[!] 模型加载失败: {e}")
    exit()

# -----------------------------------------------------------------------------
# 2. 状态评估核心函数
# -----------------------------------------------------------------------------
def evaluate_sub_step(sequence_dir: str, task_semantic: str):
    """
    读取指定路径下的所有图片进行闭环评估。
    """
    if not os.path.exists(sequence_dir):
        return f"错误：路径 {sequence_dir} 不存在。"

    # 获取所有 png 文件并按数字顺序排序（兼容 1.png 和 0001.png 两种命名）
    all_files = sorted(
        [f for f in os.listdir(sequence_dir) if f.endswith('.png')],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )

    if not all_files:
        return "错误：文件夹中未找到任何图片。"

    # 打印排序结果供调试
    print(f"[*] 文件排序结果: {all_files[:5]}...{all_files[-3:] if len(all_files) > 5 else ''}")

    images = []

    print(f"[*] 正在加载 {len(all_files)} 帧图片...")

    for f in all_files:
        img_path = os.path.join(sequence_dir, f)
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    print(f"[*] 共加载 {len(images)} 张图片，将全部输入 VLM")

    # --- B. 构造视觉裁判 Prompt (同步优化) ---
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个精通机器人控制的视觉裁判。提供的图片序列是执行过程中的【离散采样快照】，并非连续视频流。"
                "每张图片左上角都有白色数字序号（如 0000, 0001...），代表时间先后。"
                "由于采样频率限制，某些瞬间（如物体滑落的瞬间）可能未被直接捕获，你必须通过前后帧的状态差异进行逻辑推断。"
            )
        },
        {
            "role": "user",
            "content": [
                # 注入标注了序号的所有图片
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": (
                    f"### 当前子步骤语义信息 ###\n{task_semantic}\n\n"
                    "### 评估要求 ###\n"
                    "请结合图片左上角的帧序号和当前子步骤的任务语义，分析整个动作过程，并给出评估报告：\n"
                    "1. 【过程关键帧】：请列举出运行状态中的关键帧，并给出帧序号。比如机械臂开始接触物体的帧序号，以及物体被放置后的帧序号等等。\n"
                    "2. 【执行状态】：判断是 [执行成功] 还是 [执行异常]。\n"
                    "3. 【原因分析】：请引用具体帧序号作为依据。\n"
                    "4. 【决策建议】：子步骤执行成功时，输出’子步骤执行成功’，如果子步骤失败则输出’子步骤执行失败’。\n"
                    "5.  我的建议是你首先需要根据图中的帧序号理解每一帧。来分析动作执行的时间顺序，找出关键帧，然后结合每个关键帧的视觉信息来判断执行状态和原因分析。请务必结合帧序号来进行分析，不要忽略时间维度的信息。\n"
                    "6.  还需要简要描述下整个流程"
                )}
            ]
        }
    ]

    # --- C. VLM 推理阶段 ---
    with torch.inference_mode():
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # 调试：确认传入 processor 的图片数量
        print(f"[*] 传入 processor 的图片数量: {len(images)}")

        inputs = processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response

# -----------------------------------------------------------------------------
# 3. 运行接口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SEQUENCE_PATH = "/home/user/mujoco_fr3/captures/global_sequence/cycle_000"
    SEMANTIC_INFO = "当前子步骤：将红色方块抓取后，放置在场景中的黄色五角标记的所在位置。"

    print("\n" + "="*55)
    print(f"[*] 启动带视觉锚点的状态验证...")

    start_time = time.time()
    result = evaluate_sub_step(SEQUENCE_PATH, SEMANTIC_INFO)
    end_time = time.time()

    print("-" * 55)
    print(f"【视觉裁判结论】:\n{result}")
    print(f"[*] 评估耗时: {end_time - start_time:.2f} 秒")
    print("="*55 + "\n")

    torch.cuda.empty_cache()