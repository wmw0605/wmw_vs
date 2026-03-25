from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re
import json
import os
import subprocess
from PIL import Image, ImageDraw

# ================= 配置区域 =================
# 环境 B 的 Python 解释器路径 
SAM2_PYTHON_PATH = "/home/user/anaconda3/envs/sam2/bin/python"
# 环境 B 的推理脚本路径
SAM2_SCRIPT_PATH = "/home/user/Desktop/Grounded-SAM-2/sam2/test_run.py"
# 用于跨环境传递坐标的中间文件
TRANSFER_JSON_PATH = "/home/user/wmw_vs/shared_coords.json"
# 模型路径
model_path = "/home/user/qwen3_vl_8b"
# 输入图片路径
img_path = "/home/user/wmw_vs/123.jpg"
# ======================================/home/user/Desktop=====


# 1. 加载模型
print("正在加载 Qwen3-VL 模型...")
model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# 2. 准备多模态输入
system_prompt = (
    "你是一个精准的机器人视觉感知系统。你的任务是执行视觉定位（Grounding）。\n"
    "要求：\n"
    "1. 识别任务中涉及的所有目标对象的坐标，并以 JSON 格式输出每一个定位到的检测框 [{'bbox_2d': [ymin, xmin, ymax, xmax]}] \n"
    "2. 还需要输出目标对象的关键点（目标点），并以 JSON 格式输出 [{'point_2d': [x, y]}]\n"
    "3. 还需要说明每一个定位框的对象名称label,label对应的值必须是能够进行区分的\n"
    "4. **数据关联**：每一个检测对象必须包含 label、bbox_2d 和 point_2d，以确保 ID 对应关系清晰。\n\n"
        "### 输出格式规范 ###\n"
        "严格以 JSON 数组格式输出，不要包含任何额外的推理文字。格式示例如下：\n"
        "[\n"
        "  {\n"
        "    \"label\": \"cube_red\",\n"
        "    \"bbox_2d\": [ymin, xmin, ymax, xmax],\n"
        "    \"point_2d\": [x, y]\n"
        "  },\n"
        "  {\n"
        "    \"label\": \"bottle_cap\",\n"
        "    \"bbox_2d\": [ymin, xmin, ymax, xmax],\n"
        "    \"point_2d\": [x, y]\n"
        "  }\n"
        "]"
)

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": "将图中的方块按照红黄蓝绿的颜色顺序，从左到右进行排列，五角星的位置是初始位置，用于放置第一个方块"},
        ]
    }
]

# 3. 数据处理与生成
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

print("模型正在推理中...")
generated_ids = model.generate(**inputs, max_new_tokens=512)
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print("\n模型回复内容:")
print(output_text[0])

# 4. 提取回复中的 JSON (增强版解析)
full_response = output_text[0]
assistant_reply = full_response.split("assistant")[-1].strip()

json_data = None
try:
    json_pattern = r"```json\s*(.*?)\s*```"
    json_match = re.search(json_pattern, assistant_reply, re.DOTALL)
    if json_match:
        json_data = json.loads(json_match.group(1))
    else:
        json_data = json.loads(assistant_reply)
except Exception as e:
    print(f"❌ JSON 解析失败: {e}")

if json_data:
    try:
        # 5. 读取原图并应用你的纯净映射逻辑
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        draw = ImageDraw.Draw(image)

        all_targets = []
        for item in json_data:
            label = item.get("label", "obj")
            qwen3_bbox = item.get("bbox_2d")
            qwen3_point = item.get("point_2d")

            # --- 纯净映射逻辑：去掉所有裁剪限制 ---
            pk_x, pk_y = 0, 0
            x1, y1, x2, y2 = 0, 0, 0, 0
            
            if qwen3_bbox:
                # 严格按照你的 round(val / 1000 * size) 逻辑
                x1 = int(round(qwen3_bbox[0] / 1000 * orig_width))
                y1 = int(round(qwen3_bbox[1] / 1000 * orig_height))
                x2 = int(round(qwen3_bbox[2] / 1000 * orig_width))
                y2 = int(round(qwen3_bbox[3] / 1000 * orig_height))
                
                # 绘图
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=10)
                draw.text((x1, y1 - 20), label, fill="red")

            if qwen3_point:
                pk_x = int(round(qwen3_point[0] / 1000 * orig_width))
                pk_y = int(round(qwen3_point[1] / 1000 * orig_height))
                
                r = 30
                draw.ellipse([pk_x - r, pk_y - r, pk_x + r, pk_y + r], fill="green")

            all_targets.append({
                "label": label,
                "bbox": [x1, y1, x2, y2] if qwen3_bbox else None,
                "point": [pk_x, pk_y] if qwen3_point else None
            })

        image.save("vlm_detection_result.png")
        print(f"✅ VLM 检测图已保存（未裁剪原始坐标）。")

        # 6. 保存数据用于跨环境通信 (纯净版：仅保存路径和全目标列表)
        if all_targets:
            transfer_data = {
                "image_path": img_path,
                "targets": all_targets 
            }
            
            with open(TRANSFER_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(transfer_data, f, ensure_ascii=False, indent=4)
            
            print(f"✅ 数据已写入: {TRANSFER_JSON_PATH} (仅包含 targets 列表)")

        # 7. 释放 VLM 显存，准备启动 SAM2
        print("正在清理显存...")
        del model
        torch.cuda.empty_cache()

        # 8. 跨环境调用 SAM2 环境中的代码
        print(f"\n🚀 正在跨环境调用 SAM2 (环境: {os.path.basename(os.path.dirname(os.path.dirname(SAM2_PYTHON_PATH)))})")

        result = subprocess.run(
            [SAM2_PYTHON_PATH, SAM2_SCRIPT_PATH],
            cwd="/home/user/Desktop/Grounded-SAM-2", 
            capture_output=True,
            text=True
)

        if result.returncode == 0:
            print("✅ SAM2 分割任务顺利完成！")
            print("SAM2 输出内容:\n", result.stdout)
        else:
            print("❌ SAM2 运行出错:")
            print(result.stderr)

    except Exception as e:
        print(f"❌ 解析或执行过程中出错: {e}")
else:
    print("❌ 未能从模型回复中找到 JSON 坐标。")


