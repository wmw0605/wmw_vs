# import os
# import torch
# from PIL import Image
# from unsloth import FastLanguageModel

# # ================================
# # 1. 环境与路径配置
# # ================================
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# # 替换为你实际的 LoRA 路径
# LORA_PATH = "/home/user/wmw_vs/models/lora_model"
# IMAGE_PATH = "/home/user/wmw_vs/color1.png"

# # 4.示例任务（捡起水果刀递给我）：
# # {"task":"Pick up the fruit knife","response":"The task requires the robot to locate, grasp, and lift the fruit knife safely.","steps":[{"step":1,"action":"detect_object","object":{"name":"knife","parts":[{"part":"handle","color":"black"},{"part":"blade","color":"silver"}]}},{"step":2,"action":"grasp_object","grasp_target":"knife handle"},{"step":3,"action":"handover_object"}]}

# # 你的机器人行为准则 (System Prompt)
# SYSTEM_PROMPT = """1.你是一个处于家庭场景中的机器人，当前输入的图片就是你所处场景，你需要模拟与用户之间的日常对话。
# 2.在对话的结束你会询问是否执行任务，用户发出确认执行任务的意图时，必须仅能生成特定格式的动作排版。
# 3.四种基本动作对应的json格式和示例分别如下：
# 3.1 detect_object: {"step":1,"action":"detect_object","object":{"name":"knife","parts":[{"part":"handle","color":"black"},{"part":"blade","color":"silver"}]}}
# 3.2 grasp_object: {"step":1,"action":"grasp_object","grasp_target":"knife handle"}
# 3.3 handover_object: {"step":1,"action":"handover_object"}
# 3.4 place_object: {"step":1,"action":"place_object","place_position":"plate"}
# 4.示例任务（捡起水果刀递给我）：
# {"task":"Pick up the fruit knife","response":"The task requires the robot to locate, grasp, and lift the fruit knife safely.","steps":[{"step":1,"action":"detect_object","object":{"name":"knife","parts":[{"part":"handle","color":"black"},{"part":"blade","color":"silver"}]}},{"step":2,"action":"grasp_object","grasp_target":"knife handle"},{"step":3,"action":"handover_object"}]}

# """

# # ================================
# # 2. 加载模型
# # ================================
# print(f"正在加载模型: {LORA_PATH}...")
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = LORA_PATH,
#     max_seq_length = 2048,
#     load_in_4bit = True,
# )
# FastLanguageModel.for_inference(model)

# # 加载图片
# if os.path.exists(IMAGE_PATH):
#     image = Image.open(IMAGE_PATH).convert("RGB")
# else:
#     print(f"警告：找不到图片 {IMAGE_PATH}")
#     image = None

# # ================================
# # 3. 多轮对话循环
# # ================================
# messages = []

# print("\n" + "="*30)
# print("机器人系统已启动。输入 'exit' 退出，'clear' 重置。")
# print("="*30 + "\n")

# while True:
#     user_input = input("用户 >> ").strip()
    
#     if user_input.lower() == 'exit':
#         break
#     if user_input.lower() == 'clear':
#         messages = []
#         print("--- 对话已重置 ---")
#         continue

#     # 构建当前消息
#     if not messages:
#         # 第一轮：包含图片、系统指令和用户输入
#         content = [
#             {"type": "image", "image": image},
#             {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n当前任务：{user_input}"}
#         ]
#     else:
#         # 后续轮：仅包含文本
#         content = [
#             {"type": "text", "text": user_input}
#         ]
    
#     messages.append({"role": "user", "content": content})

#     # 准备推理输入
#     inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt = True,
#         tokenize = True,
#         return_dict = True,
#         return_tensors = "pt",
#     ).to("cuda")

#     # 执行生成
#     print("机器人思考中...", end="\r")
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens = 512,
#         use_cache = True,
#         pad_token_id = tokenizer.eos_token_id
#     )

#     # 提取回复内容（切掉 input 的部分）
#     input_len = inputs.input_ids.shape[1]
#     new_tokens = outputs[0][input_len:]
#     response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

#     print(f"机器人 >> {response_text}")

#     # 将机器人回复存入历史
#     messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

#     # 显存保护：如果对话轮数过多，删除最早的记录（保留第一轮的图片引用）
#     if len(messages) > 11: # 约 5 轮完整对话
#         messages = [messages[0]] + messages[-10:]


# #--------------------------------------------------- 微调前后模型对比（未微调）----------------------------------------------------
# #--------------------------------------------------- 微调前后模型对比（未微调）----------------------------------------------------
# #--------------------------------------------------- 微调前后模型对比（未微调）----------------------------------------------------
# #--------------------------------------------------- 微调前后模型对比（未微调）----------------------------------------------------
# #--------------------------------------------------- 微调前后模型对比（未微调）----------------------------------------------------




import os
import torch
from PIL import Image
from unsloth import FastLanguageModel

# ================================
# 1. 环境与路径配置
# ================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 【关键修改】：这里改为你的基础模型路径（例如 Qwen2-VL-7B-Instruct 的本地目录或 ID）
# 不要指向 /models/lora_model，而是指向你微调前下载的那个原始模型目录
BASE_MODEL_PATH = "/home/user/qwen3_vl_8b" # 或者是你的本地原始路径
IMAGE_PATH = "/home/user/wmw_vs/color1.png"

# 你的机器人行为准则 (System Prompt) - 保持一致以进行公平对比
SYSTEM_PROMPT = """1.你是一个处于家庭场景中的机器人，当前输入的图片就是你所处场景，你需要模拟与用户之间的日常对话。
2.在对话的结束你会询问是否执行任务，用户发出确认执行任务的意图时，必须仅能生成特定格式的动作排版。
3.四种基本动作对应的json格式和示例分别如下：
3.1 detect_object: {"step":1,"action":"detect_object","object":{"name":"knife","parts":[{"part":"handle","color":"black"},{"part":"blade","color":"silver"}]}}
3.2 grasp_object: {"step":1,"action":"grasp_object","grasp_target":"knife handle"}
3.3 handover_object: {"step":1,"action":"handover_object"}
3.4 place_object: {"step":1,"action":"place_object","place_position":"plate"}
4.示例任务（捡起水果刀递给我）：
{"task":"Pick up the fruit knife","response":"The task requires the robot to locate, grasp, and lift the fruit knife safely.","steps":[{"step":1,"action":"detect_object","object":{"name":"knife","parts":[{"part":"handle","color":"black"},{"part":"blade","color":"silver"}]}},{"step":2,"action":"grasp_object","grasp_target":"knife handle"},{"step":3,"action":"handover_object"}]}"""

# ================================
# 2. 加载原始基础模型
# ================================
print(f"正在加载【原始基础模型】: {BASE_MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL_PATH, # 指向基础模型
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 加载图片
if os.path.exists(IMAGE_PATH):
    image = Image.open(IMAGE_PATH).convert("RGB")
else:
    print(f"警告：找不到图片 {IMAGE_PATH}")
    image = None

# ================================
# 3. 多轮对话循环
# ================================
messages = []

print("\n" + "="*40)
print("【原始模型测试模式】已启动。")
print("注意：原始模型可能无法完美遵循你的 JSON 格式要求。")
print("="*40 + "\n")

while True:
    user_input = input("用户 >> ").strip()
    
    if user_input.lower() == 'exit':
        break
    if user_input.lower() == 'clear':
        messages = []
        print("--- 对话已重置 ---")
        continue

    # 构建当前消息
    if not messages:
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n当前任务：{user_input}"}
        ]
    else:
        content = [
            {"type": "text", "text": user_input}
        ]
    
    messages.append({"role": "user", "content": content})

    # 准备推理输入
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")

    # 执行生成
    print("原始模型思考中...", end="\r")
    outputs = model.generate(
        **inputs,
        max_new_tokens = 512,
        use_cache = True,
        pad_token_id = tokenizer.eos_token_id
    )

    # 提取回复内容
    input_len = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"机器人(Base) >> {response_text}")

    # 将机器人回复存入历史
    messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

    # 显存保护
    if len(messages) > 11:
        messages = [messages[0]] + messages[-10:]