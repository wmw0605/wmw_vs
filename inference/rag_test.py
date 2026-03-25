import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForImageTextToText

# 初始化组件
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = AutoTokenizer.from_pretrained("/home/user/qwen3_vl_8b")
model = AutoModelForImageTextToText.from_pretrained("/home/user/qwen3_vl_8b")

# 构建小型知识库（实际可用数据库或文档系统替代）
documents = [
    "机械臂动作技能1-平面抓取（适合机械臂从上至下执行物体的抓取）",
    "机械臂动作技能2-六自由度抓取（适合机械臂从不同的方向抓取物体）",
    "机械臂动作技能3-推动（推动桌面上的物体）",
    "机械臂动作技能4-放置物体（放置抓起来的物体）",
    "小米14采用骁龙8 Gen 3，主摄为光影猎人9000传感器。"
]
doc_embeddings = embedding_model.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# RAG主函数
def rag_generate(question: str):
    # 1. 检索
    query_vec = embedding_model.encode([question])
    _, indices = index.search(query_vec, k=1)
    retrieved_doc = documents[indices[0][0]]

    # 2. 构造增强提示embedding_model
    prompt = f"""
    根据以下信息回答问题：

    资料：{retrieved_doc}

    问题：{question}

    回答：
    """

    # 3. 生成
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=1000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# 测试一下
print(rag_generate("现在你的角色是机械臂，需要满足我的需求。现在我有点口渴，你会执行什么动作？（目前的桌面上有剪刀，鼠标，电脑，水果篮和马克杯）"))
