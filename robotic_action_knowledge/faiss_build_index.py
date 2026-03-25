import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_skill_index(json_path, model_path, index_save_path):
    # 1. 加载 JSON 技能库
    with open(json_path, 'r', encoding='utf-8') as f:
        skills = json.load(f)
    
    # 2. 提取所有技能的描述 (D_i)
    # 建议格式：技能名称 + 描述，增强匹配准确度
    descriptions = [f"{s['name']}: {s['description']}" for s in skills]
    
    # 3. 加载 Embedding 模型 (使用本地的路径)
    embed_model = SentenceTransformer(model_path, device="cuda")
    
    # 4. 生成向量并转换为 float32 (Faiss 必须要求)
    print("正在计算技能描述向量...")
    embeddings = embed_model.encode(descriptions).astype('float32')
    faiss.normalize_L2(embeddings) # 关键步骤：L2 归一化
    
    # 5. 创建 Faiss 索引 (L2 距离)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # 6. 保存索引文件
    faiss.write_index(index, index_save_path)
    print(f"成功！Faiss 索引已保存至: {index_save_path}")

# 执行构建
if __name__ == "__main__":
    build_skill_index(
        json_path="robotic_action_knowledge/skills.json", 
        model_path="/home/user/models/paraphrase-multilingual-MiniLM-L12-v2",
        index_save_path="robotic_action_knowledge/skills.index"
    )