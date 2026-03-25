import pathlib
import json
import os


def get_dialog_list(root_path="dataset/Fine-tuningdataset"):
    dataset_list = []
    base_dir = pathlib.Path(root_path)
    
    if not base_dir.exists():
        print(f"路径 {root_path} 不存在")
        return []

    # 获取并按数字大小排序子目录
    subdirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )

    for subdir in subdirs:
        for json_file in subdir.glob("dialo*.json"):
            try:
                # 读取并解析 JSON
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 根据数据结构选择添加方式
                if isinstance(data, list):
                    dataset_list.extend(data)  # 如果文件里是列表，合并进去
                else:
                    dataset_list.append(data)   # 如果文件里是单个对象，追加进去
                    
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")

    return dataset_list






def load_valit_dataset(file_path="/home/user/wmw_vs/dataset/validate_dataset.json"):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 假设每行是一个 JSON 对象 (JSONL 格式)
        for line in f:
            if line.strip():
                result.append(json.loads(line))
    return result

