import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# 导入 SAM2 相关组件
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ================= 配置区域 =================
TRANSFER_JSON_PATH = "/home/user/wmw_vs/shared_coords.json"
checkpoint = "/home/user/Desktop/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"

# 绝对路径：指向包含 'sam2.1' 文件夹的那个 configs 目录
CONFIG_DIR = "/home/user/Desktop/Grounded-SAM-2/sam2/configs"
# 逻辑路径：相对于 CONFIG_DIR
CONFIG_NAME = "sam2.1/sam2.1_hiera_l" 

OUTPUT_MASK_PATH = "/home/user/wmw_vs/final_segmentation_result.png"
# ===========================================

def run_segmentation():
    # 1. 加载数据
    if not os.path.exists(TRANSFER_JSON_PATH):
        print(f"错误: 找不到共享坐标文件 {TRANSFER_JSON_PATH}")
        return
    with open(TRANSFER_JSON_PATH, "r") as f:
        data = json.load(f)
    img_path, point_coords, point_labels, input_box = data["image_path"], np.array(data["point_coords"]), np.array(data["point_labels"]), np.array(data["bbox"])

    # 2. 初始化 SAM 2 模型 (强制手动指定配置路径)
    print("正在加载 SAM 2 模型权重...")
    
    # 清理 Hydra 状态
    GlobalHydra.instance().clear()
    
    # 使用 hydra 的初始化器手动指定目录
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        # 虽然我们拿到了 cfg，但为了保持 build_sam2 的内部逻辑（它处理了模型缩放等复杂映射）
        # 我们直接把 CONFIG_NAME 传给它，因为我们已经在 initialize_config_dir 中对齐了搜索路径
        # 此时 build_sam2 内部的 compose 就能找到这个文件了
        predictor = SAM2ImagePredictor(build_sam2(CONFIG_NAME, checkpoint))

    # 3. 加载图像
    image_pil = Image.open(img_path).convert("RGB")
    image_np = np.array(image_pil)

    # 4. 执行推理
    print("开始像素级分割推理(点+框联合提示)...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_np)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords, 
            point_labels=point_labels, 
            box=input_box,
            multimask_output=False   # 当有点+框时，通常结果非常明确，不需要多掩码
        )

    # 5. 可视化并保存
    best_mask = masks[np.argmax(scores)].astype(bool)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_pil)
    
    mask_overlay = np.zeros((*best_mask.shape, 4))
    mask_overlay[best_mask, :3] = [0, 1, 0] # 绿色
    mask_overlay[best_mask, 3] = 0.4       # 透明度
    
    plt.imshow(mask_overlay)
    plt.scatter(point_coords[:, 0], point_coords[:, 1], color='blue', marker='*', s=200)
    plt.axis('off')
    plt.savefig(OUTPUT_MASK_PATH, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 保存二值掩码图
    mask_img = Image.fromarray((best_mask * 255).astype(np.uint8))
    mask_img.save("/home/user/wmw_vs/pure_mask.png")
    
    print(f"✅ 分割成功！结果已保存至: {OUTPUT_MASK_PATH}")

if __name__ == "__main__":
    run_segmentation()