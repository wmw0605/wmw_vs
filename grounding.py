from PIL import Image, ImageDraw

# 1. 读取 1280x720 原图
image = Image.open("color1.png")
orig_width, orig_height = image.size  # (1280, 720)

# 2. 从 Qwen3-VL 获取 0-1000 坐标
# {"bbox_2d": [569, 152, 657, 550], "point_2d": [608, 327]}
qwen3_bbox = [365, 248, 440, 477]  # 矩形框坐标
qwen3_point = [608, 327]  # 关键点坐标

# 3. 转换为像素坐标
x1 = int(round(qwen3_bbox[0] / 1000 * orig_width))
y1 = int(round(qwen3_bbox[1] / 1000 * orig_height))
x2 = int(round(qwen3_bbox[2] / 1000 * orig_width))
y2 = int(round(qwen3_bbox[3] / 1000 * orig_height))

# 新增：转换为像素坐标 (Point)
pk_x = int(round(qwen3_point[0] / 1000 * orig_width))
pk_y = int(round(qwen3_point[1] / 1000 * orig_height))

# 4. 确保坐标在范围内
x1 = max(0, min(x1, orig_width-1))
y1 = max(0, min(y1, orig_height-1))
x2 = max(0, min(x2, orig_width-1))
y2 = max(0, min(y2, orig_height-1))
pk_x, pk_y = max(0, min(pk_x, orig_width-1)), max(0, min(pk_y, orig_height-1))

print(f"Bounding Box in Pixels: ({x1}, {y1}), ({x2}, {y2})")

# 5. 绘制边界框
draw = ImageDraw.Draw(image)
draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
# 绘制关键点 (半径为5的蓝色实心圆)
r = 10
draw.ellipse([pk_x - r, pk_y - r, pk_x + r, pk_y + r], fill="green")

# 6. 保存结果
image.save("annotated_result.png")