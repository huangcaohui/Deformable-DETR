import torch
import datasets.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from util.box_ops import box_cxcywh_to_xyxy

# 创建目标检测模型，这里使用 Faster R-CNN 作为示例
model = torch.load("./exps/r50_deformable_detr/checkpoint0094.pth")['model']  # 加载训练好的模型权重
model.to('cuda')  # 将模型移动到 CUDA 上以加速推理

# 加载模型权重
model.eval()  # 将模型设置为评估模式

# 读取测试图像

test_image_path = './data/odo_annotated_coco/test/odo (1).jpg'  # 替换为您的测试图像路径
test_image = Image.open(test_image_path).convert("RGB")

# 进行图像转换
test_image, _ = T.RandomResize([800], max_size=1333)(test_image)
img, _ = T.ToTensor()(test_image)
img, _ = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
img = img.unsqueeze(0).to('cuda')  # 添加批次维度
h, w = img.shape[-2:]

# 使用模型进行目标检测
with torch.no_grad():
    predictions = model(img)

# 提取预测的目标框、类别
boxes = torch.squeeze(predictions['pred_boxes'])
labels = torch.squeeze(predictions['pred_logits'])

# 设置阈值以过滤低置信度的目标
threshold = 0.2
filtered_indices = labels > threshold
labels = labels[filtered_indices]
boxes = boxes[filtered_indices]

# 创建可绘制对象
draw = ImageDraw.Draw(test_image)
font = ImageFont.truetype('/usr/share/fonts/smc/Meera.ttf', size=40)

# 在图像上绘制目标框和标签
for box, label in zip(boxes, labels):
    box = box.cpu() * torch.tensor([w, h, w, h], dtype=torch.float32)
    box = box_cxcywh_to_xyxy(box).numpy()
    score = label.cpu().numpy()
    color = (0, 255, 0)  # 设置目标框颜色为红色 (R, G, B)
    draw.rectangle(box, outline=color, width=3)
    draw.text((box[0], box[1] - 35), f'Score: {score:.2f}', fill=color, font=font)

# 可视化结果
plt.imshow(test_image)
plt.axis('off')
plt.show()
