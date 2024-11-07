import torch
from ultralytics import YOLO

# 加载训练好的权重
model = YOLO('runs/detect/yolov8n-det-grape-+-dcn-wiou/weights/best.pt')

# 创建一个随机输入张量，假设输入图像大小为640x640，3个通道
input_tensor = torch.randn(1, 3, 640, 640)

# 计算参数量
def get_parameter_count(model):
    return sum(p.numel() for p in model.model.parameters())  # 修改为model.model

# 计算FLOPS
def get_flops(model, input_tensor):
    flops = 0
    hooks = []

    def count_flops(module, input, output):
        nonlocal flops
        flops += 2 * input[0].numel() * output.numel() / output.shape[0]  # 计算FLOPS

    for layer in model.model.modules():  # 使用model.model来访问具体的模型
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(count_flops))

    with torch.no_grad():
        model.model(input_tensor)  # 使用model.model进行推理

    for hook in hooks:
        hook.remove()

    return flops

# 获取参数量和FLOPS
parameter_count = get_parameter_count(model)
flops = get_flops(model, input_tensor)

print(f'参数量: {parameter_count}')
print(f'FLOPS: {flops}')
