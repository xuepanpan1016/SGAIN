#!/bin/bash

# =================================================================
# LPIPS (Learned Perceptual Image Patch Similarity) 评估脚本
# -----------------------------------------------------------------
# 用途:
#   计算“生成图像”和“源图像”之间的感知相似度。
#   分数越低，代表人眼看来两张图片越相似。
#
# 注意:
#   此脚本需要先安装 lpips 包: `pip install lpips`
#   它会自动处理源图像尺寸不一和格式多样的问题。
#
# 运行方式:
#   cd /root/autodl-tmp/stable-diffusion
#   bash ./tkeva/eval_lpips.sh
# =================================================================

set -e # 如果任何命令失败，立即退出脚本

# --- 配置 ---
# 生成图像的路径
GENERATED_IMG_DIR=""
# 源验证集图像的路径
ORIGINAL_IMG_DIR="/root/autodl-tmp/DiT/dataset/"

echo "================================================="
echo "开始 LPIPS 评估..."
echo "================================================="

echo
echo "--- 核心评估逻辑（Python） ---"

# 使用一个Python脚本来完成所有操作
python -c "
import os
import glob
import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor

try:
    import lpips
except ImportError:
    print('错误: 未找到 lpips 包。')
    print('请先运行: pip install lpips')
    sys.exit(1)

# --- 配置 ---
generated_dir = '${GENERATED_IMG_DIR}'
original_dir = '${ORIGINAL_IMG_DIR}'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
target_size = (512, 512)

# --- 加载 LPIPS 模型 ---
# net='alex' 是标准用法。 spatial=False 确保返回一个单一的分数而不是一个特征图。
print(f'正在加载 LPIPS (AlexNet) 模型到 {device}...')
loss_fn = lpips.LPIPS(net='alex', spatial=False).to(device)
print('模型加载成功。')

def load_and_preprocess_image(image_path, target_size):
    '''加载图片，将其转换为RGB，缩放到目标尺寸，并归一化到[-1, 1]范围'''
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    # 转换成Tensor，此时范围在[0, 1]
    tensor = ToTensor()(img)
    # 归一化到[-1, 1]
    tensor = (tensor * 2) - 1
    # 添加 batch 维度并发送到设备
    return tensor.unsqueeze(0).to(device)

# --- 查找配对并计分 ---
all_distances = []
generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.png')]
print(f'在 {generated_dir} 中找到 {len(generated_files)} 张生成的图像，开始查找配对并计算距离...')

for i, gen_filename in enumerate(generated_files):
    prefix = os.path.splitext(gen_filename)[0]
    
    # 使用 glob 模糊匹配不同后缀的源文件
    search_pattern = os.path.join(original_dir, f'{prefix}.*')
    original_files = glob.glob(search_pattern)
    
    if not original_files:
        print(f'警告: 跳过 {gen_filename}，未在 {original_dir} 中找到任何对应的源文件。')
        continue
    
    # 即使有多个匹配（如.jpg和.jpeg），也只取第一个
    original_filepath = original_files[0]
    generated_filepath = os.path.join(generated_dir, gen_filename)
    
    try:
        # 加载并预处理两张图片
        img_gen = load_and_preprocess_image(generated_filepath, target_size)
        img_orig = load_and_preprocess_image(original_filepath, target_size)
        
        with torch.no_grad():
            dist = loss_fn.forward(img_gen, img_orig)
        
        distance_value = dist.item()
        all_distances.append(distance_value)
        print(f'  ({i+1}/{len(generated_files)}) 对比: {gen_filename} vs {os.path.basename(original_filepath)}, LPIPS距离: {distance_value:.4f}')

    except Exception as e:
        print(f'错误: 处理文件 {gen_filename} 或 {os.path.basename(original_filepath)} 时发生错误: {e}')


# --- 计算平均距离 ---
if not all_distances:
    print('错误: 未能计算任何距离。请检查路径和文件。')
    sys.exit(1)

average_distance = sum(all_distances) / len(all_distances)

print()
print('----------------------------------------------------')
print(f'所有图像对的平均 LPIPS 距离: {average_distance:.4f}')
print('(分数越低越好)')
print('----------------------------------------------------')
"

echo
echo "================================================="
echo "LPIPS 评估完成！"
echo "=================================================" 