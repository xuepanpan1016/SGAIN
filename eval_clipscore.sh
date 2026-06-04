#!/bin/bash

# =================================================================
# CLIPScore 评估脚本
# -----------------------------------------------------------------
# 用途:
#   计算生成图像与对应文本提示词之间的CLIP相似度分数。
#   分数越高，代表图文匹配度越好。
#
# 运行方式:
#   cd /root/autodl-tmp/stable-diffusion
#   bash ./tkeva/eval_clipscore.sh
# =================================================================

set -e # 如果任何命令失败，立即退出脚本

# --- 配置 ---
# 生成图像的存放路径
GENERATED_IMG_DIR=""
# 文本提示词的存放路径
TEXT_PROMPT_DIR=""
# CLIPScore 源代码的路径
CLIPSCORCE_SRC_DIR="/root/autodl-tmp/eva/clipscore-main"
# 此脚本的输出/工作目录
WORK_DIR="/root/autodl-tmp/DiT/dataset"
# 自动生成的中间JSON文件的路径
CANDIDATES_JSON_PATH="${WORK_DIR}/clipscore_candidates1.json"

echo "================================================="
echo "开始 CLIPScore 评估..."
echo "================================================="

echo
echo "--- 步骤 1/2: 准备图文对应的 JSON 文件 ---"

# 使用Python动态创建 clipscore 所需的 candidates.json 文件
# 该文件格式为 {"图片名1": "描述1", "图片名2": "描述2", ...}
python -c "
import os
import json

generated_dir = '${GENERATED_IMG_DIR}'
text_dir = '${TEXT_PROMPT_DIR}'
output_json_path = '${CANDIDATES_JSON_PATH}'

candidates = {}
# 只读取 .png 格式的生成图片
image_files = [f for f in os.listdir(generated_dir) if f.endswith('.png')]
print(f'在 {generated_dir} 中找到 {len(image_files)} 张生成的图像。')

for img_filename in image_files:
    # 从 'image_name.png' 中提取 'image_name'
    prefix = os.path.splitext(img_filename)[0]
    text_filepath = os.path.join(text_dir, f'{prefix}.txt')
    
    if os.path.exists(text_filepath):
        with open(text_filepath, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        # JSON的key是图片名（不带后缀），value是描述
        candidates[prefix] = caption
    else:
        print(f'警告: 未找到 {img_filename} 对应的文本文件 {prefix}.txt')

if not candidates:
    print('错误: 未能生成任何候选数据，请检查路径和文件。脚本退出。')
    exit(1)

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(candidates, f)

print(f'成功创建 JOSN 文件于: {output_json_path}')
"

echo
echo "--- 步骤 2/2: 运行 CLIPScore 核心脚本 ---"

# 检查 clipscore.py 是否存在
if [ ! -f "${CLIPSCORCE_SRC_DIR}/clipscore.py" ]; then
    echo "错误: 在 ${CLIPSCORCE_SRC_DIR} 中未找到 clipscore.py"
    exit 1
fi

# 运行评估。它需要两个参数:
# 1. 我们刚刚生成的 JSON 文件
# 2. 包含生成图片的目录
# 注意: clipscore 脚本会自动处理图片名和后缀
python ${CLIPSCORCE_SRC_DIR}/clipscore.py ${CANDIDATES_JSON_PATH} ${GENERATED_IMG_DIR}

echo
echo "================================================="
echo "CLIPScore 评估完成！"
echo "=================================================" 