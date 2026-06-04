# 解决OMP冲突，必须放在最前面
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------- 最宽松模式配置 --------------------------
# 本地U2Net权重文件路径
U2NET_WEIGHT_PATH = "u2net.pth"

# 最宽松TSDE参数配置
TSDE_CONFIG = {
    "alpha": 0.4,  # 中心构图得分权重
    "beta": 0.3,  # 双边平衡得分权重
    "gamma": 0.3,  # 结构细节得分权重
    "lambda_": 1.0,  # 背景噪声惩罚系数（降至最低）
    "epsilon": 1e-8,  # 防止除零的极小值

    # 分割最宽松参数
    "seg_high_threshold": 0.1,  # 核心前景高阈值（大幅降低）
    "seg_low_threshold": 0.05,  # 边缘扩展低阈值（大幅降低）
    "center_weight": 1.5,  # 取消中心区域加权
    "min_area_ratio": 0.01,  # 最小前景面积比例（降至1%）
    "morph_kernel_size": 1,  # 取消形态学净化

    # 边缘最宽松参数
    "edge_threshold1": 10,  # Canny低阈值（大幅降低）
    "edge_threshold2": 100,  # Canny高阈值（大幅降低）
    "min_edge_length": 3  # 最小边缘长度（降至3像素）
}

# 自动检测设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"运行设备: {DEVICE}")


# -------------------------- U2Net官方原版模型定义（100%匹配权重） --------------------------
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)


# RSU-7
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


# RSU-6
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


# RSU-5
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


# RSU-4
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


# RSU-4F (无下采样)
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin


# 完整U2Net模型（与官方权重完全匹配）
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # side output
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6), d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)


# -------------------------- 加载本地U2Net权重 --------------------------
U2NET_MODEL = None


def load_u2net_model():
    """加载本地U2Net预训练权重，完全离线运行"""
    global U2NET_MODEL
    if U2NET_MODEL is None:
        # 检查权重文件是否存在
        if not os.path.exists(U2NET_WEIGHT_PATH):
            raise FileNotFoundError(
                f"U2Net权重文件不存在！请将下载好的u2net.pth放在以下路径：\n{os.path.abspath(U2NET_WEIGHT_PATH)}\n"
                "或修改代码开头的U2NET_WEIGHT_PATH变量为你的权重文件路径。"
            )

        print("🔄 正在加载本地U2Net权重...")
        # 实例化模型
        U2NET_MODEL = U2NET(3, 1).to(DEVICE)
        # 加载本地权重（严格匹配）
        U2NET_MODEL.load_state_dict(torch.load(U2NET_WEIGHT_PATH, map_location=DEVICE), strict=True)
        # 设置为评估模式
        U2NET_MODEL.eval()
        print("✅ U2Net模型加载成功（本地权重，严格匹配）")
    return U2NET_MODEL


# -------------------------- 最宽松版唐卡前景分割 --------------------------
def get_foreground_mask(img):
    """
    最宽松版唐卡前景分割：
    1. 取消中心区域加权，完整保留所有前景
    2. 极低阈值分割，最大限度保留边缘细节
    3. 极小面积过滤，保留所有与主尊相连的元素
    4. 取消形态学净化，完整保留原始轮廓
    5. 保留内部孔洞，不做强制填充
    """
    model = load_u2net_model()
    H, W = img.shape[:2]
    total_pixels = H * W

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((320, 320)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 模型推理获取原始显著性图
    with torch.no_grad():
        pred = model(input_tensor)[0]

    # 恢复原尺寸
    pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
    saliency_map = pred.squeeze().cpu().numpy()

    # -------------------------- 宽松化步骤1：取消中心加权 --------------------------
    weighted_saliency = saliency_map  # 直接使用原始显著性图，不做任何加权

    # -------------------------- 宽松化步骤2：极低阈值分割 --------------------------
    # 高阈值提取核心前景
    core_mask = (weighted_saliency > TSDE_CONFIG["seg_high_threshold"]).astype(np.uint8)

    # 低阈值提取边缘区域
    edge_mask = (weighted_saliency > TSDE_CONFIG["seg_low_threshold"]).astype(np.uint8)

    # 只保留与核心前景相连的边缘区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_mask)
    valid_labels = set()

    for i in range(1, num_labels):
        if np.any(core_mask[labels == i]):
            valid_labels.add(i)

    combined_mask = np.zeros_like(edge_mask)
    for label in valid_labels:
        combined_mask[labels == label] = 1

    # -------------------------- 宽松化步骤3：极小面积过滤 --------------------------
    min_area = total_pixels * TSDE_CONFIG["min_area_ratio"]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask)

    if num_labels > 1:
        # 按面积从大到小排序
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1] + 1

        # 只保留面积大于最小阈值的连通域
        final_mask = np.zeros_like(combined_mask)
        for idx in sorted_indices:
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                final_mask[labels == idx] = 1
            else:
                break
    else:
        final_mask = combined_mask

    # -------------------------- 宽松化步骤4：取消形态学净化 --------------------------
    # 核大小为1，腐蚀和膨胀无效果，相当于跳过这一步
    kernel = np.ones((TSDE_CONFIG["morph_kernel_size"], TSDE_CONFIG["morph_kernel_size"]), np.uint8)
    final_mask = cv2.erode(final_mask, kernel, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    # -------------------------- 宽松化步骤5：保留内部孔洞 --------------------------
    # 注释掉孔洞填充代码，保留原始结构
    # contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     cv2.drawContours(final_mask, [contour], -1, 1, thickness=cv2.FILLED)

    return final_mask.astype(np.float32)


# -------------------------- 最宽松版结构边缘提取 --------------------------
def get_edge_map(img, mask):
    """
    最宽松版结构边缘提取：
    1. 低阈值Canny，保留所有细节边缘
    2. 极短边缘过滤，只去掉孤立噪声点
    3. 边缘与分割掩码100%对齐
    """
    # 1. 提取前景区域的灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    foreground_gray = gray * mask

    # 2. 使用低阈值Canny提取所有细节边缘
    edges = cv2.Canny(
        foreground_gray.astype(np.uint8),
        TSDE_CONFIG["edge_threshold1"],
        TSDE_CONFIG["edge_threshold2"],
        apertureSize=3,
        L2gradient=True
    )

    # 3. 只过滤极短的孤立噪声
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_edges = np.zeros_like(edges)
    for contour in contours:
        if cv2.arcLength(contour, closed=False) >= TSDE_CONFIG["min_edge_length"]:
            cv2.drawContours(filtered_edges, [contour], -1, 255, thickness=1)

    # 4. 只保留前景区域内的边缘
    edge_map = (filtered_edges / 255.0).astype(np.float32)
    edge_map = edge_map * mask

    return edge_map


# -------------------------- TSDE核心计算函数 --------------------------
def calculate_central_composition_score(mask):
    """计算中心构图得分 C(G) 公式(20)"""
    H, W = mask.shape
    center_x, center_y = W / 2.0, H / 2.0

    # 计算前景质心
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return 0.0

    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    # 计算欧氏距离
    distance = np.sqrt((centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2)
    max_distance = np.sqrt(H ** 2 + W ** 2) / 2.0

    return 1.0 - (distance / (max_distance + TSDE_CONFIG["epsilon"]))


def calculate_bilateral_balance_score(mask):
    """计算双边平衡得分 S(G) 公式(21)"""
    H, W = mask.shape
    mid = W // 2

    left_mask = mask[:, :mid]
    right_mask = mask[:, mid:]

    # 翻转右半部分进行对称对比
    flipped_right = cv2.flip(right_mask, 1)

    # 计算L1距离
    diff = np.abs(left_mask - flipped_right)
    l1_distance = np.sum(diff)

    total_foreground = np.sum(left_mask) + np.sum(right_mask)

    return 1.0 - (l1_distance / (total_foreground + TSDE_CONFIG["epsilon"]))


def calculate_structural_detail_score(mask, edge_map):
    """最宽松版结构细节得分 D(G)"""
    background_mask = 1.0 - mask

    # 前景边缘密度（保留所有细节边缘）
    foreground_edges = edge_map * mask
    Rf = np.sum(foreground_edges) / (np.sum(mask) + TSDE_CONFIG["epsilon"])

    # 背景边缘密度（降低惩罚）
    background_edges = edge_map * background_mask
    Rb = np.sum(background_edges) / (np.sum(background_mask) + TSDE_CONFIG["epsilon"])

    # 最低的惩罚系数，最大限度保留细节得分
    return Rf / (Rf + TSDE_CONFIG["lambda_"] * Rb + TSDE_CONFIG["epsilon"])


def calculate_tsde(image_path):
    """
    计算单张唐卡图像的TSDE总分及各子得分
    返回: (总TSDE, 中心得分C, 双边得分S, 细节得分D, 掩码M, 边缘图E, 原始图像, 前景图)
    """
    # 读取图像并转换为RGB
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)

    # 计算掩码和边缘
    mask = get_foreground_mask(img)
    edge_map = get_edge_map(img, mask)

    # 生成前景图（主尊抠图，白色背景）
    foreground_img = np.ones_like(img) * 255
    foreground_img[mask > 0] = img[mask > 0]

    # 计算各子得分
    C = calculate_central_composition_score(mask)
    S = calculate_bilateral_balance_score(mask)
    D = calculate_structural_detail_score(mask, edge_map)

    # 计算总TSDE
    total_tsde = (TSDE_CONFIG["alpha"] * C +
                  TSDE_CONFIG["beta"] * S +
                  TSDE_CONFIG["gamma"] * D)

    return total_tsde, C, S, D, mask, edge_map, img, foreground_img


# -------------------------- 结果可视化与单独输出 --------------------------
def visualize_tsde_result(image_path, save_dir="tsde_output"):
    """
    可视化TSDE计算结果，并单独输出前景图和结构细节图
    所有输出自动保存在save_dir目录下
    """
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 计算TSDE
    tsde, C, S, D, mask, edge_map, img, foreground_img = calculate_tsde(image_path)

    # -------------------------- 单独输出高清图 --------------------------
    # 输出前景图（白色背景）
    foreground_path = os.path.join(save_dir, f"{base_name}_foreground.png")
    Image.fromarray(foreground_img).save(foreground_path, dpi=(300, 300))
    print(f"✅ 前景图已保存: {foreground_path}")

    # 输出结构细节边缘图（黑色背景，白色边缘）
    edge_img = (edge_map * 255).astype(np.uint8)
    edge_path = os.path.join(save_dir, f"{base_name}_edges.png")
    Image.fromarray(edge_img).save(edge_path, dpi=(300, 300))
    print(f"✅ 结构细节图已保存: {edge_path}")

    # 输出掩码图
    mask_img = (mask * 255).astype(np.uint8)
    mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
    Image.fromarray(mask_img).save(mask_path, dpi=(300, 300))
    print(f"✅ 掩码图已保存: {mask_path}")

    # -------------------------- 生成4列对比图 --------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. 原始图像
    axes[0].imshow(img)
    axes[0].set_title("Original Thangka Image", fontsize=14, pad=10)
    axes[0].axis("off")

    # 2. 前景掩码 + 中心得分
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Foreground Mask\nCentral Score C = {C:.4f}", fontsize=14, pad=10)
    axes[1].axis("off")

    # 3. 结构边缘图 + 双边得分
    axes[2].imshow(edge_map, cmap="gray")
    axes[2].set_title(f"Structural Edge Map\nBilateral Score S = {S:.4f}", fontsize=14, pad=10)
    axes[2].axis("off")

    # 4. 得分汇总
    axes[3].text(0.5, 0.75, f"Total TSDE = {tsde:.4f}", fontsize=18, ha="center", va="center", fontweight="bold")
    axes[3].text(0.5, 0.55, f"Detail Score D = {D:.4f}", fontsize=14, ha="center", va="center")
    axes[3].text(0.5, 0.35, f"α={TSDE_CONFIG['alpha']}, β={TSDE_CONFIG['beta']}, γ={TSDE_CONFIG['gamma']}",
                 fontsize=12, ha="center", va="center")
    axes[3].set_title("TSDE Score Summary", fontsize=14, pad=10)
    axes[3].axis("off")

    plt.tight_layout()

    # 保存对比图
    comparison_path = os.path.join(save_dir, f"{base_name}_tsde_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    print(f"✅ 对比图已保存: {comparison_path}")

    plt.show()

    # 打印详细结果
    print("\n" + "=" * 50)
    print(f"TSDE计算结果 - {os.path.basename(image_path)}")
    print("=" * 50)
    print(f"总TSDE得分: {tsde:.4f}")
    print(f"中心构图得分C: {C:.4f}")
    print(f"双边平衡得分S: {S:.4f}")
    print(f"结构细节得分D: {D:.4f}")
    print("=" * 50 + "\n")

    return tsde, C, S, D


# -------------------------- 使用示例 --------------------------
if __name__ == "__main__":
    # 替换为你的唐卡图像路径
    IMAGE_PATH = "./exzample/1776.jpg"

    # 计算并可视化TSDE，所有结果自动保存在tsde_output目录下
    visualize_tsde_result(IMAGE_PATH)