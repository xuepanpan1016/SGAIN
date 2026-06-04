#!/usr/bin/env python3
"""
测试内存优化后的DiT模型
验证HybridMlpBlock参数量减少后是否能解决GPU显存不足问题
"""

import torch
import torch.nn as nn
from models import DiT_L_2, HybridMlpBlock
import gc
import psutil
import os

def get_gpu_memory():
    """获取GPU显存使用情况"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return None

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def test_hybrid_mlp_block():
    """测试HybridMlpBlock的参数量和内存使用"""
    print("=" * 60)
    print("测试HybridMlpBlock参数量和内存使用")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试不同尺寸的HybridMlpBlock
    test_configs = [
        {'in_features': 1024, 'batch_size': 4, 'seq_len': 256},
        {'in_features': 1024, 'batch_size': 8, 'seq_len': 256},
        {'in_features': 1024, 'batch_size': 16, 'seq_len': 256},
    ]
    
    for config in test_configs:
        print(f"\n测试配置: {config}")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        try:
            # 创建HybridMlpBlock
            mlp_block = HybridMlpBlock(
                in_features=config['in_features'],
                hidden_features=config['in_features'] * 4,  # 标准4倍扩展
                act_layer=nn.GELU,
                drop=0.1
            ).to(device)
            
            # 计算参数量
            total_params, trainable_params = count_parameters(mlp_block)
            print(f"  总参数量: {total_params:,}")
            print(f"  可训练参数量: {trainable_params:,}")
            
            # 创建测试输入
            batch_size = config['batch_size']
            seq_len = config['seq_len']
            H = W = int(seq_len ** 0.5)  # 假设是正方形
            
            x = torch.randn(batch_size, seq_len, config['in_features']).to(device)
            
            # 前向传播测试
            with torch.no_grad():
                output = mlp_block(x, H, W)
                print(f"  输入形状: {x.shape}")
                print(f"  输出形状: {output.shape}")
            
            # 显存使用情况
            gpu_mem = get_gpu_memory()
            if gpu_mem:
                print(f"  GPU显存使用: {gpu_mem['allocated']:.2f}GB / {gpu_mem['reserved']:.2f}GB")
                print(f"  峰值显存: {gpu_mem['max_allocated']:.2f}GB")
            
            print("  ✓ 测试通过")
            
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            return False
    
    return True

def test_dit_model():
    """测试完整的DiT模型"""
    print("\n" + "=" * 60)
    print("测试完整DiT-L/2模型")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        # 创建DiT-L/2模型
        model = DiT_L_2(
            input_size=32,
            in_channels=4,
            num_classes=1000,
            learn_sigma=True
        ).to(device)
        
        # 计算模型参数量
        total_params, trainable_params = count_parameters(model)
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 测试不同批次大小
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            print(f"\n测试批次大小: {batch_size}")
            
            try:
                # 创建测试输入
                x = torch.randn(batch_size, 4, 32, 32).to(device)  # 图像输入
                t = torch.randint(0, 1000, (batch_size,)).to(device)  # 时间步
                y = torch.randn(batch_size, 512).to(device)  # 条件输入
                
                # 前向传播
                with torch.no_grad():
                    output = model(x, t, y)
                    print(f"  输入形状: {x.shape}")
                    print(f"  输出形状: {output.shape}")
                
                # 显存使用情况
                gpu_mem = get_gpu_memory()
                if gpu_mem:
                    print(f"  GPU显存使用: {gpu_mem['allocated']:.2f}GB / {gpu_mem['reserved']:.2f}GB")
                    print(f"  峰值显存: {gpu_mem['max_allocated']:.2f}GB")
                    
                    # 检查是否超过23GB限制
                    if gpu_mem['max_allocated'] > 22.0:  # 留1GB缓冲
                        print(f"  ⚠️  警告: 显存使用接近23GB限制")
                    else:
                        print(f"  ✓ 显存使用在安全范围内")
                
                print(f"  ✓ 批次大小 {batch_size} 测试通过")
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"  ✗ 批次大小 {batch_size} 显存不足: {e}")
                return False
            except Exception as e:
                print(f"  ✗ 批次大小 {batch_size} 测试失败: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"模型创建失败: {e}")
        return False

def test_gradient_computation():
    """测试梯度计算和反向传播"""
    print("\n" + "=" * 60)
    print("测试梯度计算和反向传播")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        # 创建小型模型进行梯度测试
        model = DiT_L_2(
            input_size=16,  # 更小的输入尺寸
            in_channels=4,
            num_classes=1000,
            learn_sigma=True
        ).to(device)
        
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 创建测试输入
        batch_size = 2
        x = torch.randn(batch_size, 4, 16, 16).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        y = torch.randn(batch_size, 512).to(device)
        target = torch.randn(batch_size, 8, 16, 16).to(device)  # 目标输出
        
        # 前向传播
        output = model(x, t, y)
        
        # 计算损失
        loss = nn.MSELoss()(output, target)
        print(f"损失值: {loss.item():.6f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 检查梯度
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        print(f"梯度范数: {grad_norm:.6f}")
        
        # 显存使用情况
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            print(f"训练时GPU显存使用: {gpu_mem['allocated']:.2f}GB / {gpu_mem['reserved']:.2f}GB")
            print(f"训练时峰值显存: {gpu_mem['max_allocated']:.2f}GB")
        
        print("✓ 梯度计算测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 梯度计算测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("DiT模型内存优化测试")
    print(f"Python版本: {psutil.sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"GPU总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    else:
        print("CUDA不可用，将使用CPU测试")
    
    # 运行测试
    tests = [
        ("HybridMlpBlock测试", test_hybrid_mlp_block),
        ("DiT模型测试", test_dit_model),
        ("梯度计算测试", test_gradient_computation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n开始 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} 发生异常: {e}")
            results.append((test_name, False))
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！内存优化成功，模型应该可以在23GB GPU上正常运行。")
    else:
        print("\n❌ 部分测试失败，可能需要进一步优化。")
    
    # 最终显存检查
    if torch.cuda.is_available():
        final_mem = get_gpu_memory()
        print(f"\n最终GPU显存使用: {final_mem['allocated']:.2f}GB / {final_mem['reserved']:.2f}GB")
        print(f"测试期间峰值显存: {final_mem['max_allocated']:.2f}GB")
        
        if final_mem['max_allocated'] < 22.0:
            print("✓ 显存使用在23GB限制内，优化成功！")
        else:
            print("⚠️ 显存使用仍然较高，可能需要进一步优化。")

if __name__ == "__main__":
    main()