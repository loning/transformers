#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能测试脚本：对比不同隐藏层维度的宇宙本体模型性能

本脚本测试不同隐藏层维度(768, 2048, 8192, 32768)的宇宙本体模型在延迟、
吞吐量、内存使用和参数效率方面的性能差异。
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import psutil
import os
import gc
from typing import Dict, List, Tuple

# 确保每次运行结果一致
torch.manual_seed(42)
np.random.seed(42)

def create_ontological_transformer(hidden_dim: int, num_layers: int = 6) -> nn.Module:
    """
    创建指定隐藏层维度的宇宙本体模型
    """
    # 根据隐藏层维度确定注意力头数量
    if hidden_dim == 768:
        num_heads = 12
    elif hidden_dim == 2048:
        num_heads = 16
    elif hidden_dim == 8192:
        num_heads = 32
    elif hidden_dim == 32768:
        num_heads = 64
    else:
        num_heads = max(8, hidden_dim // 64)
    
    class OntologicalTransformer(nn.Module):
        def __init__(self, hidden_dim, num_layers, num_heads):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.num_heads = num_heads
            
            # 嵌入层
            self.embedding = nn.Embedding(30000, hidden_dim)
            
            # Ontological层
            self.layers = nn.ModuleList([
                OntologicalLayer(hidden_dim, num_heads) for _ in range(num_layers)
            ])
            
            # 输出层
            self.output = nn.Linear(hidden_dim, 30000)
            
        def forward(self, x):
            # 嵌入
            x = self.embedding(x)
            
            # 通过Ontological层
            for layer in self.layers:
                x = layer(x)
            
            # 输出
            return self.output(x)
        
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters())
    
    class OntologicalLayer(nn.Module):
        def __init__(self, hidden_dim, num_heads):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            
            # XOR操作的参数
            self.xor_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
            
            # SHIFT操作的参数
            self.shift_weights = nn.Parameter(torch.randn(hidden_dim))
            
            # FLIP操作的参数
            self.flip_matrix = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
            
            # Layer Norm
            self.layer_norm = nn.LayerNorm(hidden_dim)
            
        def forward(self, x):
            # XOR操作
            xor_out = torch.matmul(x, self.xor_weights)
            
            # SHIFT操作
            shift_out = x + self.shift_weights.unsqueeze(0).unsqueeze(0)
            
            # FLIP操作
            flip_out = torch.matmul(shift_out, self.flip_matrix)
            
            # 残差连接和Layer Norm
            output = self.layer_norm(x + flip_out)
            
            return output
    
    # 创建模型实例
    model = OntologicalTransformer(hidden_dim, num_layers, num_heads)
    return model

def measure_performance(model: nn.Module, input_data: torch.Tensor, batch_size: int = 4, seq_length: int = 128, num_runs: int = 50) -> Dict:
    """
    测量模型性能，包括延迟、吞吐量和内存使用
    """
    device = next(model.parameters()).device
    
    # 清除缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # 记录初始内存使用
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # 预热
    for _ in range(5):
        _ = model(input_data)
    
    # 测量延迟
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_data)
    end_time = time.time()
    
    # 计算延迟和吞吐量
    total_time = end_time - start_time
    latency_ms = (total_time / num_runs) * 1000  # 转换为毫秒
    throughput = (batch_size * seq_length * num_runs) / total_time  # tokens/sec
    
    # 记录峰值内存使用
    current_memory = process.memory_info().rss / (1024 * 1024)  # MB
    memory_usage = current_memory - initial_memory
    
    # 返回性能指标
    return {
        'latency': latency_ms,
        'throughput': throughput,
        'memory': memory_usage,
        'params': model.count_parameters()
    }

def plot_results(results: Dict) -> None:
    """
    绘制性能结果图表
    """
    hidden_dims = list(results.keys())
    hidden_dims.sort()  # 确保顺序
    
    # 提取数据
    latencies = [results[dim]['latency'] for dim in hidden_dims]
    throughputs = [results[dim]['throughput'] for dim in hidden_dims]
    memories = [results[dim]['memory'] for dim in hidden_dims]
    param_counts = [results[dim]['params'] for dim in hidden_dims]
    
    # 计算参数效率
    param_efficiency = [throughputs[i] / (param_counts[i] / 1e6) for i in range(len(hidden_dims))]
    
    # 设置图表样式
    plt.style.use('ggplot')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 延迟图表
    axes[0, 0].bar(range(len(hidden_dims)), latencies, color=colors)
    axes[0, 0].set_xticks(range(len(hidden_dims)))
    axes[0, 0].set_xticklabels([f"dim={dim}" for dim in hidden_dims])
    axes[0, 0].set_ylabel('Latency (ms)')
    axes[0, 0].set_title('Model Latency by Hidden Dimension')
    
    # 吞吐量图表
    axes[0, 1].bar(range(len(hidden_dims)), throughputs, color=colors)
    axes[0, 1].set_xticks(range(len(hidden_dims)))
    axes[0, 1].set_xticklabels([f"dim={dim}" for dim in hidden_dims])
    axes[0, 1].set_ylabel('Throughput (tokens/sec)')
    axes[0, 1].set_title('Model Throughput by Hidden Dimension')
    
    # 内存使用图表
    axes[1, 0].bar(range(len(hidden_dims)), memories, color=colors)
    axes[1, 0].set_xticks(range(len(hidden_dims)))
    axes[1, 0].set_xticklabels([f"dim={dim}" for dim in hidden_dims])
    axes[1, 0].set_ylabel('Memory Usage (MB)')
    axes[1, 0].set_title('Model Memory Usage by Hidden Dimension')
    
    # 参数效率图表
    axes[1, 1].bar(range(len(hidden_dims)), param_efficiency, color=colors)
    axes[1, 1].set_xticks(range(len(hidden_dims)))
    axes[1, 1].set_xticklabels([f"dim={dim}" for dim in hidden_dims])
    axes[1, 1].set_ylabel('Throughput per Million Parameters')
    axes[1, 1].set_title('Parameter Efficiency by Hidden Dimension')
    
    plt.tight_layout()
    plt.savefig('ontological_dimensions_performance.png', dpi=120)
    plt.show()

def run_benchmark(batch_size: int = 4, seq_length: int = 128, device: str = 'cpu') -> None:
    """
    运行不同隐藏层维度的宇宙本体模型性能测试
    """
    print(f"运行宇宙本体模型不同隐藏层维度性能测试...")
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"序列长度: {seq_length}")
    
    device = torch.device(device)
    
    # 准备测试的隐藏层维度
    hidden_dimensions = [768, 2048, 8192, 32768]
    
    # 准备输入数据
    input_data = torch.randint(0, 30000, (batch_size, seq_length)).to(device)
    
    # 存储结果
    results = {}
    
    # 测试每个隐藏层维度
    for hidden_dim in hidden_dimensions:
        print(f"\n测试隐藏层维度: {hidden_dim}")
        
        # 创建模型
        model = create_ontological_transformer(hidden_dim).to(device)
        
        # 打印模型参数
        param_count = model.count_parameters()
        param_str = f"{param_count:,}"
        print(f"模型参数量: {param_str} ({param_count / 1e6:.2f}M)")
        
        # 测量性能
        performance = measure_performance(model, input_data, batch_size, seq_length)
        
        # 打印结果
        print(f"延迟: {performance['latency']:.2f} ms")
        print(f"吞吐量: {performance['throughput']:.2f} tokens/sec")
        print(f"内存使用: {performance['memory']:.2f} MB")
        print(f"参数效率: {performance['throughput'] / (param_count / 1e6):.2f} 吞吐量/百万参数")
        
        # 存储结果
        results[hidden_dim] = performance
        
        # 清理内存
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # 绘制结果
    plot_results(results)
    
    # 打印总结
    print("\n性能测试总结:")
    print(f"{'隐藏层维度':<10} {'参数量':<15} {'延迟(ms)':<10} {'吞吐量':<15} {'内存(MB)':<10} {'参数效率':<15}")
    print("-" * 80)
    
    for hidden_dim in hidden_dimensions:
        perf = results[hidden_dim]
        param_count = perf['params']
        param_str = f"{param_count / 1e6:.2f}M"
        efficiency = perf['throughput'] / (param_count / 1e6)
        
        print(f"{hidden_dim:<10} {param_str:<15} {perf['latency']:<10.2f} {perf['throughput']:<15.2f} {perf['memory']:<10.2f} {efficiency:<15.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试不同隐藏层维度的宇宙本体模型性能")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--seq_length", type=int, default=128, help="序列长度")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="运行设备")
    
    args = parser.parse_args()
    
    run_benchmark(args.batch_size, args.seq_length, args.device) 