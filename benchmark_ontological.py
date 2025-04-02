#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
import numpy as np
import sys
import os
import psutil
import matplotlib.pyplot as plt
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置matplotlib使用英文
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 创建一个简单的OntologicalOperations类，避免导入问题
class OntologicalOperations:
    """
    Implement basic ontological operations: XOR, SHIFT, FLIP
    """
    @staticmethod
    def xor(x, y):
        """
        Simulate XOR operation, implemented bitwise on tensors
        """
        return (x + y) - 2 * (x * y)
    
    @staticmethod
    def shift(x, shift_amount=1):
        """
        Simulate SHIFT operation, circularly shift values in tensor
        """
        return torch.roll(x, shifts=shift_amount, dims=-1)
    
    @staticmethod
    def flip(x):
        """
        Simulate FLIP operation, invert values in tensor
        """
        return 1.0 - x

# 宇宙本体模型配置类
class OntologicalTransformerConfig:
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        use_xor_attention=True,
        use_shift_ffn=True,
        use_flip_output=True,
        num_labels=2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.use_xor_attention = use_xor_attention
        self.use_shift_ffn = use_shift_ffn
        self.use_flip_output = use_flip_output
        self.num_labels = num_labels

    def __str__(self):
        return (f"OntologicalTransformerConfig(hidden_size={self.hidden_size}, "
                f"num_hidden_layers={self.num_hidden_layers}, "
                f"num_attention_heads={self.num_attention_heads}, "
                f"use_xor_attention={self.use_xor_attention}, "
                f"use_shift_ffn={self.use_shift_ffn}, "
                f"use_flip_output={self.use_flip_output})")

# 宇宙本体模型实现
class OntologicalTransformerModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.encoder_layers = torch.nn.ModuleList(
            [self._create_layer() for _ in range(config.num_hidden_layers)]
        )
        
        self.pooler = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = torch.nn.Tanh()
        
    def _create_layer(self):
        config = self.config
        return torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
        )
        
    def forward(self, input_ids=None, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # 嵌入层
        embeddings = self.embeddings(input_ids)
        
        # 位置编码
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        # 合并嵌入
        hidden_states = OntologicalOperations.xor(embeddings, position_embeddings)
        
        # 编码器层
        for layer in self.encoder_layers:
            layer_output = layer(hidden_states)
            shifted_states = OntologicalOperations.shift(hidden_states)
            flipped_output = OntologicalOperations.flip(layer_output)
            hidden_states = OntologicalOperations.xor(shifted_states, flipped_output)
        
        # 池化
        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = self.pooler_activation(pooled_output)
        
        return type('obj', (object,), {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_output
        })

    def get_parameter_count(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters())

# 基础Transformer模型（用于比较）
class VanillaTransformerModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        # 标准Transformer编码器层
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
            batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_hidden_layers
        )
        
        self.pooler = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = torch.nn.Tanh()
        
    def forward(self, input_ids=None, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # 嵌入层
        embeddings = self.embeddings(input_ids)
        
        # 位置编码
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        # 合并嵌入 (标准方式: 加法)
        hidden_states = embeddings + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 转换attention_mask以用于Transformer编码器
        # 在PyTorch TransformerEncoder中，1表示不掩蔽，0表示掩蔽
        if attention_mask is not None:
            # 创建一个适用于nn.TransformerEncoder的注意力掩码
            # 它期望掩码形状为 [batch_size, seq_len, seq_len]
            # 值为 -inf 的地方将被掩蔽，0 的地方将被保留
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
            # 适应PyTorch TransformerEncoder的格式
            # 传递给模型的mask是一个布尔掩码，True表示要掩蔽的位置
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
            
        # 编码器层
        hidden_states = self.encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)
        
        # 池化
        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = self.pooler_activation(pooled_output)
        
        return type('obj', (object,), {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_output
        })

    def get_parameter_count(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters())

# 简化版BERT模型（用于比较）
class SimplifiedBertModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(2, config.hidden_size)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        # 创建BERT层
        self.layers = torch.nn.ModuleList([
            self._create_bert_layer(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.pooler = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = torch.nn.Tanh()
        
    def _create_bert_layer(self, config):
        # Self-attention
        self_attn = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        # Feed-forward
        feed_forward = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.intermediate_size),
            torch.nn.GELU(),
            torch.nn.Linear(config.intermediate_size, config.hidden_size),
            torch.nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Layer norms
        attn_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        ff_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 将它们组合成一个层
        return torch.nn.ModuleDict({
            'attention': self_attn,
            'attention_layer_norm': attn_layer_norm,
            'feed_forward': feed_forward,
            'feed_forward_layer_norm': ff_layer_norm
        })
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        batch_size, seq_length = input_ids.shape
        
        # 如果没有提供token_type_ids，创建全零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 嵌入层
        inputs_embeds = self.embeddings(input_ids)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # 合并嵌入 (BERT方式)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        hidden_states = self.dropout(embeddings)
        
        # 准备注意力掩码
        if attention_mask is not None:
            # 转换为适合MultiheadAttention的掩码
            # 在MultiheadAttention中，True表示要掩蔽的位置
            key_padding_mask = (attention_mask == 0)
            attn_mask = None
        else:
            key_padding_mask = None
            attn_mask = None
        
        # 处理每一层
        for layer_module in self.layers:
            # 自注意力
            attn_output, _ = layer_module['attention'](
                hidden_states, hidden_states, hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            hidden_states = layer_module['attention_layer_norm'](hidden_states + attn_output)
            
            # 前馈网络
            feed_forward_output = layer_module['feed_forward'](hidden_states)
            hidden_states = layer_module['feed_forward_layer_norm'](hidden_states + feed_forward_output)
        
        # 池化
        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = self.pooler_activation(pooled_output)
        
        return type('obj', (object,), {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_output
        })

    def get_parameter_count(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters())

def measure_latency(func, args=(), iterations=100, cpu_memory=False):
    """测量函数执行的延迟时间和资源使用"""
    latencies = []
    cpu_percentages = []
    memory_usages = []
    process = psutil.Process(os.getpid())
    
    for _ in range(iterations):
        # 测量CPU和内存
        if cpu_memory:
            cpu_percentages.append(process.cpu_percent(interval=0.1))
            memory_usages.append(process.memory_info().rss / (1024 * 1024))  # MB
        
        # 测量延迟
        start_time = time.time()
        func(*args)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    latencies = np.array(latencies)
    result = {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "p90": np.percentile(latencies, 90),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99)
    }
    
    if cpu_memory:
        result["cpu"] = {
            "mean": np.mean(cpu_percentages),
            "max": np.max(cpu_percentages)
        }
        result["memory"] = {
            "mean": np.mean(memory_usages),
            "max": np.max(memory_usages)
        }
    
    return result

def profile_operations(tensor_sizes=[100, 500, 1000, 2000]):
    """测试基本宇宙本论操作的性能"""
    print("Testing Ontological Operations Performance")
    
    results = {}
    
    for size in tensor_sizes:
        print(f"\nTesting tensor size: {size}x{size}")
        
        # 创建测试张量
        x = torch.rand(size, size)
        y = torch.rand(size, size)
        
        # 测量XOR操作
        xor_latency = measure_latency(OntologicalOperations.xor, args=(x, y), iterations=20, cpu_memory=True)
        print(f"XOR operation average latency: {xor_latency['mean']:.4f}ms")
        print(f"XOR operation CPU usage: {xor_latency['cpu']['mean']:.2f}% (max: {xor_latency['cpu']['max']:.2f}%)")
        print(f"XOR operation memory usage: {xor_latency['memory']['mean']:.2f} MB (max: {xor_latency['memory']['max']:.2f} MB)")
        
        # 测量SHIFT操作
        shift_latency = measure_latency(OntologicalOperations.shift, args=(x,), iterations=20, cpu_memory=True)
        print(f"SHIFT operation average latency: {shift_latency['mean']:.4f}ms")
        print(f"SHIFT operation CPU usage: {shift_latency['cpu']['mean']:.2f}% (max: {shift_latency['cpu']['max']:.2f}%)")
        print(f"SHIFT operation memory usage: {shift_latency['memory']['mean']:.2f} MB (max: {shift_latency['memory']['max']:.2f} MB)")
        
        # 测量FLIP操作
        flip_latency = measure_latency(OntologicalOperations.flip, args=(x,), iterations=20, cpu_memory=True)
        print(f"FLIP operation average latency: {flip_latency['mean']:.4f}ms")
        print(f"FLIP operation CPU usage: {flip_latency['cpu']['mean']:.2f}% (max: {flip_latency['cpu']['max']:.2f}%)")
        print(f"FLIP operation memory usage: {flip_latency['memory']['mean']:.2f} MB (max: {flip_latency['memory']['max']:.2f} MB)")
        
        results[size] = {
            "xor": xor_latency["mean"],
            "shift": shift_latency["mean"],
            "flip": flip_latency["mean"]
        }
    
    # 绘制结果图表
    plt.figure(figsize=(10, 6))
    sizes = list(results.keys())
    xor_latencies = [results[size]["xor"] for size in sizes]
    shift_latencies = [results[size]["shift"] for size in sizes]
    flip_latencies = [results[size]["flip"] for size in sizes]
    
    plt.plot(sizes, xor_latencies, 'o-', label='XOR')
    plt.plot(sizes, shift_latencies, 's-', label='SHIFT')
    plt.plot(sizes, flip_latencies, '^-', label='FLIP')
    plt.xlabel('Tensor Size')
    plt.ylabel('Average Latency (ms)')
    plt.title('Ontological Operations Performance Analysis')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'ontological_ops_performance_{timestamp}.png')
    print(f"\nPerformance chart saved as: ontological_ops_performance_{timestamp}.png")

def profile_model(model_class, config, batch_size=4, seq_length=16, name="Ontological Transformer"):
    """测试模型的性能"""
    print(f"\nTesting {name} Model Performance (batch_size={batch_size}, seq_length={seq_length}, hidden_size={config.hidden_size}, layers={config.num_hidden_layers})")
    
    # 实例化模型
    model = model_class(config)
    param_count = model.get_parameter_count()
    
    # 如果CUDA可用，将模型移至GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Model parameters: {param_count:,}")
    
    # 创建模型输入
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones((batch_size, seq_length), device=device)
    
    # 额外的输入，如果是BERT模型
    token_type_ids = None
    if name == "BERT":
        token_type_ids = torch.zeros_like(input_ids)
    
    # 预热
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            if name == "BERT":
                _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 测量前向传播延迟
    def forward_pass():
        with torch.no_grad():
            if name == "BERT":
                return model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                return model(input_ids=input_ids, attention_mask=attention_mask)
    
    forward_latency = measure_latency(forward_pass, iterations=20, cpu_memory=True)
    print(f"Forward pass average latency: {forward_latency['mean']:.4f}ms")
    print(f"Forward pass CPU usage: {forward_latency['cpu']['mean']:.2f}% (max: {forward_latency['cpu']['max']:.2f}%)")
    print(f"Forward pass memory usage: {forward_latency['memory']['mean']:.2f} MB (max: {forward_latency['memory']['max']:.2f} MB)")
    
    # 测量内存使用
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            if name == "BERT":
                _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转换为MB
        print(f"Max GPU memory usage: {memory_usage:.2f} MB")
    
    # 性能指标
    tokens_per_second = (batch_size * seq_length) / (forward_latency['mean'] / 1000)
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    
    return {
        "name": name,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "params": param_count,
        "latency": forward_latency['mean'],
        "throughput": tokens_per_second,
        "memory": forward_latency['memory']['max'] if not torch.cuda.is_available() else memory_usage
    }

def compare_models(batch_size=4, seq_length=32, hidden_size=256, layers=4):
    """比较不同模型的性能"""
    print("\nComparing Model Performance")
    
    # 创建基本配置
    base_config = OntologicalTransformerConfig(
        vocab_size=1000,
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=max(1, hidden_size // 64),
        intermediate_size=hidden_size*4,
        use_xor_attention=True,
        use_shift_ffn=True,
        use_flip_output=True
    )
    
    # 测试不同模型
    results = []
    
    # 宇宙本体模型
    onto_result = profile_model(
        OntologicalTransformerModel, 
        base_config, 
        batch_size=batch_size, 
        seq_length=seq_length,
        name="Ontological Transformer"
    )
    results.append(onto_result)
    
    # 普通Transformer模型
    vanilla_result = profile_model(
        VanillaTransformerModel, 
        base_config, 
        batch_size=batch_size, 
        seq_length=seq_length,
        name="Vanilla Transformer"
    )
    results.append(vanilla_result)
    
    # BERT模型
    bert_result = profile_model(
        SimplifiedBertModel, 
        base_config, 
        batch_size=batch_size, 
        seq_length=seq_length,
        name="BERT"
    )
    results.append(bert_result)
    
    # 绘制对比图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 绘制延迟对比
    plt.figure(figsize=(12, 6))
    
    # 延迟对比
    plt.subplot(1, 2, 1)
    model_names = [r['name'] for r in results]
    latencies = [r['latency'] for r in results]
    plt.bar(model_names, latencies)
    plt.xlabel('Model')
    plt.ylabel('Latency (ms)')
    plt.title('Model Latency Comparison')
    plt.grid(True, axis='y')
    
    # 吞吐量对比
    plt.subplot(1, 2, 2)
    throughputs = [r['throughput'] for r in results]
    plt.bar(model_names, throughputs)
    plt.xlabel('Model')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Model Throughput Comparison')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{timestamp}.png')
    print(f"\nModel comparison chart saved as: model_comparison_{timestamp}.png")
    
    # 创建比较表格
    print("\nModel Comparison Table:")
    print("Model\tParams\tLatency(ms)\tThroughput(tokens/sec)\tMemory(MB)")
    for r in results:
        print(f"{r['name']}\t{r['params']:,}\t{r['latency']:.2f}\t{r['throughput']:.2f}\t{r['memory']:.2f}")
    
    return results

def profile_scaling():
    """测试宇宙本体模型在不同配置下的性能"""
    print("\nTesting Ontological Model Scaling Performance")
    
    # 测试不同批次大小
    print("\nTesting impact of different batch sizes:")
    batch_results = []
    for batch_size in [1, 2, 4, 8, 16]:
        config = OntologicalTransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=256*4
        )
        result = profile_model(
            OntologicalTransformerModel,
            config,
            batch_size=batch_size, 
            seq_length=32,
            name="Ontological Transformer"
        )
        batch_results.append(result)
    
    # 测试不同序列长度
    print("\nTesting impact of different sequence lengths:")
    seq_results = []
    for seq_length in [16, 32, 64, 128, 256]:
        config = OntologicalTransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=256*4
        )
        result = profile_model(
            OntologicalTransformerModel,
            config,
            batch_size=4, 
            seq_length=seq_length,
            name="Ontological Transformer"
        )
        seq_results.append(result)
    
    # 测试不同隐藏层大小
    print("\nTesting impact of different hidden sizes:")
    hidden_results = []
    for hidden_size in [128, 256, 512, 768]:
        config = OntologicalTransformerConfig(
            vocab_size=1000,
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=max(1, hidden_size // 64),
            intermediate_size=hidden_size*4
        )
        result = profile_model(
            OntologicalTransformerModel,
            config,
            batch_size=4, 
            seq_length=32,
            name="Ontological Transformer"
        )
        hidden_results.append(result)
    
    # 测试不同层数
    print("\nTesting impact of different layer counts:")
    layer_results = []
    for layers in [2, 4, 6, 8]:
        config = OntologicalTransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=layers,
            num_attention_heads=4,
            intermediate_size=256*4
        )
        result = profile_model(
            OntologicalTransformerModel,
            config,
            batch_size=4, 
            seq_length=32,
            name="Ontological Transformer"
        )
        layer_results.append(result)
    
    # 绘制结果图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 批次大小vs性能
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot([r['batch_size'] for r in batch_results], [r['latency'] for r in batch_results], 'o-')
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.title('Batch Size vs Latency')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot([r['batch_size'] for r in batch_results], [r['throughput'] for r in batch_results], 's-')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Batch Size vs Throughput')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot([r['seq_length'] for r in seq_results], [r['latency'] for r in seq_results], '^-')
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.title('Sequence Length vs Latency')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot([r['hidden_size'] for r in hidden_results], [r['latency'] for r in hidden_results], 'D-')
    plt.xlabel('Hidden Size')
    plt.ylabel('Latency (ms)')
    plt.title('Hidden Size vs Latency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'ontological_transformer_scaling_{timestamp}.png')
    print(f"\nScaling performance chart saved as: ontological_transformer_scaling_{timestamp}.png")
    
    # 创建比较表格
    print("\nPerformance Comparison Tables:")
    print("Batch Size\tLatency(ms)\tThroughput(tokens/sec)\tMemory(MB)")
    for r in batch_results:
        print(f"{r['batch_size']}\t{r['latency']:.2f}\t{r['throughput']:.2f}\t{r['memory']:.2f}")
    
    print("\nSequence Length\tLatency(ms)\tThroughput(tokens/sec)\tMemory(MB)")
    for r in seq_results:
        print(f"{r['seq_length']}\t{r['latency']:.2f}\t{r['throughput']:.2f}\t{r['memory']:.2f}")
    
    print("\nHidden Size\tParameters\tLatency(ms)\tMemory(MB)")
    for r in hidden_results:
        print(f"{r['hidden_size']}\t{r['params']:,}\t{r['latency']:.2f}\t{r['memory']:.2f}")
    
    print("\nLayers\tParameters\tLatency(ms)\tMemory(MB)")
    for r in layer_results:
        print(f"{r['layers']}\t{r['params']:,}\t{r['latency']:.2f}\t{r['memory']:.2f}")

def compare_ontological_operations():
    """比较启用和禁用宇宙本体操作的性能差异"""
    print("\nComparing Impact of Ontological Operations")
    
    configs = [
        {"use_xor_attention": True, "use_shift_ffn": True, "use_flip_output": True, "name": "All Enabled"},
        {"use_xor_attention": False, "use_shift_ffn": True, "use_flip_output": True, "name": "XOR Disabled"},
        {"use_xor_attention": True, "use_shift_ffn": False, "use_flip_output": True, "name": "SHIFT Disabled"},
        {"use_xor_attention": True, "use_shift_ffn": True, "use_flip_output": False, "name": "FLIP Disabled"},
        {"use_xor_attention": False, "use_shift_ffn": False, "use_flip_output": False, "name": "All Disabled"}
    ]
    
    results = []
    for cfg in configs:
        print(f"\nTesting configuration: {cfg['name']}")
        config = OntologicalTransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            use_xor_attention=cfg["use_xor_attention"],
            use_shift_ffn=cfg["use_shift_ffn"],
            use_flip_output=cfg["use_flip_output"]
        )
        
        # 实例化模型
        model = OntologicalTransformerModel(config)
        
        # 如果CUDA可用，将模型移至GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 创建模型输入
        batch_size, seq_length = 4, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones((batch_size, seq_length), device=device)
        
        # 预热
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 测量性能
        def forward_pass():
            with torch.no_grad():
                return model(input_ids=input_ids, attention_mask=attention_mask)
        
        latency_results = measure_latency(forward_pass, iterations=20)
        print(f"Average latency: {latency_results['mean']:.4f}ms")
        
        results.append({
            "name": cfg["name"],
            "xor": cfg["use_xor_attention"],
            "shift": cfg["use_shift_ffn"],
            "flip": cfg["use_flip_output"],
            "latency": latency_results["mean"]
        })
    
    # 绘制比较图表
    plt.figure(figsize=(10, 6))
    plt.bar([r["name"] for r in results], [r["latency"] for r in results])
    plt.xlabel('Configuration')
    plt.ylabel('Average Latency (ms)')
    plt.title('Impact of Ontological Operations on Model Performance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'ontological_ops_comparison_{timestamp}.png')
    print(f"\nOperations comparison chart saved as: ontological_ops_comparison_{timestamp}.png")

if __name__ == "__main__":
    print("Ontological Transformer Model Performance Benchmark\n")
    
    try:
        # 测试基本操作性能（减少数据量）
        profile_operations(tensor_sizes=[100, 500])
        
        # 比较不同模型的性能（只比较两种模型）
        # 创建基本配置
        base_config = OntologicalTransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            use_xor_attention=True,
            use_shift_ffn=True,
            use_flip_output=True
        )
        
        # 测试不同模型
        results = []
        
        # 宇宙本体模型
        onto_result = profile_model(
            OntologicalTransformerModel, 
            base_config, 
            batch_size=4, 
            seq_length=32,
            name="Ontological Transformer"
        )
        results.append(onto_result)
        
        # 普通Transformer模型
        vanilla_result = profile_model(
            VanillaTransformerModel, 
            base_config, 
            batch_size=4, 
            seq_length=32,
            name="Vanilla Transformer"
        )
        results.append(vanilla_result)
        
        # 绘制对比图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 延迟对比
        plt.figure(figsize=(12, 6))
        
        # 延迟对比
        plt.subplot(1, 2, 1)
        model_names = [r['name'] for r in results]
        latencies = [r['latency'] for r in results]
        plt.bar(model_names, latencies)
        plt.xlabel('Model')
        plt.ylabel('Latency (ms)')
        plt.title('Model Latency Comparison')
        plt.grid(True, axis='y')
        
        # 吞吐量对比
        plt.subplot(1, 2, 2)
        throughputs = [r['throughput'] for r in results]
        plt.bar(model_names, throughputs)
        plt.xlabel('Model')
        plt.ylabel('Throughput (tokens/sec)')
        plt.title('Model Throughput Comparison')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'model_comparison_{timestamp}.png')
        print(f"\nModel comparison chart saved as: model_comparison_{timestamp}.png")
        
        # 创建比较表格
        print("\nModel Comparison Table:")
        print("Model\tParams\tLatency(ms)\tThroughput(tokens/sec)\tMemory(MB)")
        for r in results:
            print(f"{r['name']}\t{r['params']:,}\t{r['latency']:.2f}\t{r['throughput']:.2f}\t{r['memory']:.2f}")
        
        # 比较宇宙本体操作对性能的影响
        compare_ontological_operations()
        
        print("\nAll performance tests completed!")
    except Exception as e:
        print(f"\nTest failed: {e}") 