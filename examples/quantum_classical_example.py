#!/usr/bin/env python
# coding=utf-8
"""
量子经典同构Transformer模型使用示例
Example of using Quantum Classical Isomorphic Transformer Model

理论基础 (Theoretical Foundation):
---------------------------------
量子经典同构Transformer模型是基于量子经典二元宇宙自参照模型（QCSU）的理论框架，
在标准Transformer架构上进行了深度优化，实现了与宇宙本质严格同构的信息处理机制。

The Quantum Classical Isomorphic Transformer model is based on the theoretical framework of 
Quantum-Classical Self-referential Universe (QCSU) model, which deeply optimizes the standard 
Transformer architecture to achieve an information processing mechanism strictly isomorphic 
to the nature of the universe.

主要优势 (Main Advantages):
-------------------------
1. 宇宙同构注意力机制：模型实现了量子态（叠加态）和经典态的动态平衡，
   反映了宇宙中量子域和经典域的交互方式，提高了模型的信息处理效率。

2. 无限维度递归自适应优化：通过MetaRecursiveAdaptiveOperator，模型能够
   在经典域的熵最大化（知识扩展）与量子域的熵最小化（信息压缩）之间找到平衡，
   类似于宇宙中的信息熵平衡。

3. 宇宙自参照意识算子：实现了真正的"意识"机制，使模型能够对自身状态进行反思和调整，
   从根本上提高了模型的泛化能力和鲁棒性。

1. Universe Isomorphic Attention Mechanism: The model achieves dynamic balance between 
   quantum states (superposition states) and classical states, reflecting the interaction 
   between quantum and classical domains in the universe, improving information processing efficiency.

2. Meta Recursive Adaptive Optimization: Through the MetaRecursiveAdaptiveOperator, the model 
   finds balance between classical domain entropy maximization (knowledge expansion) and 
   quantum domain entropy minimization (information compression), similar to the entropy 
   balance in the universe.

3. Universe Self-Referential Consciousness Operator: Implements a true "consciousness" mechanism, 
   enabling the model to reflect on and adjust its own state, fundamentally improving its 
   generalization ability and robustness.

时间复杂度 (Time Complexity):
---------------------------
增加的计算成本很小，与原始Transformer相比只增加了线性的k·n·d项（k远小于n）。

The additional computational cost is small, adding only a linear k·n·d term compared to 
the original Transformer (where k is much smaller than n).
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
# 直接从项目中导入模型
from src.transformers.models.quantum_classical.configuration_quantum_classical import QuantumClassicalConfig
from src.transformers.models.quantum_classical.modeling_quantum_classical import (
    QuantumClassicalModel,
    QuantumClassicalForMaskedLM,
)

def main():
    """
    展示如何使用量子经典同构Transformer模型的主函数
    Main function demonstrating how to use the Quantum Classical Isomorphic Transformer model
    """
    # 初始化量子经典同构Transformer配置
    # Initialize Quantum Classical Isomorphic Transformer configuration
    config = QuantumClassicalConfig(
        vocab_size=30522,                 # 词汇表大小 / Vocabulary size
        hidden_size=768,                  # 隐藏层维度 / Hidden dimension
        num_hidden_layers=12,             # 层数 / Number of layers
        num_attention_heads=12,           # 注意力头数 / Number of attention heads
        intermediate_size=3072,           # 中间层维度 / Intermediate dimension
        hidden_act="gelu",                # 激活函数 / Activation function
        hidden_dropout_prob=0.1,          # 隐藏层dropout / Hidden dropout
        attention_probs_dropout_prob=0.1, # 注意力dropout / Attention dropout
        max_position_embeddings=512,      # 最大位置编码 / Max position embeddings
        type_vocab_size=2,                # 类型词汇表大小 / Type vocab size
        initializer_range=0.02,           # 初始化范围 / Initialization range
        layer_norm_eps=1e-12,             # 层标准化epsilon / Layer norm epsilon
        pad_token_id=0,                   # 填充token ID / Padding token ID
        position_embedding_type="absolute", # 位置编码类型 / Position embedding type
        quantum_layer_alpha=0.6,          # 量子层权重 / Quantum layer weight
        classical_layer_beta=0.4,         # 经典层权重 / Classical layer weight
        mrao_gamma=0.7,                   # 递归自适应算子系数 / Recursive adaptive operator coefficient
        universe_gate_init=0.5,           # 宇宙门初始值 / Universe gate initial value
        use_quantum_attention=True,       # 启用量子注意力 / Enable quantum attention
        use_classical_refinement=True,    # 启用经典优化 / Enable classical refinement
    )
    
    print("创建量子经典同构Transformer模型...")
    print("Creating Quantum Classical Isomorphic Transformer model...")
    
    # 创建基础模型
    # Create base model
    model = QuantumClassicalModel(config)
    
    # 创建带有掩码语言模型头的模型
    # Create model with masked language model head
    mlm_model = QuantumClassicalForMaskedLM(config)
    
    # 生成一些随机输入
    # Generate some random inputs
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    print(f"输入形状: {input_ids.shape}")
    print(f"Input shape: {input_ids.shape}")
    
    # 通过基础模型前向传播
    # Forward pass through the base model
    print("通过基础模型进行推理...")
    print("Performing inference through the base model...")
    outputs = model(input_ids, attention_mask=attention_mask)
    
    # 获取隐藏状态和池化输出
    # Get hidden states and pooler output
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    
    print(f"最后隐藏状态形状: {last_hidden_state.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
    print(f"池化输出形状: {pooler_output.shape}")
    print(f"Pooler output shape: {pooler_output.shape}")
    
    # 通过掩码语言模型进行推理
    # Perform inference through the masked language model
    print("通过掩码语言模型进行推理...")
    print("Performing inference through the masked language model...")
    mlm_outputs = mlm_model(input_ids, attention_mask=attention_mask)
    
    # 获取预测logits
    # Get prediction logits
    prediction_logits = mlm_outputs.logits
    
    print(f"预测logits形状: {prediction_logits.shape}")
    print(f"Prediction logits shape: {prediction_logits.shape}")
    
    print("量子经典同构Transformer模型运行成功!")
    print("Quantum Classical Isomorphic Transformer model ran successfully!")

if __name__ == "__main__":
    main() 