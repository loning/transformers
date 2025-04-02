#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import os

# 直接导入模块，不通过transformers包的导入机制
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from transformers.models.ontological_transformer.configuration_ontological_transformer import OntologicalTransformerConfig
from transformers.models.ontological_transformer.modeling_ontological_transformer import (
    OntologicalTransformerModel,
    OntologicalTransformerForSequenceClassification,
    OntologicalOperations
)

# 测试基本操作
def test_ontological_operations():
    print("测试宇宙本论基本操作：")
    
    # 创建测试张量
    a = torch.tensor([0.2, 0.5, 0.8])
    b = torch.tensor([0.3, 0.4, 0.7])
    
    # 测试XOR操作
    xor_result = OntologicalOperations.xor(a, b)
    print(f"XOR操作: {a} XOR {b} = {xor_result}")
    
    # 测试SHIFT操作
    shift_result = OntologicalOperations.shift(a)
    print(f"SHIFT操作: SHIFT({a}) = {shift_result}")
    
    # 测试FLIP操作
    flip_result = OntologicalOperations.flip(a)
    print(f"FLIP操作: FLIP({a}) = {flip_result}")

# 测试模型配置
def test_model_config():
    print("\n测试模型配置：")
    
    # 创建默认配置
    config = OntologicalTransformerConfig()
    print(f"默认配置: {config}")
    
    # 创建自定义配置
    custom_config = OntologicalTransformerConfig(
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        use_xor_attention=True,
        use_shift_ffn=True,
        use_flip_output=True
    )
    print(f"自定义配置: {custom_config}")

# 测试模型实例化
def test_model_instantiation():
    print("\n测试模型实例化：")
    
    # 创建配置
    config = OntologicalTransformerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512
    )
    
    # 实例化基本模型
    model = OntologicalTransformerModel(config)
    print(f"基本模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 实例化分类模型
    config.num_labels = 2
    classifier_model = OntologicalTransformerForSequenceClassification(config)
    print(f"分类模型参数数量: {sum(p.numel() for p in classifier_model.parameters())}")

# 测试模型前向传播
def test_model_forward():
    print("\n测试模型前向传播：")
    
    # 创建配置
    config = OntologicalTransformerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512
    )
    
    # 实例化模型
    model = OntologicalTransformerModel(config)
    
    # 创建输入
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    # 执行前向传播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"输出隐藏状态形状: {outputs.last_hidden_state.shape}")
    print(f"池化输出形状: {outputs.pooler_output.shape}")
    
    # 测试分类模型
    config.num_labels = 3
    classifier_model = OntologicalTransformerForSequenceClassification(config)
    
    classification_outputs = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"分类模型输出形状: {classification_outputs.logits.shape}")

if __name__ == "__main__":
    print("基于宇宙本论的Transformer模型测试\n")
    
    try:
        test_ontological_operations()
        test_model_config()
        test_model_instantiation()
        test_model_forward()
        
        print("\n所有测试通过！宇宙本论模型成功实现。")
    except Exception as e:
        print(f"\n测试失败: {e}") 