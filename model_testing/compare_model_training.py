#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比较不同模型在小型数据集上的训练性能
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from datasets import load_dataset
import argparse
from tqdm import tqdm
import logging
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 获取设备
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 宇宙本体模型基本操作
class XOROperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        x_proj = self.proj(x)
        return x ^ x_proj  # 元素级XOR操作

class SHIFTOperation(nn.Module):
    def __init__(self, dim, shift_size=1):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        
    def forward(self, x):
        # 实现循环右移操作
        return torch.roll(x, shifts=self.shift_size, dims=-1)

class FLIPOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # 翻转最高有效位
        msb_mask = torch.ones_like(x)
        msb_mask[..., 0] = -1  # 假设最高有效位在第一个位置
        return x * msb_mask

# 标准自注意力实现
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x):
        batch_size, seq_length = x.size(0), x.size(1)
        new_shape = (batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length = hidden_states.size(0), hidden_states.size(1)
        
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        
        # 注意力得分计算
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 归一化得分为概率
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # 加权汇总值向量
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = (batch_size, seq_length, self.all_head_size)
        context_layer = context_layer.view(*new_shape)
        
        # 输出投影
        output = self.output(context_layer)
        return output

# 宇宙本体模型层实现
class OntologicalTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 定义XOR、SHIFT和FLIP操作
        self.xor_op = XOROperation(hidden_size)
        self.shift_op = SHIFTOperation(hidden_size)
        self.flip_op = FLIPOperation(hidden_size)
        
        # 混合输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states):
        # 应用基本操作
        xor_output = self.xor_op(hidden_states)
        shift_output = self.shift_op(hidden_states)
        flip_output = self.flip_op(hidden_states)
        
        # 组合操作结果
        combined = xor_output + shift_output + flip_output
        
        # 输出层和残差连接
        output = self.output(combined)
        output = self.layer_norm(hidden_states + output)
        
        return output

# 标准Transformer层实现
class StandardTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8, dropout_prob=0.1):
        super().__init__()
        
        # 自注意力层
        self.self_attention = SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_prob=dropout_prob
        )
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        # 层归一化
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        # 自注意力模块
        attention_output = self.self_attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        hidden_states = self.self_attn_layer_norm(hidden_states + attention_output)
        
        # 前馈神经网络模块
        ffn_output = self.ffn(hidden_states)
        ffn_output = self.dropout(ffn_output)
        hidden_states = self.ffn_layer_norm(hidden_states + ffn_output)
        
        return hidden_states

# BERT风格模型层
class BertStyleLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8, dropout_prob=0.1):
        super().__init__()
        
        # 自注意力层
        self.self_attention = SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_prob=dropout_prob
        )
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        # 层归一化 (LayerNorm首先应用)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        # 层归一化后自注意力
        norm_output = self.self_attn_layer_norm(hidden_states)
        attention_output = self.self_attention(norm_output, attention_mask)
        attention_output = self.dropout(attention_output)
        hidden_states = hidden_states + attention_output
        
        # 层归一化后前馈神经网络
        norm_output = self.ffn_layer_norm(hidden_states)
        ffn_output = self.ffn(norm_output)
        ffn_output = self.dropout(ffn_output)
        hidden_states = hidden_states + ffn_output
        
        return hidden_states

# 宇宙本体模型
class OntologicalTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_classes=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        
        self.layers = nn.ModuleList([
            OntologicalTransformerLayer(hidden_size)
            for _ in range(num_layers)
        ])
        
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # 嵌入层
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = embeddings + position_embeddings
        
        # Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # 池化和分类
        pooled_output = hidden_states[:, 0]  # 使用第一个token的表示（CLS token）
        pooled_output = self.activation(self.pooler(pooled_output))
        logits = self.classifier(pooled_output)
        
        return logits

# 标准Transformer模型
class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_classes=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        
        self.layers = nn.ModuleList([
            StandardTransformerLayer(hidden_size)
            for _ in range(num_layers)
        ])
        
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # 嵌入层
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = embeddings + position_embeddings
        
        # 计算注意力掩码
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)
        
        # 池化和分类
        pooled_output = hidden_states[:, 0]  # 使用第一个token的表示（CLS token）
        pooled_output = self.activation(self.pooler(pooled_output))
        logits = self.classifier(pooled_output)
        
        return logits

# BERT风格模型
class BertStyleModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_classes=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        self.layers = nn.ModuleList([
            BertStyleLayer(hidden_size)
            for _ in range(num_layers)
        ])
        
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 嵌入层
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        hidden_states = embeddings + position_embeddings + token_type_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 计算注意力掩码
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)
        
        # 池化和分类
        pooled_output = hidden_states[:, 0]  # 使用第一个token的表示（CLS token）
        pooled_output = self.activation(self.pooler(pooled_output))
        logits = self.classifier(pooled_output)
        
        return logits

# 数据处理
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 加载小型数据集
def load_small_dataset(dataset_name, train_size=5000, val_size=500):
    if dataset_name == "sst2":
        # 加载SST-2情感分析数据集
        dataset = load_dataset("glue", "sst2")
        train_data = dataset["train"].shuffle(seed=42).select(range(train_size))
        val_data = dataset["validation"].shuffle(seed=42).select(range(val_size))
        
        train_texts = train_data["sentence"]
        train_labels = train_data["label"]
        val_texts = val_data["sentence"]
        val_labels = val_data["label"]
        
        return train_texts, train_labels, val_texts, val_labels, 2
        
    elif dataset_name == "cola":
        # 加载CoLA语法接受性数据集
        dataset = load_dataset("glue", "cola")
        train_data = dataset["train"].shuffle(seed=42).select(range(train_size))
        val_data = dataset["validation"].shuffle(seed=42).select(range(val_size))
        
        train_texts = train_data["sentence"]
        train_labels = train_data["label"]
        val_texts = val_data["sentence"]
        val_labels = val_data["label"]
        
        return train_texts, train_labels, val_texts, val_labels, 2
        
    elif dataset_name == "mnli":
        # 加载MNLI自然语言推理数据集
        dataset = load_dataset("glue", "mnli")
        train_data = dataset["train"].shuffle(seed=42).select(range(train_size))
        val_data = dataset["validation_matched"].shuffle(seed=42).select(range(val_size))
        
        train_texts = [f"{premise} [SEP] {hypothesis}" for premise, hypothesis in zip(train_data["premise"], train_data["hypothesis"])]
        train_labels = train_data["label"]
        val_texts = [f"{premise} [SEP] {hypothesis}" for premise, hypothesis in zip(val_data["premise"], val_data["hypothesis"])]
        val_labels = val_data["label"]
        
        return train_texts, train_labels, val_texts, val_labels, 3
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

# 训练一个epoch
def train_epoch(model, data_loader, optimizer, device, criterion):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        if isinstance(model, BertStyleModel):
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        elif isinstance(model, StandardTransformer):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:  # OntologicalTransformer
            outputs = model(input_ids=input_ids)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return epoch_loss / len(data_loader), correct_predictions / total_predictions

# 评估模型
def evaluate(model, data_loader, device, criterion):
    model.eval()
    eval_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            if isinstance(model, BertStyleModel):
                token_type_ids = batch['token_type_ids'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            elif isinstance(model, StandardTransformer):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:  # OntologicalTransformer
                outputs = model(input_ids=input_ids)
            
            loss = criterion(outputs, labels)
            
            eval_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    return eval_loss / len(data_loader), correct_predictions / total_predictions

# 训练和评估
def train_and_evaluate(model, train_loader, val_loader, optimizer, device, criterion, num_epochs=5, model_name="model"):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion)
        train_time = time.time() - start_time
        
        # 评估
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        logger.info(f"Epoch time: {train_time:.2f}s")
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pt")
            logger.info(f"Saved best model with validation accuracy: {val_acc:.4f}")
    
    return history

# 可视化学习曲线
def plot_learning_curves(results, save_path="learning_curves.png"):
    plt.figure(figsize=(12, 10))
    
    # 绘制训练损失
    plt.subplot(2, 2, 1)
    for model_name, history in results.items():
        plt.plot(history["train_loss"], marker='o', label=model_name)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # 绘制验证损失
    plt.subplot(2, 2, 2)
    for model_name, history in results.items():
        plt.plot(history["val_loss"], marker='o', label=model_name)
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # 绘制训练准确率
    plt.subplot(2, 2, 3)
    for model_name, history in results.items():
        plt.plot(history["train_acc"], marker='o', label=model_name)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    # 绘制验证准确率
    plt.subplot(2, 2, 4)
    for model_name, history in results.items():
        plt.plot(history["val_acc"], marker='o', label=model_name)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Learning curves saved to {save_path}")
    plt.close()

def compute_param_count(model):
    return sum(p.numel() for p in model.parameters())

def main():
    parser = argparse.ArgumentParser(description="比较不同模型在小型数据集上的训练性能")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "cola", "mnli"], help="数据集名称")
    parser.add_argument("--train_size", type=int, default=2000, help="训练集大小")
    parser.add_argument("--val_size", type=int, default=400, help="验证集大小")
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=4, help="模型层数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮次")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="model_comparison_results", help="结果保存目录")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 加载数据集
    logger.info(f"加载 {args.dataset} 数据集")
    train_texts, train_labels, val_texts, val_labels, num_classes = load_small_dataset(
        args.dataset, train_size=args.train_size, val_size=args.val_size
    )
    
    # 创建数据集和数据加载器
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 词汇表大小
    vocab_size = tokenizer.vocab_size
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 模型列表
    models = {
        "Ontological": OntologicalTransformer(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=num_classes
        ),
        "Standard": StandardTransformer(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=num_classes
        ),
        "BERT-Style": BertStyleModel(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=num_classes
        )
    }
    
    # 训练和评估每个模型
    results = {}
    model_info = {}
    
    for name, model in models.items():
        logger.info(f"训练 {name} 模型")
        model.to(device)
        
        # 计算参数数量
        param_count = compute_param_count(model)
        logger.info(f"{name} 模型参数数量: {param_count:,}")
        model_info[name] = {"params": param_count}
        
        # 设置优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
        # 训练和评估模型
        history = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            num_epochs=args.epochs,
            model_name=f"{name.lower()}_{args.dataset}"
        )
        
        results[name] = history
        
        # 清除GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 绘制学习曲线
    plot_learning_curves(results, save_path=os.path.join(args.save_dir, f"learning_curves_{args.dataset}.png"))
    
    # 保存结果
    for name, history in results.items():
        history.update({"model_info": model_info[name]})
    
    with open(os.path.join(args.save_dir, f"results_{args.dataset}.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # 创建比较表格
    comparison_data = []
    for name in models.keys():
        comparison_data.append({
            "Model": name,
            "Parameters": f"{model_info[name]['params']:,}",
            "Final Train Acc": f"{results[name]['train_acc'][-1]:.4f}",
            "Final Val Acc": f"{results[name]['val_acc'][-1]:.4f}",
            "Best Val Acc": f"{max(results[name]['val_acc']):.4f}",
            "Param Efficiency": f"{max(results[name]['val_acc']) / model_info[name]['params'] * 1e6:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    comparison_table_path = os.path.join(args.save_dir, f"comparison_table_{args.dataset}.csv")
    df.to_csv(comparison_table_path, index=False)
    logger.info(f"比较表格已保存到 {comparison_table_path}")
    
    # 打印比较表格
    logger.info("\n模型比较:")
    print(df.to_string(index=False))
    
    logger.info("实验完成!")

if __name__ == "__main__":
    main() 