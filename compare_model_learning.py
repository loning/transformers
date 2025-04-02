#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型学习效率对比脚本
比较宇宙本体模型、标准Transformer和BERT风格模型在小型数据集上的学习效率
"""

import os
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForSequenceClassification

# 设置随机种子，确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 宇宙本体模型基础操作
class OntologicalOperations(nn.Module):
    def __init__(self):
        super().__init__()
    
    def xor_operation(self, x, y):
        # 实现XOR操作
        return x * (1 - y) + y * (1 - x)
    
    def shift_operation(self, x, shift_size=1):
        # 实现SHIFT操作
        batch_size, seq_len, hidden_size = x.size()
        shifted = torch.zeros_like(x)
        shifted[:, shift_size:, :] = x[:, :-shift_size, :]
        return shifted
    
    def flip_operation(self, x):
        # 实现FLIP操作
        return 1 - x

# 宇宙本体模型注意力机制
class OntologicalAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.operations = OntologicalOperations()
        
        # 查询、键、值投影
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def transpose_for_scores(self, x):
        batch_size, seq_length, hidden_size = x.size()
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # 生成查询、键、值向量
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 使用XOR操作计算注意力分数
        attention_scores = torch.zeros(batch_size, self.num_attention_heads, seq_length, seq_length,
                                      device=hidden_states.device)
        
        for i in range(seq_length):
            q = query_layer[:, :, i, :].unsqueeze(2)  # [batch, heads, 1, head_size]
            attention_scores[:, :, i, :] = self.operations.xor_operation(q, key_layer).mean(dim=-1)
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # 应用softmax获取注意力权重
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)
        
        # 应用输出投影
        output = self.output(context_layer)
        return output

# 宇宙本体前馈网络
class OntologicalFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.operations = OntologicalOperations()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states):
        # 第一个线性变换
        hidden_states = self.dense1(hidden_states)
        
        # 应用XOR和FLIP操作代替传统激活函数
        flipped = self.operations.flip_operation(hidden_states)
        hidden_states = self.operations.xor_operation(hidden_states, flipped)
        
        # 第二个线性变换和dropout
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# 宇宙本体模型层
class OntologicalLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super().__init__()
        self.attention = OntologicalAttention(hidden_size, num_attention_heads, dropout_prob)
        self.feed_forward = OntologicalFeedForward(hidden_size, intermediate_size, dropout_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        # 自注意力子层
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + self.dropout(attention_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + self.dropout(ff_output))
        
        return hidden_states

# 宇宙本体模型
class OntologicalTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, num_attention_heads=12, 
                 intermediate_size=3072, dropout_prob=0.1, max_position_embeddings=512):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 嵌入层
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        # 创建Transformer层
        self.layers = nn.ModuleList([
            OntologicalLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
            for _ in range(num_layers)
        ])
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        
        # 创建位置索引
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 获取嵌入
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # 合并嵌入
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 应用Transformer层
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states

# 标准Transformer的注意力机制
class StandardAttention(nn.Module):
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
        batch_size, seq_length, hidden_size = x.size()
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 传统的注意力计算
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)
        
        output = self.output(context_layer)
        return output

# 标准前馈网络
class StandardFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# 标准Transformer层
class StandardLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super().__init__()
        self.attention = StandardAttention(hidden_size, num_attention_heads, dropout_prob)
        self.feed_forward = StandardFeedForward(hidden_size, intermediate_size, dropout_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + self.dropout(attention_output))
        
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + self.dropout(ff_output))
        
        return hidden_states

# 标准Transformer模型
class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, num_attention_heads=12, 
                 intermediate_size=3072, dropout_prob=0.1, max_position_embeddings=512):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.layers = nn.ModuleList([
            StandardLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
            for _ in range(num_layers)
        ])
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states

# BERT风格模型
class BERTStyleModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, num_attention_heads=12, 
                 intermediate_size=3072, dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.layers = nn.ModuleList([
            StandardLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
            for _ in range(num_layers)
        ])
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states

# 分类头部
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, hidden_states):
        # 使用[CLS]令牌的表示进行分类（假设它是序列的第一个令牌）
        cls_token = hidden_states[:, 0]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        return logits

# 完整的分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, transformer_model, hidden_size, num_classes, dropout_prob=0.1):
        super().__init__()
        self.transformer = transformer_model
        self.classification_head = ClassificationHead(hidden_size, num_classes, dropout_prob)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if hasattr(self.transformer, 'token_type_embeddings') and token_type_ids is not None:
            outputs = self.transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            outputs = self.transformer(input_ids, attention_mask=attention_mask)
        
        logits = self.classification_head(outputs)
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
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(),
            'label': torch.tensor(label)
        }

# 训练函数
def train_epoch(model, dataloader, optimizer, scheduler, device, criterion, epoch, total_epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 计算准确率
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == labels).sum().item()
        
        batch_size = input_ids.size(0)
        epoch_loss += loss.item() * batch_size
        epoch_accuracy += correct
        total_samples += batch_size
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f"{epoch_loss/total_samples:.4f}",
            'acc': f"{epoch_accuracy/total_samples:.4f}"
        })
    
    return epoch_loss / total_samples, epoch_accuracy / total_samples

# 评估函数
def evaluate(model, dataloader, device, criterion):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(outputs, labels)
            
            # 计算准确率
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels).sum().item()
            
            batch_size = input_ids.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_accuracy += correct
            total_samples += batch_size
            
            progress_bar.set_postfix({
                'loss': f"{epoch_loss/total_samples:.4f}",
                'acc': f"{epoch_accuracy/total_samples:.4f}"
            })
    
    return epoch_loss / total_samples, epoch_accuracy / total_samples

# 绘制学习曲线
def plot_learning_curves(results, output_dir):
    epochs = list(range(1, len(results[list(results.keys())[0]]['train_acc']) + 1))
    
    # 创建准确率对比图
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        plt.plot(epochs, metrics['train_acc'], 'o-', label=f"{model_name} (Train)")
        plt.plot(epochs, metrics['val_acc'], 's--', label=f"{model_name} (Val)")
    
    plt.title('Accuracy Comparison Across Models')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    accuracy_plot_path = os.path.join(output_dir, f"accuracy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(accuracy_plot_path)
    
    # 创建损失对比图
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        plt.plot(epochs, metrics['train_loss'], 'o-', label=f"{model_name} (Train)")
        plt.plot(epochs, metrics['val_loss'], 's--', label=f"{model_name} (Val)")
    
    plt.title('Loss Comparison Across Models')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, f"loss_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(loss_plot_path)
    
    # 创建学习速度对比图（前几个epoch的验证准确率）
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        # 仅使用前5个epoch或全部epoch（如果少于5个）
        num_epochs = min(5, len(metrics['val_acc']))
        plt.plot(epochs[:num_epochs], metrics['val_acc'][:num_epochs], 'o-', label=model_name)
    
    plt.title('Early Learning Speed Comparison (Validation Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    speed_plot_path = os.path.join(output_dir, f"learning_speed_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(speed_plot_path)
    
    # 创建最终性能对比条形图
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    final_train_accs = [results[name]['train_acc'][-1] for name in model_names]
    final_val_accs = [results[name]['val_acc'][-1] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, final_train_accs, width, label='Train Accuracy')
    plt.bar(x + width/2, final_val_accs, width, label='Validation Accuracy')
    
    plt.xlabel('Model')
    plt.ylabel('Final Accuracy')
    plt.title('Final Performance Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    final_plot_path = os.path.join(output_dir, f"final_performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(final_plot_path)
    
    return accuracy_plot_path, loss_plot_path, speed_plot_path, final_plot_path

# 生成总结报告
def generate_summary_report(results, model_params, plot_paths, output_dir):
    report = {
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": model_params["dataset"],
        "epochs": model_params["epochs"],
        "batch_size": model_params["batch_size"],
        "learning_rate": model_params["learning_rate"],
        "models_tested": list(results.keys()),
        "results": results,
        "plot_paths": {
            "accuracy": plot_paths[0],
            "loss": plot_paths[1],
            "learning_speed": plot_paths[2],
            "final_performance": plot_paths[3]
        }
    }
    
    # 添加关键性能指标和对比分析
    report["performance_summary"] = {}
    for model_name, metrics in results.items():
        report["performance_summary"][model_name] = {
            "final_train_acc": metrics["train_acc"][-1],
            "final_val_acc": metrics["val_acc"][-1],
            "convergence_epoch": next((i+1 for i, acc in enumerate(metrics["val_acc"]) 
                                     if acc >= 0.95 * max(metrics["val_acc"])), len(metrics["val_acc"])),
            "learning_speed": (metrics["val_acc"][min(2, len(metrics["val_acc"])-1)] 
                              if len(metrics["val_acc"]) > 2 else metrics["val_acc"][-1])
        }
    
    # 保存报告为JSON文件
    report_path = os.path.join(output_dir, f"model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # 创建人类可读的摘要文件
    summary_path = os.path.join(output_dir, f"model_comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("模型学习效率对比摘要\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"测试日期: {report['test_date']}\n")
        f.write(f"数据集: {report['dataset']}\n")
        f.write(f"训练轮数: {report['epochs']}\n")
        f.write(f"批次大小: {report['batch_size']}\n")
        f.write(f"学习率: {report['learning_rate']}\n\n")
        
        f.write("最终性能对比:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'模型':<20} {'训练准确率':<15} {'验证准确率':<15} {'收敛轮数':<15} {'早期学习速度':<15}\n")
        for model_name, metrics in report["performance_summary"].items():
            f.write(f"{model_name:<20} {metrics['final_train_acc']:<15.4f} {metrics['final_val_acc']:<15.4f} "
                   f"{metrics['convergence_epoch']:<15} {metrics['learning_speed']:<15.4f}\n")
        
        f.write("\n性能分析:\n")
        f.write("-" * 80 + "\n")
        
        # 找出性能最好的模型
        best_model = max(report["performance_summary"].items(), 
                        key=lambda x: x[1]["final_val_acc"])
        fastest_model = min(report["performance_summary"].items(), 
                          key=lambda x: x[1]["convergence_epoch"])
        
        f.write(f"最佳验证准确率: {best_model[0]} ({best_model[1]['final_val_acc']:.4f})\n")
        f.write(f"最快收敛模型: {fastest_model[0]} (第{fastest_model[1]['convergence_epoch']}轮)\n\n")
        
        f.write("各模型学习特点:\n")
        for model_name, metrics in report["performance_summary"].items():
            if model_name == 'OntologicalTransformer':
                model_desc = "宇宙本体模型使用XOR、SHIFT和FLIP操作替代标准Transformer中的矩阵乘法和激活函数"
            elif model_name == 'StandardTransformer':
                model_desc = "标准Transformer使用传统的自注意力机制和前馈网络结构"
            elif model_name == 'BERTStyleModel':
                model_desc = "BERT风格模型增加了token类型嵌入，使用与标准Transformer相同的核心组件"
            else:
                model_desc = ""
            
            f.write(f"\n{model_name}:\n")
            f.write(f"- 描述: {model_desc}\n")
            f.write(f"- 最终验证准确率: {metrics['final_val_acc']:.4f}\n")
            f.write(f"- 收敛速度: {'快速' if metrics['convergence_epoch'] <= 3 else '中等' if metrics['convergence_epoch'] <= 6 else '缓慢'}\n")
            f.write(f"- 早期学习表现: {'优秀' if metrics['learning_speed'] >= 0.7 else '良好' if metrics['learning_speed'] >= 0.6 else '一般'}\n")
    
    return report_path, summary_path

# 设置加速设备
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    else:
        return torch.device("cpu")

# DeepSeek V3 模型包装器
class DeepSeekV3Wrapper(nn.Module):
    def __init__(self, model_name="deepseek-ai/deepseek-v3-base"):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.hidden_size = self.model.config.hidden_size
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# 主函数
def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    if args.dataset == "sst2":
        print(f"加载 {args.dataset} 数据集...")
        dataset = load_dataset("glue", "sst2")
        
        if args.limit_samples > 0:
            # 对于测试目的，仅使用有限数量的样本
            train_data = dataset["train"].select(range(min(args.limit_samples, len(dataset["train"]))))
            val_data = dataset["validation"].select(range(min(args.limit_samples // 5, len(dataset["validation"]))))
        else:
            train_data = dataset["train"]
            val_data = dataset["validation"]
        
        train_texts = train_data["sentence"]
        train_labels = train_data["label"]
        val_texts = val_data["sentence"]
        val_labels = val_data["label"]
        
        num_classes = 2
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    print(f"训练样本: {len(train_texts)}，验证样本: {len(val_texts)}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    
    # 创建训练和验证数据集
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length=args.max_seq_length
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_length=args.max_seq_length
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    # 初始化模型
    models = {}
    
    # 模型配置
    model_configs = {
        "ont1024": {
            "name": "OntologicalTransformer-1024",
            "hidden_size": 1024,
            "num_heads": 16,
            "intermediate_size": 4096
        },
        "ont2048": {
            "name": "OntologicalTransformer-2048",
            "hidden_size": 2048,
            "num_heads": 32,
            "intermediate_size": 8192
        },
        "standard": {
            "name": "StandardTransformer",
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "intermediate_size": args.hidden_size * 4
        },
        "bert": {
            "name": "BERTStyleModel",
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "intermediate_size": args.hidden_size * 4
        },
        "deepseekv3": {
            "name": "DeepSeekV3",
        }
    }
    
    # 创建所选模型
    for model_type in args.models:
        if model_type == "ont1024":
            config = model_configs[model_type]
            ont_transformer = OntologicalTransformer(
                vocab_size=vocab_size,
                hidden_size=config["hidden_size"],
                num_layers=args.num_layers,
                num_attention_heads=config["num_heads"],
                intermediate_size=config["intermediate_size"],
                dropout_prob=args.dropout
            )
            ont_classifier = TransformerClassifier(
                ont_transformer, config["hidden_size"], num_classes, args.dropout
            )
            models[config["name"]] = ont_classifier.to(device)
            
        elif model_type == "ont2048":
            config = model_configs[model_type]
            ont_transformer = OntologicalTransformer(
                vocab_size=vocab_size,
                hidden_size=config["hidden_size"],
                num_layers=args.num_layers,
                num_attention_heads=config["num_heads"],
                intermediate_size=config["intermediate_size"],
                dropout_prob=args.dropout
            )
            ont_classifier = TransformerClassifier(
                ont_transformer, config["hidden_size"], num_classes, args.dropout
            )
            models[config["name"]] = ont_classifier.to(device)
            
        elif model_type == "standard":
            config = model_configs[model_type]
            std_transformer = StandardTransformer(
                vocab_size=vocab_size,
                hidden_size=config["hidden_size"],
                num_layers=args.num_layers,
                num_attention_heads=config["num_heads"],
                intermediate_size=config["intermediate_size"],
                dropout_prob=args.dropout
            )
            std_classifier = TransformerClassifier(
                std_transformer, config["hidden_size"], num_classes, args.dropout
            )
            models[config["name"]] = std_classifier.to(device)
            
        elif model_type == "bert":
            config = model_configs[model_type]
            bert_model = BERTStyleModel(
                vocab_size=vocab_size,
                hidden_size=config["hidden_size"],
                num_layers=args.num_layers,
                num_attention_heads=config["num_heads"],
                intermediate_size=config["intermediate_size"],
                dropout_prob=args.dropout
            )
            bert_classifier = TransformerClassifier(
                bert_model, config["hidden_size"], num_classes, args.dropout
            )
            models[config["name"]] = bert_classifier.to(device)
            
        elif model_type == "deepseekv3":
            config = model_configs[model_type]
            try:
                deepseek_model = DeepSeekV3Wrapper()
                models[config["name"]] = deepseek_model.to(device)
            except Exception as e:
                print(f"无法加载DeepSeek V3模型: {e}")
                print("跳过DeepSeek V3模型")
    
    # 打印模型参数计数
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name} 总参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 跟踪训练结果
    results = {name: {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
    } for name in models.keys()}
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n{'='*40}\n训练 {model_name}\n{'='*40}")
        
        # 设置优化器
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        
        # 计算训练步数
        total_steps = len(train_dataloader) * args.epochs
        warmup_steps = int(total_steps * 0.1)  # 10% 的步数用于预热
        
        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        # 训练循环
        for epoch in range(args.epochs):
            # 训练
            train_loss, train_acc = train_epoch(
                model, train_dataloader, optimizer, scheduler, device, criterion, epoch, args.epochs
            )
            
            # 评估
            val_loss, val_acc = evaluate(model, val_dataloader, device, criterion)
            
            # 记录结果
            results[model_name]['train_loss'].append(train_loss)
            results[model_name]['train_acc'].append(train_acc)
            results[model_name]['val_loss'].append(val_loss)
            results[model_name]['val_acc'].append(val_acc)
            
            print(f"轮次 {epoch+1}/{args.epochs} - 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, "
                 f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 保存模型（可选）
        if args.save_models:
            model_path = os.path.join(args.output_dir, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到 {model_path}")
    
    # 绘制学习曲线
    plot_paths = plot_learning_curves(results, args.output_dir)
    
    # 生成总结报告
    model_params = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads
    }
    report_path, summary_path = generate_summary_report(results, model_params, plot_paths, args.output_dir)
    
    print(f"\n训练完成!")
    print(f"结果图表已保存到 {args.output_dir}")
    print(f"摘要报告已保存到 {summary_path}")
    print(f"详细结果已保存到 {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较不同Transformer架构的学习效率")
    
    # 数据集参数
    parser.add_argument("--dataset", type=str, default="sst2", help="要使用的数据集 (sst2)")
    parser.add_argument("--limit_samples", type=int, default=1000, help="限制训练样本数量 (0表示不限制)")
    parser.add_argument("--max_seq_length", type=int, default=64, help="最大序列长度")
    
    # 模型参数
    parser.add_argument("--models", type=str, nargs="+", default=["ont1024", "ont2048", "standard", "bert", "deepseekv3"], 
                        help="要比较的模型 (ont1024, ont2048, standard, bert, deepseekv3)")
    parser.add_argument("--hidden_size", type=int, default=256, help="标准模型的隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=3, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=8, help="标准模型的注意力头数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout概率")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="训练和评估的批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载的工作进程数")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_dir", type=str, default="model_comparison_results", help="结果输出目录")
    parser.add_argument("--save_models", action="store_true", help="保存训练后的模型")
    
    args = parser.parse_args()
    main(args) 