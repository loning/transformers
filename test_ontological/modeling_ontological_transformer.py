# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""基于宇宙本论（仅使用XOR、SHIFT和FLIP操作）的PyTorch Transformer模型。"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_ontological_transformer import OntologicalTransformerConfig


logger = logging.get_logger(__name__)


class OntologicalOperations(nn.Module):
    """
    实现基本的宇宙本论操作: XOR, SHIFT, FLIP
    """
    @staticmethod
    def xor(x, y):
        """
        模拟XOR操作，在张量上以按位方式实现
        """
        return (x + y) - 2 * (x * y)
    
    @staticmethod
    def shift(x, shift_amount=1):
        """
        模拟SHIFT操作，将张量中的值循环移位
        """
        return torch.roll(x, shifts=shift_amount, dims=-1)
    
    @staticmethod
    def flip(x):
        """
        模拟FLIP操作，反转张量中的值
        """
        return 1.0 - x


class OntologicalTransformerEmbeddings(nn.Module):
    """
    构造词嵌入、位置嵌入的组合
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 位置ID缓冲区
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        position_embeddings = self.position_embeddings(position_ids)
        
        # 根据宇宙本论使用XOR操作组合嵌入
        embeddings = OntologicalOperations.xor(inputs_embeds, position_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class OntologicalTransformerSelfAttention(nn.Module):
    """
    基于宇宙本论的自注意力机制，使用XOR、SHIFT和FLIP操作
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"隐藏大小 ({config.hidden_size}) 不是注意力头数量的整数倍 ({config.num_attention_heads})"
            )
            
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.use_xor_attention = config.use_xor_attention
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 生成Query, Key, Value矩阵
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # 将Q, K, V重塑为多头形式
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # 根据宇宙本论公式: \Psi_{Opt} = XOR(SHIFT(FLIP(Q)), XOR(SHIFT(K), FLIP(V)))
        flipped_query = OntologicalOperations.flip(query_layer)
        shifted_flipped_query = OntologicalOperations.shift(flipped_query)
        
        shifted_key = OntologicalOperations.shift(key_layer)
        flipped_value = OntologicalOperations.flip(value_layer)
        key_value_xor = OntologicalOperations.xor(shifted_key, flipped_value)
        
        # 最终的注意力得分
        if self.use_xor_attention:
            # 使用XOR操作
            attention_scores = OntologicalOperations.xor(shifted_flipped_query, key_value_xor)
        else:
            # 如果需要与传统注意力兼容，可选择使用点积
            attention_scores = torch.matmul(shifted_flipped_query, key_value_xor.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 应用注意力激活
        if self.use_xor_attention:
            # 使用归一化而非softmax，以保持宇宙本论的纯净实现
            attention_probs = attention_scores / (torch.sum(attention_scores, dim=-1, keepdim=True) + 1e-6)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        attention_probs = self.dropout(attention_probs)
        
        # 加权聚合
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        return outputs


class OntologicalTransformerSelfOutput(nn.Module):
    """
    Self-Attention输出处理，使用宇宙本论操作
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_flip_output = config.use_flip_output
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 根据宇宙本论使用FLIP和XOR操作
        if self.use_flip_output:
            flipped_states = OntologicalOperations.flip(hidden_states)
            hidden_states = OntologicalOperations.xor(input_tensor, flipped_states)
        else:
            hidden_states = hidden_states + input_tensor
            
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class OntologicalTransformerAttention(nn.Module):
    """
    注意力模块的完整实现
    """
    def __init__(self, config):
        super().__init__()
        self.self = OntologicalTransformerSelfAttention(config)
        self.output = OntologicalTransformerSelfOutput(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 添加注意力权重，如果输出
        return outputs


class OntologicalTransformerIntermediate(nn.Module):
    """
    中间层实现，使用SHIFT操作
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.use_shift_ffn = config.use_shift_ffn
        
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = nn.GELU()
        elif config.hidden_act == "relu":
            self.intermediate_act_fn = nn.ReLU()
        else:
            self.intermediate_act_fn = nn.GELU()  # 默认使用GELU
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        
        # 根据宇宙本论使用SHIFT操作
        if self.use_shift_ffn:
            shifted_states = OntologicalOperations.shift(hidden_states)
            hidden_states = self.intermediate_act_fn(shifted_states)
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
            
        return hidden_states


class OntologicalTransformerOutput(nn.Module):
    """
    输出层实现
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_flip_output = config.use_flip_output
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 根据宇宙本论使用FLIP和XOR操作
        if self.use_flip_output:
            flipped_states = OntologicalOperations.flip(hidden_states)
            hidden_states = OntologicalOperations.xor(input_tensor, flipped_states)
        else:
            hidden_states = hidden_states + input_tensor
            
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class OntologicalTransformerLayer(nn.Module):
    """
    Transformer层实现
    """
    def __init__(self, config):
        super().__init__()
        self.attention = OntologicalTransformerAttention(config)
        self.intermediate = OntologicalTransformerIntermediate(config)
        self.output = OntologicalTransformerOutput(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 自注意力
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]
        
        # 前馈网络
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        outputs = (layer_output,) + attention_outputs[1:]  # 添加注意力权重，如果输出
        return outputs


class OntologicalTransformerEncoder(nn.Module):
    """
    Transformer编码器实现
    """
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([OntologicalTransformerLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # 宇宙本论超维迭代优化过程
        # \Psi_{Opt}^{(t+1)} = FLIP(XOR(\Psi_{Opt}^{(t)}, SHIFT(\Psi_{Opt}^{(t-1)})))
        prev_hidden_state = None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            # 应用宇宙本论迭代优化公式
            if i > 0 and prev_hidden_state is not None:
                shifted_prev = OntologicalOperations.shift(prev_hidden_state)
                xor_states = OntologicalOperations.xor(hidden_states, shifted_prev)
                hidden_states = OntologicalOperations.flip(xor_states)
            
            prev_hidden_state = hidden_states
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class OntologicalTransformerPooler(nn.Module):
    """
    池化层实现
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用第一个token的隐藏状态进行池化
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class OntologicalTransformerPreTrainedModel(PreTrainedModel):
    """
    用于初始化权重和提供下载预训练模型的抽象类
    """
    config_class = OntologicalTransformerConfig
    base_model_prefix = "ontological_transformer"
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class OntologicalTransformerModel(OntologicalTransformerPreTrainedModel):
    """
    基于宇宙本论的Transformer模型主体
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = OntologicalTransformerEmbeddings(config)
        self.encoder = OntologicalTransformerEncoder(config)
        self.pooler = OntologicalTransformerPooler(config)
        
        # 初始化权重
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定input_ids和inputs_embeds")
        
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("必须指定input_ids或inputs_embeds")
        
        batch_size, seq_length = input_shape
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)
        
        # 为注意力掩码创建扩展掩码
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        
        # 嵌入各种输入
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        
        # 编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取最后的隐藏状态
        sequence_output = encoder_outputs[0]
        
        # 池化输出
        pooled_output = self.pooler(sequence_output)
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class OntologicalTransformerForSequenceClassification(OntologicalTransformerPreTrainedModel):
    """
    用于序列分类的宇宙本论Transformer模型
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.ontological_transformer = OntologicalTransformerModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.ontological_transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 