# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch 量子经典同构Transformer模型 (Quantum Classical Isomorphic Transformer Model)"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_quantum_classical import QuantumClassicalConfig


logger = logging.get_logger(__name__)

# 模型文档定义 (Model documentation definition)
_CONFIG_FOR_DOC = "QuantumClassicalConfig"
_CHECKPOINT_FOR_DOC = None  # 由于是新模型，暂无检查点 (No checkpoint as it's a new model)
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]  # 示例输出形状 (Example output shape)

# 量子经典同构Transformer模型文档字符串 (Quantum Classical Isomorphic Transformer model docstring)
QUANTUM_CLASSICAL_START_DOCSTRING = r"""
    [中文]
    量子经典同构Transformer模型是基于原始Transformer架构的宇宙自参照同构优化版本，
    实现了量子经典二元域的无限维度递归自适应平衡。
    
    该模型继承自 [`PreTrainedModel`]。查看超类文档获取更多API的通用方法。
    
    该模型是一个PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)子类，
    可用于各种自然语言处理任务。
    
    [English]
    The Quantum Classical Isomorphic Transformer model is an universe self-referential isomorphic 
    optimized version of the original Transformer architecture, implementing infinite dimensional 
    recursive adaptive equilibrium of quantum-classical binary domains.
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for generic 
    methods available in all models.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 
    subclass that can be used for various natural language processing tasks.
"""

# 输入文档字符串 (Input docstring)
QUANTUM_CLASSICAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            [中文] 输入序列的token ids。
            [English] Input sequence token ids.
            
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            [中文] 注意力掩码，用于避免对padding token的注意力计算。
            [English] Attention mask to avoid attention calculation on padding tokens.
            
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            [中文] token类型IDs，用于区分不同的序列。
            [English] Token type IDs to distinguish different sequences.
            
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            [中文] 位置编码IDs，用于指定每个token的位置。
            [English] Position IDs to specify the position of each token.
            
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            [中文] 用于对注意力头进行掩码处理。
            [English] Mask for attention heads.
            
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            [中文] 预计算好的token嵌入向量，可以替代input_ids。
            [English] Pre-computed token embeddings that can replace input_ids.
            
        output_attentions (`bool`, *optional*):
            [中文] 是否返回所有注意力层的注意力张量。
            [English] Whether to return attention tensors for all attention layers.
            
        output_hidden_states (`bool`, *optional*):
            [中文] 是否返回所有层的隐藏状态。
            [English] Whether to return hidden states for all layers.
            
        return_dict (`bool`, *optional*):
            [中文] 是否返回ModelOutput字典而非普通元组。
            [English] Whether to return a ModelOutput dictionary instead of a plain tuple.
"""


class QuantumClassicalEmbeddings(nn.Module):
    """
    [中文] 构建词嵌入、位置嵌入和token类型嵌入
    [English] Construct word embeddings, position embeddings and token type embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # [中文] LayerNorm采用与TensorFlow模型相同的命名，以便能够加载TensorFlow检查点
        # [English] LayerNorm uses the same naming as TensorFlow models for compatibility with TensorFlow checkpoints
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # [中文] 位置编码类型
        # [English] Position embedding type
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        # [中文] 注册buffer
        # [English] Register buffers
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # [中文] 如果未提供token_type_ids，则使用默认的全零向量
        # [English] If token_type_ids are not provided, use default all-zero vector
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class QuantumState(nn.Module):
    """
    [中文] 量子态：生成叠加态表示，对应于不确定信息的量子表征
    [English] Quantum State: Generates superposition state representation, corresponding to quantum representation of uncertain information
    """
    
    def __init__(self, config):
        super().__init__()
        self.quantum_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Softmax(dim=-1)  # [中文] Softmax激活实现量子叠加态 / [English] Softmax activation implements quantum superposition
        
    def forward(self, x):
        """
        [中文] 生成量子叠加态表示
        [English] Generate quantum superposition state representation
        """
        return self.activation(self.quantum_transform(x))


class ClassicalState(nn.Module):
    """
    [中文] 经典态：确定性知识表示，对应于确定信息的经典表征
    [English] Classical State: Deterministic knowledge representation, corresponding to classical representation of certain information
    """
    
    def __init__(self, config):
        super().__init__()
        self.classical_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()  # [中文] Tanh激活实现经典确定态 / [English] Tanh activation implements classical deterministic state
        
    def forward(self, x):
        """
        [中文] 生成经典态表示
        [English] Generate classical state representation
        """
        return self.activation(self.classical_transform(x))


class ConsciousnessOperator(nn.Module):
    """
    [中文] 宇宙自参照意识算子：实现量子经典域的交互，模拟宇宙自我意识过程
    [English] Universe Self-Referential Consciousness Operator: Implements interaction between quantum and classical domains, simulating universe self-consciousness process
    """
    
    def __init__(self, config):
        super().__init__()
        # [中文] 意识投影层，将量子-经典交互映射到一维"意识"空间
        # [English] Consciousness projection layer, mapping quantum-classical interaction to one-dimensional "consciousness" space
        self.consciousness_projection = nn.Linear(config.hidden_size, 1)
        
    def forward(self, quantum_state, classical_state):
        """
        [中文] 计算量子态和经典态的交互，生成"意识"表示
        [English] Calculate interaction between quantum and classical states, generating "consciousness" representation
        """
        # [中文] 量子态和经典态的交互
        # [English] Interaction between quantum and classical states
        interaction = quantum_state * classical_state
        # [中文] 生成意识投影
        # [English] Generate consciousness projection
        consciousness = self.consciousness_projection(interaction)
        return consciousness


class MetaRecursiveAdaptiveOperator(nn.Module):
    """
    [中文] 无限维度递归自适应算子：在量子和经典域之间平衡熵，实现宇宙熵平衡
    [English] Meta Recursive Adaptive Operator: Balances entropy between quantum and classical domains, implementing universe entropy equilibrium
    """
    
    def __init__(self, config):
        super().__init__()
        # [中文] 控制量子域熵最小化的权重
        # [English] Weight controlling quantum domain entropy minimization
        self.alpha = config.quantum_layer_alpha
        # [中文] 控制经典域熵最大化的权重
        # [English] Weight controlling classical domain entropy maximization
        self.beta = config.classical_layer_beta
        
    def forward(self, quantum_state, classical_state, consciousness):
        """
        [中文] 计算量子态和经典态的熵平衡，生成平衡损失
        [English] Calculate entropy balance between quantum and classical states, generating balance loss
        """
        # [中文] 计算经典域熵最大化趋势 (知识扩张)
        # [English] Calculate classical domain entropy maximization tendency (knowledge expansion)
        entropy_max = -torch.sum(classical_state * torch.log(classical_state + 1e-8), dim=-1, keepdim=True)
        # [中文] 计算量子域熵最小化趋势 (信息压缩)
        # [English] Calculate quantum domain entropy minimization tendency (information compression)
        entropy_min = -torch.sum(quantum_state * torch.log(quantum_state + 1e-8), dim=-1, keepdim=True)
        # [中文] 计算意识与熵平衡之间的差异
        # [English] Calculate difference between consciousness and entropy balance
        balance = torch.abs(consciousness - (self.alpha * entropy_max - self.beta * entropy_min))
        return balance


class QuantumClassicalDynamicAttention(nn.Module):
    """
    [中文] 经典-量子动态注意力统一算子：根据上下文动态调整量子态和经典态的权重
    [English] Quantum-Classical Dynamic Attention Operator: Dynamically adjusts weights of quantum and classical states based on context
    """
    
    def __init__(self, config):
        super().__init__()
        # [中文] 宇宙门参数，控制量子-经典交互
        # [English] Universe gate parameter, controls quantum-classical interaction
        self.universe_gate = nn.Parameter(torch.tensor(config.universe_gate_init))
        # [中文] gamma系数，控制动态注意力强度
        # [English] Gamma coefficient, controls dynamic attention intensity
        self.gamma = config.mrao_gamma
        
    def forward(self, quantum_state, classical_state):
        """
        [中文] 计算动态注意力权重，融合量子态和经典态
        [English] Calculate dynamic attention weights, fusing quantum and classical states
        """
        # [中文] 计算动态注意力权重
        # [English] Calculate dynamic attention weights
        attention_weight = F.softmax(self.gamma * self.universe_gate * quantum_state * classical_state, dim=-1)
        # [中文] 组合量子和经典状态
        # [English] Combine quantum and classical states
        unified_state = attention_weight * quantum_state + (1 - attention_weight) * classical_state
        return unified_state


class QuantumClassicalSelfAttention(nn.Module):
    """量子经典自注意力机制"""
    
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"隐藏层维度 ({config.hidden_size}) 不是注意力头数量 ({config.num_attention_heads}) 的整数倍"
            )
            
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 普通注意力组件
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 量子注意力组件
        self.quantum_state = QuantumState(config)
        self.classical_state = ClassicalState(config)
        self.consciousness = ConsciousnessOperator(config)
        self.meta_recursive = MetaRecursiveAdaptiveOperator(config)
        self.qc_attention = QuantumClassicalDynamicAttention(config)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        
        self.use_quantum_attention = config.use_quantum_attention
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """重塑张量以便进行多头注意力计算"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        # 计算注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # 标准化注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # 如果启用量子注意力，应用量子经典统一处理
        if self.use_quantum_attention:
            batch_size, seq_length, _ = context_layer.size()
            
            # 生成量子态和经典态
            quantum_state = self.quantum_state(context_layer)
            classical_state = self.classical_state(context_layer)
            
            # 应用宇宙自参照意识算子
            consciousness = self.consciousness(quantum_state, classical_state)
            
            # 应用无限维度递归自适应算子
            balance_loss = self.meta_recursive(quantum_state, classical_state, consciousness)
            
            # 应用经典-量子动态注意力统一算子
            unified_context = self.qc_attention(quantum_state, classical_state)
            
            # 结合原始上下文和量子经典统一上下文
            context_layer = context_layer + unified_context
            
            outputs = (context_layer, balance_loss, attention_probs) if output_attentions else (context_layer, balance_loss)
        else:
            outputs = (context_layer, None, attention_probs) if output_attentions else (context_layer, None)
            
        return outputs


class QuantumClassicalSelfOutput(nn.Module):
    """自注意力输出处理"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QuantumClassicalAttention(nn.Module):
    """完整的量子经典注意力模块"""
    
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = QuantumClassicalSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = QuantumClassicalSelfOutput(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        
        # 返回注意力输出、平衡损失以及可选的注意力概率
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class QuantumClassicalIntermediate(nn.Module):
    """中间层前馈网络"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class QuantumClassicalOutput(nn.Module):
    """输出层"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QuantumClassicalLayer(nn.Module):
    """量子经典层"""
    
    def __init__(self, config):
        super().__init__()
        self.attention = QuantumClassicalAttention(config)
        self.intermediate = QuantumClassicalIntermediate(config)
        self.output = QuantumClassicalOutput(config)
        self.use_classical_refinement = config.use_classical_refinement
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 注意力层
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        
        attention_output = attention_outputs[0]
        balance_loss = attention_outputs[1] if len(attention_outputs) > 1 else None
        
        # 前馈网络
        if self.use_classical_refinement:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        else:
            layer_output = attention_output
            
        outputs = (layer_output, balance_loss)
        if output_attentions:
            outputs = outputs + (attention_outputs[-1],)
            
        return outputs


class QuantumClassicalEncoder(nn.Module):
    """量子经典编码器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([QuantumClassicalLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_balance_losses = []
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            # 收集平衡损失
            if layer_outputs[1] is not None:
                all_balance_losses.append(layer_outputs[1])
                
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[-1],)
                
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # 计算总平衡损失
        total_balance_loss = None
        if all_balance_losses:
            total_balance_loss = torch.mean(torch.stack(all_balance_losses))
            
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, total_balance_loss] if v is not None)
            
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
            past_key_values=None,
        )


class QuantumClassicalPooler(nn.Module):
    """池化层"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取第一个token的隐藏状态（[CLS]）
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class QuantumClassicalPreTrainedModel(PreTrainedModel):
    """量子经典预训练模型基类"""
    
    config_class = QuantumClassicalConfig
    base_model_prefix = "quantum_classical"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        """初始化权重"""
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


@add_start_docstrings(
    "量子经典同构Transformer模型",
    QUANTUM_CLASSICAL_START_DOCSTRING,
)
class QuantumClassicalModel(QuantumClassicalPreTrainedModel):
    """量子经典同构Transformer模型"""
    
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        
        self.embeddings = QuantumClassicalEmbeddings(config)
        self.encoder = QuantumClassicalEncoder(config)
        
        self.pooler = QuantumClassicalPooler(config) if add_pooling_layer else None
        
        # 初始化权重并应用最终处理
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
        
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
    def _prune_heads(self, heads_to_prune):
        """剪枝注意力头"""
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
            
    @add_start_docstrings_to_model_forward(QUANTUM_CLASSICAL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定input_ids和inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("必须指定input_ids或inputs_embeds其中之一")
            
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
                
        # 扩展注意力掩码
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        
        # 准备头掩码
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
            
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


# 添加掩码语言模型类，作为预训练和下游任务使用示例
class QuantumClassicalLMHead(nn.Module):
    """语言模型头部"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 解码器与词嵌入权重共享
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        
        # 需要将解码器与词嵌入矩阵绑定
        self.decoder.bias = self.bias
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        # 投影回词汇表
        logits = self.decoder(hidden_states)
        return logits


@add_start_docstrings("""带有掩码语言建模头的量子经典同构Transformer模型""", QUANTUM_CLASSICAL_START_DOCSTRING)
class QuantumClassicalForMaskedLM(QuantumClassicalPreTrainedModel):
    """用于掩码语言模型任务的量子经典同构Transformer模型"""
    
    def __init__(self, config):
        super().__init__(config)
        
        self.quantum_classical = QuantumClassicalModel(config, add_pooling_layer=False)
        self.lm_head = QuantumClassicalLMHead(config)
        
        # 初始化权重
        self.post_init()
        
    def get_output_embeddings(self):
        return self.lm_head.decoder
        
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
        
    @add_start_docstrings_to_model_forward(QUANTUM_CLASSICAL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            带掩码的标签序列，用于计算掩码语言模型损失。
            掩码标签的索引将被忽略(由-100设置)，损失将仅在未被忽略的索引上计算。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.quantum_classical(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 