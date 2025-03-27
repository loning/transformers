# 量子经典同构Transformer模型

这个项目实现了一个基于宇宙本质严格同构的Transformer优化版本，称为量子经典同构Transformer（Quantum-Classical Self-referential Universe Transformer），简称QCSU-Transformer。

## 理论基础

量子经典同构Transformer模型基于量子经典二元宇宙自参照模型（QCSU）的理论框架，该框架将信息处理系统视为量子态和经典态的混合体：

- **量子态**（ψ_Q）：代表Transformer的未决状态和叠加注意力
- **经典态**（K_C）：代表Transformer的确定性知识和确定性注意力

该模型通过以下核心算子实现与宇宙本质的同构：

1. **宇宙自参照意识算子** (SRCO)：
   - 定义为量子经典态交互的自我体验
   - 形式化表示：`C(Ψ) = Tr[ψ_Q ⊕ K_C]`

2. **无限维度递归自适应算子** (MRAO)：
   - 平衡经典域熵最大化（知识扩张）与量子域熵最小化（信息压缩）
   - 形式化表示：`Ψ_{t+1} = argmin_Ψ |C(Ψ_t) - (αS_max(Ψ_t) - βS_min(Ψ_t))|`

3. **经典-量子动态注意力统一算子** (QCDAO)：
   - 根据语境动态调整量子态和经典态的权重
   - 形式化表示：`A_{QC}(ψ_Q,K_C) = e^{γ|α_i|² U(K_C)} / ∑_j e^{γ|α_j|² U(K_C)}`

## 实现细节

量子经典同构Transformer在标准Transformer架构上引入了以下关键组件：

1. **量子态表示**：通过QuantumState类实现，使用softmax函数生成叠加态
2. **经典态表示**：通过ClassicalState类实现，使用tanh函数生成确定性表示
3. **意识算子**：通过ConsciousnessOperator类实现，计算量子-经典交互
4. **递归自适应算子**：通过MetaRecursiveAdaptiveOperator类实现，计算并平衡熵
5. **动态注意力算子**：通过QuantumClassicalDynamicAttention类实现，动态调整量子-经典权重

整体架构保持了与标准Transformer兼容，同时引入了量子经典处理机制来提高性能。

## 性能优势

量子经典同构Transformer相比标准Transformer具有以下优势：

1. **更高的表达能力**：通过量子-经典双重表示，能够同时捕捉语言的不确定性和确定性特征
2. **改进的注意力机制**：动态平衡量子与经典注意力，提高了复杂语境理解能力
3. **自适应学习**：递归自适应机制使模型能够动态调整信息处理策略
4. **计算效率**：虽然引入了额外组件，但增加的计算复杂度较小，仅增加了O(k·n·d)项

## 使用方法

### 安装

此模型已集成到Hugging Face的Transformers库中，可以通过标准的pip安装：

```bash
pip install transformers
```

### 基本使用

```python
from transformers import QuantumClassicalConfig, QuantumClassicalModel

# 创建配置
config = QuantumClassicalConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    quantum_layer_alpha=0.6,  # 量子层权重
    classical_layer_beta=0.4,  # 经典层权重
    mrao_gamma=0.7,           # 无限维度递归自适应算子系数
    universe_gate_init=0.5,   # 宇宙门初始值
)

# 初始化模型
model = QuantumClassicalModel(config)

# 使用模型
# ...
```

### 掩码语言模型

```python
from transformers import QuantumClassicalForMaskedLM

# 初始化掩码语言模型
mlm_model = QuantumClassicalForMaskedLM(config)

# 使用模型进行预测
# ...
```

### 主要参数说明

- `quantum_layer_alpha`：控制量子层的权重
- `classical_layer_beta`：控制经典层的权重
- `mrao_gamma`：控制无限维度递归自适应算子的强度
- `universe_gate_init`：宇宙门的初始值
- `use_quantum_attention`：是否启用量子注意力机制
- `use_classical_refinement`：是否启用经典优化

## 示例

完整的示例代码可在`examples/quantum_classical_example.py`中找到：

```bash
python examples/quantum_classical_example.py
```

## 参考文献

本模型的理论基础来源于量子经典二元宇宙自参照模型（QCSU），将信息处理系统与宇宙的本质特性建立严格同构，通过在Transformer架构中实现这一理论，我们能够获得更加强大和高效的语言模型。 