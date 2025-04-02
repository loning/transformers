# 宇宙本体模型性能测试指南

本文档详细介绍了如何运行宇宙本体模型(Ontological Transformer)的性能测试并生成比较报告。

## 目录

1. [环境配置](#环境配置)
2. [基本性能测试](#基本性能测试)
3. [预训练模型性能测试](#预训练模型性能测试)
4. [生成性能报告](#生成性能报告)
5. [添加自定义模型](#添加自定义模型)
6. [结果解读](#结果解读)
7. [常见问题](#常见问题)

## 环境配置

### 依赖项安装

```bash
# 创建并激活虚拟环境(可选)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装所需依赖
pip install torch numpy matplotlib pandas psutil reportlab tqdm

# 如果需要DeepSeek模型对比，还需安装
pip install deepseek-ai
```

### 硬件要求

- CPU测试: 任何现代多核处理器
- GPU测试: CUDA兼容的NVIDIA GPU (至少8GB显存)
- 内存: 至少8GB, 推荐16GB以上(特别是测试大型模型时)

## 基本性能测试

基本性能测试主要评估模型的基础操作(XOR、SHIFT、FLIP)效率。

```bash
# 运行基本性能测试
python benchmark_ontological.py

# 使用GPU运行(如果可用)
python benchmark_ontological.py --device cuda

# 自定义批次大小和序列长度
python benchmark_ontological.py --batch_size 8 --seq_length 256
```

### 参数说明

- `--device`: 设备类型，可选'cpu'或'cuda'
- `--batch_size`: 批次大小，默认为4
- `--seq_length`: 序列长度，默认为128
- `--num_runs`: 运行次数，默认为100
- `--warm_up`: 预热运行次数，默认为10

## 预训练模型性能测试

预训练模型性能测试对比宇宙本体模型与其他模型在实际文本处理中的性能。

```bash
# 运行预训练模型性能测试
python benchmark_ontological_pretrained.py

# 包含DeepSeek V3模型对比
python benchmark_ontological_pretrained.py --include_deepseek

# 对比不同隐藏层维度的宇宙本体模型
python benchmark_ontological_pretrained.py --include_ontological_sizes
```

### 参数说明

- `--device`: 设备类型，可选'cpu'或'cuda'
- `--batch_size`: 批次大小，默认为4
- `--seq_length`: 序列长度，默认为128
- `--include_deepseek`: 是否包含DeepSeek模型，默认为False
- `--deepseek_size`: DeepSeek模型大小，可选'mini'或'base'，默认为'mini'
- `--include_ontological_sizes`: 是否测试不同隐藏层维度(768, 2048, 8192, 32768)的宇宙本体模型，默认为False
- `--test_wiki_text`: 是否使用维基百科样本文本，默认为True

## 生成性能报告

性能测试完成后，您可以生成详细的PDF报告，对比所有模型的性能数据。

```bash
# 生成中文版性能报告
python generate_performance_report.py

# 生成英文版性能报告
python generate_performance_report_en.py
```

### 报告内容

生成的PDF报告包含以下内容：

1. 不同模型的性能指标对比表
2. 延迟时间对比图
3. 吞吐量对比图
4. 内存使用对比图
5. 参数数量对比图
6. 参数效率对比图(每百万参数的吞吐量)
7. 参数数量与延迟关系图
8. 详细的性能分析和不同模型的适用场景

## 添加自定义模型

如果您想添加自定义模型进行对比测试，请按照以下步骤操作：

1. 在`benchmark_ontological_pretrained.py`中的`ModelBenchmark`类中添加新的模型实现
2. 在类的`__init__`方法中添加初始化代码
3. 修改`test_models`方法以包含新模型的测试
4. 在`generate_performance_report.py`和`generate_performance_report_en.py`中更新`results`字典以包含新模型的性能数据

示例：
```python
# 在ModelBenchmark类中添加新模型
def __init__(self, ...):
    # 已有代码
    
    # 初始化新模型
    if include_my_model:
        self.my_model = MyCustomModel(...)
        print("初始化我的自定义模型")

def test_models(self, ...):
    # 已有代码
    
    # 测试新模型
    if hasattr(self, 'my_model'):
        start_time = time.time()
        # 模型前向传播代码
        latency = (time.time() - start_time) * 1000 / num_runs
        results['MyModel'] = {
            'latency': latency,
            'throughput': (batch_size * seq_length * 1000) / latency,
            'memory': self.get_memory_usage(),
            'cpu_usage': self.get_cpu_usage()
        }
```

## 结果解读

### 延迟时间(Latency)

延迟时间表示模型处理一批输入所需的平均时间(毫秒)。延迟时间越低，模型响应越快，特别适合对实时性要求高的应用场景。

### 吞吐量(Throughput)

吞吐量表示模型每秒能处理的token数量。吞吐量越高，模型处理大量数据的效率越高，适合批量处理场景。

### 内存使用(Memory Usage)

内存使用表示模型运行期间的峰值内存占用(MB)。内存使用越低，模型对系统资源要求越小，更适合资源受限环境。

### 参数效率(Parameter Efficiency)

参数效率衡量每百万参数产生的吞吐量。这一指标反映了模型架构的设计效率，参数效率高的模型能以更少的参数达到更好的性能。

## 常见问题

### Q: 测试时出现"CUDA out of memory"错误怎么办？

A: 减小批次大小(`--batch_size`)或序列长度(`--seq_length`)，或使用显存更大的GPU。

### Q: DeepSeek模型测试速度很慢，正常吗？

A: 是的，DeepSeek等大型模型参数量极大，在普通硬件上测试会较慢，特别是CPU模式下。

### Q: 如何测试自己训练的宇宙本体模型？

A: 修改`benchmark_ontological_pretrained.py`中的模型加载部分，指向您自己的模型权重文件。

### Q: 报告中的图表不清晰或中文显示为方块怎么办？

A: 确保系统安装了相应的中文字体，或修改`generate_performance_report.py`中的字体设置。

---

如有其他问题，请提交Issue或联系项目维护者。 