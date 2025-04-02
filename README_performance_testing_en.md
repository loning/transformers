# Ontological Transformer Performance Testing Guide

This document provides detailed instructions on how to run performance tests for the Ontological Transformer model and generate comparison reports.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Basic Performance Testing](#basic-performance-testing)
3. [Pretrained Model Performance Testing](#pretrained-model-performance-testing)
4. [Generating Performance Reports](#generating-performance-reports)
5. [Adding Custom Models](#adding-custom-models)
6. [Interpreting Results](#interpreting-results)
7. [Frequently Asked Questions](#frequently-asked-questions)

## Environment Setup

### Installing Dependencies

```bash
# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install required dependencies
pip install torch numpy matplotlib pandas psutil reportlab tqdm

# If DeepSeek model comparison is needed, also install
pip install deepseek-ai
```

### Hardware Requirements

- CPU testing: Any modern multi-core processor
- GPU testing: CUDA-compatible NVIDIA GPU (minimum 8GB VRAM)
- Memory: Minimum 8GB, recommended 16GB+ (especially when testing large models)

## Basic Performance Testing

Basic performance testing primarily evaluates the efficiency of the model's fundamental operations (XOR, SHIFT, FLIP).

```bash
# Run basic performance test
python benchmark_ontological.py

# Run with GPU if available
python benchmark_ontological.py --device cuda

# Customize batch size and sequence length
python benchmark_ontological.py --batch_size 8 --seq_length 256
```

### Parameter Description

- `--device`: Device type, options 'cpu' or 'cuda'
- `--batch_size`: Batch size, default is 4
- `--seq_length`: Sequence length, default is 128
- `--num_runs`: Number of runs, default is 100
- `--warm_up`: Number of warm-up runs, default is 10

## Pretrained Model Performance Testing

Pretrained model performance testing compares the Ontological Transformer with other models in processing actual text.

```bash
# Run pretrained model performance test
python benchmark_ontological_pretrained.py

# Include DeepSeek V3 model comparison
python benchmark_ontological_pretrained.py --include_deepseek

# Compare different hidden dimension sizes of Ontological Transformer
python benchmark_ontological_pretrained.py --include_ontological_sizes
```

### Parameter Description

- `--device`: Device type, options 'cpu' or 'cuda'
- `--batch_size`: Batch size, default is 4
- `--seq_length`: Sequence length, default is 128
- `--include_deepseek`: Whether to include DeepSeek model, default is False
- `--deepseek_size`: DeepSeek model size, options 'mini' or 'base', default is 'mini'
- `--include_ontological_sizes`: Whether to test different hidden dimensions (768, 2048, 8192, 32768) of Ontological Transformer, default is False
- `--test_wiki_text`: Whether to use Wikipedia sample text, default is True

## Generating Performance Reports

After performance testing is complete, you can generate detailed PDF reports comparing the performance data of all models.

```bash
# Generate Chinese version performance report
python generate_performance_report.py

# Generate English version performance report
python generate_performance_report_en.py
```

### Report Contents

The generated PDF report includes the following:

1. Performance metrics comparison table for different models
2. Latency comparison chart
3. Throughput comparison chart
4. Memory usage comparison chart
5. Parameter count comparison chart
6. Parameter efficiency comparison chart (throughput per million parameters)
7. Parameter count vs. latency relationship chart
8. Detailed performance analysis and application scenarios for different models

## Adding Custom Models

If you want to add custom models for comparison testing, follow these steps:

1. Add new model implementation in the `ModelBenchmark` class in `benchmark_ontological_pretrained.py`
2. Add initialization code in the class's `__init__` method
3. Modify the `test_models` method to include testing of the new model
4. Update the `results` dictionary in `generate_performance_report.py` and `generate_performance_report_en.py` to include performance data of the new model

Example:
```python
# Add new model in ModelBenchmark class
def __init__(self, ...):
    # Existing code
    
    # Initialize new model
    if include_my_model:
        self.my_model = MyCustomModel(...)
        print("Initializing My Custom Model")

def test_models(self, ...):
    # Existing code
    
    # Test new model
    if hasattr(self, 'my_model'):
        start_time = time.time()
        # Model forward pass code
        latency = (time.time() - start_time) * 1000 / num_runs
        results['MyModel'] = {
            'latency': latency,
            'throughput': (batch_size * seq_length * 1000) / latency,
            'memory': self.get_memory_usage(),
            'cpu_usage': self.get_cpu_usage()
        }
```

## Interpreting Results

### Latency

Latency represents the average time (milliseconds) required for the model to process a batch of inputs. Lower latency means faster model response, especially suitable for applications requiring real-time processing.

### Throughput

Throughput represents the number of tokens the model can process per second. Higher throughput means higher efficiency in processing large amounts of data, suitable for batch processing scenarios.

### Memory Usage

Memory usage represents the peak memory consumption (MB) during model operation. Lower memory usage means the model requires fewer system resources, more suitable for resource-constrained environments.

### Parameter Efficiency

Parameter efficiency measures the throughput produced per million parameters. This metric reflects the design efficiency of the model architecture; models with high parameter efficiency can achieve better performance with fewer parameters.

## Frequently Asked Questions

### Q: What should I do if I encounter a "CUDA out of memory" error during testing?

A: Reduce the batch size (`--batch_size`) or sequence length (`--seq_length`), or use a GPU with more VRAM.

### Q: Is it normal that testing DeepSeek models is very slow?

A: Yes, large models like DeepSeek have an extremely large number of parameters and will test slowly on ordinary hardware, especially in CPU mode.

### Q: How do I test my own trained Ontological Transformer model?

A: Modify the model loading part in `benchmark_ontological_pretrained.py` to point to your own model weight files.

### Q: What if the charts in the report are unclear or Chinese characters display as squares?

A: Ensure that the corresponding Chinese fonts are installed on your system, or modify the font settings in `generate_performance_report.py`.

---

For other questions, please submit an issue or contact the project maintainer. 