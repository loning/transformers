# Ontological Transformer Performance Testing Guide

This document provides detailed instructions on how to run performance tests for the Ontological Transformer model and generate reports. Tests include basic operation performance tests, comparisons with the Standard Transformer, comparisons with BERT-style models, comparisons with DeepSeek V3 series models, and comparisons of Ontological Transformers with different hidden dimensions (768, 2048, 8192, 32768).

## Table of Contents

- [Test Environment Setup](#test-environment-setup)
- [Basic Performance Tests](#basic-performance-tests)
- [Model Comparison Tests](#model-comparison-tests)
- [Hidden Dimension Tests](#hidden-dimension-tests)
- [Generating Performance Reports](#generating-performance-reports)
- [Interpreting Test Results](#interpreting-test-results)

## Test Environment Setup

### Dependencies Installation

Ensure all necessary dependencies are installed:

```bash
pip install torch numpy matplotlib pandas reportlab psutil
```

If you want to run comparisons with DeepSeek V3, you'll also need to install:

```bash
pip install deepseek-ai
```

### Hardware Requirements

For reproducible results, it's recommended to run tests on hardware with these specifications:

- CPU: At least 8 cores
- Memory: At least 16GB RAM
- Storage: At least 10GB free space
- GPU (optional): For accelerating DeepSeek V3 model tests

## Basic Performance Tests

Basic performance tests evaluate the execution efficiency of the three basic operations (XOR, SHIFT, FLIP) in the Ontological Transformer:

```bash
python benchmark_ontological.py
```

This script will measure:
- Execution time for each basic operation
- Scalability of basic operations with different input sizes
- Memory usage of basic operations

Test results will be output to the terminal and can optionally be saved to a CSV file.

## Model Comparison Tests

### Comparison with Standard Transformer

```bash
python benchmark_ontological_vs_standard.py
```

This test compares the performance of the Ontological Transformer with the Standard Transformer model when processing the same input, including:
- Latency comparison
- Throughput comparison
- Memory usage comparison
- Parameter count comparison

### Comparison with Pre-trained Models

To compare the Ontological Transformer with BERT-style models and DeepSeek V3 models:

```bash
python benchmark_ontological_pretrained.py
```

This script uses a Wikipedia text sample by default. You can modify test settings with parameters:

```bash
python benchmark_ontological_pretrained.py --batch_size 8 --seq_length 256 --include_deepseek --deepseek_size mini
```

Parameter explanations:
- `--batch_size`: Set batch size (default is 4)
- `--seq_length`: Set sequence length (default is 128)
- `--include_deepseek`: Whether to include DeepSeek V3 model
- `--deepseek_size`: DeepSeek V3 model size (mini or base)
- `--device`: Running device (cpu or cuda, default is cpu)

## Hidden Dimension Tests

To test the performance of Ontological Transformers with different hidden dimensions (768, 2048, 8192, 32768):

```bash
python benchmark_ontological_dimensions.py
```

This script will compare the performance differences of Ontological Transformers with different hidden dimensions, including:
- Latency changes with increasing parameter count
- Throughput changes with increasing parameter count
- Memory usage changes with increasing parameter count
- Parameter efficiency evaluation (throughput per million parameters)

## Generating Performance Reports

After completing tests, you can generate detailed PDF performance reports:

### English Version Report

```bash
python generate_performance_report_en.py
```

### Chinese Version Report

```bash
python generate_performance_report.py
```

The generated PDF reports include:
- Detailed performance comparisons of all tested models
- Visualization charts of various performance metrics
- Descriptions of test methods and environment
- Performance analysis and application scenario recommendations
- Conclusions and future work directions

The report files will be saved as:
- English version: `ontological_model_performance_report_en.pdf`
- Chinese version: `ontological_model_performance_report.pdf`

## Interpreting Test Results

### Key Performance Metrics

Test results include the following key metrics:

1. **Latency**: Average time required for the model to process input, measured in milliseconds (ms). Lower is better.

2. **Throughput**: Number of tokens processed per second by the model, measured in tokens/sec. Higher is better.

3. **Memory Usage**: Peak memory occupation during model execution, measured in MB. Lower is better.

4. **Parameter Count**: Total number of parameters in the model, reflecting model complexity and storage requirements.

5. **Parameter Efficiency**: Throughput per million parameters, reflecting the efficiency of the model architecture. Higher is better.

### Model Performance Analysis

Based on test results, you can observe the following points:

1. **Basic Operation Efficiency**: The XOR, SHIFT, and FLIP basic operations in the Ontological Transformer are more efficient than traditional matrix multiplication, which is the foundation of its overall performance advantage.

2. **Relationship Between Scale and Performance**: As hidden dimension increases, the Ontological Transformer's parameter count increases, but the performance degradation rate is far lower than the parameter growth rate, indicating good scalability of its architecture.

3. **Parameter Efficiency**: The Ontological Transformer-768 has the highest parameter efficiency, producing a throughput of about 1,872 per million parameters, far higher than other models.

4. **Application Scenarios**: Different scales of the Ontological Transformer are suitable for different application scenarios, from resource-constrained mobile devices (768) to complex tasks requiring deep semantic understanding (32768).

### Notes for Reproducing Results

1. Ensure tests are run in the same hardware environment, as different CPU/GPU models may lead to result differences.

2. System load affects test results; it's recommended to run tests under low load conditions.

3. For DeepSeek V3 model tests, downloading model weights may take a significant amount of time on first run.

4. Running tests multiple times and taking the average can yield more stable results. 