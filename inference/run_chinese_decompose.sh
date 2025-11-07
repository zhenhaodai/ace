#!/bin/bash

# Chinese Claim Decomposition Script
# 使用示例脚本来运行中文 claim 分解

# 配置参数
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"  # 模型路径，可通过环境变量设置
INPUT_FILE="${INPUT_FILE:-inference/example_chinese_claims.xlsx}"  # 输入文件
OUTPUT_DIR="${OUTPUT_DIR:-output/decomposed_claims}"  # 输出目录
TEMPERATURE="${TEMPERATURE:-0.0}"  # 温度参数
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"  # GPU 数量
EXPNAME="${EXPNAME:-chinese_claim_decompose}"  # 实验名称

echo "======================================"
echo "中文 Claim 分解脚本"
echo "======================================"
echo "模型路径: $MODEL_PATH"
echo "输入文件: $INPUT_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "温度参数: $TEMPERATURE"
echo "GPU 数量: $TENSOR_PARALLEL_SIZE"
echo "实验名称: $EXPNAME"
echo "======================================"
echo ""

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    echo "请先运行: python inference/create_example_xlsx.py 来创建示例文件"
    exit 1
fi

# 运行分解脚本
python inference/decompose_vllm_chinese.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --expname "$EXPNAME" \
    --top_p 0.99 \
    --max_tokens 2048 \
    --batch_size 32

echo ""
echo "======================================"
echo "完成！结果保存在: $OUTPUT_DIR"
echo "======================================"
