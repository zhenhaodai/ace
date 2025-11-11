#!/bin/bash

# 事实核查评测脚本
# 用法示例：
# ./run_evaluation.sh <生成结果文件路径> <数据集名称>

if [ $# -lt 2 ]; then
    echo "用法: $0 <生成结果文件> <数据集名称>"
    echo "示例: $0 output_answers.jsonl hotpotqa"
    echo ""
    echo "支持的数据集: hotpotqa, hover, exfever, bamboogle, musique, 2wikimqa"
    exit 1
fi

INPUT_FILE=$1
DATASET=$2
OUTPUT_FILE="${INPUT_FILE%.jsonl}_metrics.json"

echo "开始评测..."
echo "输入文件: $INPUT_FILE"
echo "数据集: $DATASET"
echo "输出文件: $OUTPUT_FILE"
echo ""

python3 inference/evaluate.py \
    --input_file "$INPUT_FILE" \
    --dataset "$DATASET" \
    --output_file "$OUTPUT_FILE" \
    --verbose

echo ""
echo "评测完成！"
