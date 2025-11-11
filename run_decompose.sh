#!/bin/bash

# 中文 Claim 分解脚本运行脚本
# 使用方法: bash run_decompose.sh

MODEL_PATH="/mnt/disk0/s84396777/AceSearcher/saves/lora/qwen3_32b_dpo_lr2e-5_beta0.1_pref0.5_combine_translation/checkpoint-274-merged"
INPUT_FILE="/mnt/disk0/s84396777/AceSearcher/data/0918_data.xlsx"
OUTPUT_DIR="output_chinese_claims"
EXPNAME="qwen3_32b_chinese_decompose"

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请检查路径是否正确，或者模型是否已下载"
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    echo "请检查文件路径是否正确"
    exit 1
fi

# 检查依赖是否安装
echo "检查依赖..."
python -c "import pandas, openpyxl, transformers, vllm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 缺少必要的依赖包"
    echo "请运行: pip install -r requirements.txt"
    exit 1
fi

echo "=========================================="
echo "中文 Claim 分解任务"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "输入文件: $INPUT_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "实验名称: $EXPNAME"
echo "=========================================="
echo ""
echo "开始运行..."
echo ""

# 运行脚本
python inference/decompose_vllm_chinese.py \
    --model_path "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --expname "$EXPNAME" \
    --temperature 0.0 \
    --top_p 0.99 \
    --tensor_parallel_size 1

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "运行完成！"
    echo "=========================================="
    echo "输出文件位置:"
    echo "  - JSONL: $OUTPUT_DIR/decompose_chinese_t0.0_$EXPNAME/decomposed_claims.jsonl"
    echo "  - JSON:  $OUTPUT_DIR/decompose_chinese_t0.0_$EXPNAME/decomposed_claims.json"
    echo "  - Excel: $OUTPUT_DIR/decompose_chinese_t0.0_$EXPNAME/decomposed_claims.xlsx"
    echo ""

    # 显示统计信息
    if [ -f "$OUTPUT_DIR/decompose_chinese_t0.0_$EXPNAME/decomposed_claims.jsonl" ]; then
        TOTAL_LINES=$(wc -l < "$OUTPUT_DIR/decompose_chinese_t0.0_$EXPNAME/decomposed_claims.jsonl")
        echo "共处理了 $TOTAL_LINES 条记录"
    fi
else
    echo ""
    echo "=========================================="
    echo "运行失败！请检查错误信息"
    echo "=========================================="
    exit 1
fi
