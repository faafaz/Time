#!/bin/bash
# ============================================================
# GAN微调模型测试脚本 (Linux/Mac)
# ============================================================

echo "============================================================"
echo "GAN微调模型性能测试"
echo "============================================================"
echo ""

# ============================================================
# 配置参数 - 请根据实际情况修改
# ============================================================

# GAN微调后的模型文件夹名称（必填）
GAN_CHECKPOINT="20251110_010603_DLinear_Graph_GAN"

# 原始模型文件夹名称（可选，用于对比）
ORIGINAL_CHECKPOINT="20251109_133212_DLinear_Graph"

# 数据集路径
DATA_TRAIN="dataset/cur_dataset/wind_farm/farm6/train.csv"
DATA_VAL="dataset/cur_dataset/wind_farm/farm6/val.csv"
DATA_TEST="dataset/cur_dataset/wind_farm/farm6/test.csv"

# 模型参数
SEQ_LEN=96
PRED_LEN=16
BATCH_SIZE=64

# ============================================================
# 开始测试
# ============================================================

echo "配置信息:"
echo "  GAN模型: $GAN_CHECKPOINT"
echo "  原始模型: $ORIGINAL_CHECKPOINT"
echo "  测试数据: $DATA_TEST"
echo "  序列长度: $SEQ_LEN"
echo "  预测长度: $PRED_LEN"
echo "  批次大小: $BATCH_SIZE"
echo ""
echo "============================================================"
echo ""

# 运行测试脚本
python test_gan_model_performance.py \
    --gan_checkpoint "$GAN_CHECKPOINT" \
    --original_checkpoint "$ORIGINAL_CHECKPOINT" \
    --data_train "$DATA_TRAIN" \
    --data_val "$DATA_VAL" \
    --data_test "$DATA_TEST" \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --batch_size $BATCH_SIZE

echo ""
echo "============================================================"
echo "测试完成!"
echo "============================================================"
echo ""

