#!/bin/bash
# 特征选择与目标直通集成 - 快速开始脚本
# 
# 使用方法:
#   bash examples/run_with_feature_selection.sh [实验编号]
# 
# 实验编号:
#   1 - Baseline (所有开关关闭)
#   2 - 仅GFLU特征选择
#   3 - 仅目标直通
#   4 - GFLU + 目标直通 (推荐)
#   5 - Channel Attention + 目标直通
#   6 - VSN + 目标直通

# 公共参数
MODEL_NAME="iTransformer_xLSTM"
DATA_PATH_LIST="['datasets/train_dataset.csv','datasets/test_dataset.csv','datasets/val_dataset.csv']"
SEQ_LEN=96
PRED_LEN=16
LABEL_LEN=0
DROPOUT=0.1
FREQ="60min"
LRADJ="adjust_fuc"
LEARNING_RATE=0.0001
PATIENCE=10
BATCH_SIZE=128
TRAIN_EPOCHS=100

# iTransformer参数
D_MODEL=512
N_HEADS=8
E_LAYERS=2
D_FF=2048

# xLSTM参数
XLSTM_HIDDEN=128
XLSTM_LAYERS=2
XLSTM_KERNELS="[5, 11, 23]"
RNN_TYPE="slstm"

# 选择实验
EXPERIMENT=${1:-4}  # 默认实验4 (GFLU + 目标直通)

case $EXPERIMENT in
  1)
    echo "=========================================="
    echo "实验 1: Baseline (所有开关关闭)"
    echo "=========================================="
    python run.py \
      --run_type 0 \
      --model_name $MODEL_NAME \
      --data_path_list "$DATA_PATH_LIST" \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --label_len $LABEL_LEN \
      --dropout $DROPOUT \
      --freq $FREQ \
      --lradj $LRADJ \
      --learning_rate $LEARNING_RATE \
      --patience $PATIENCE \
      --batch_size $BATCH_SIZE \
      --train_epochs $TRAIN_EPOCHS \
      --d_model $D_MODEL \
      --n_heads $N_HEADS \
      --e_layers $E_LAYERS \
      --d_ff $D_FF \
      --xlstm_hidden $XLSTM_HIDDEN \
      --xlstm_layers $XLSTM_LAYERS \
      --rnn_type $RNN_TYPE \
      --enable_feature_selection False \
      --enable_target_bypass False
    ;;
  
  2)
    echo "=========================================="
    echo "实验 2: 仅GFLU特征选择"
    echo "=========================================="
    python run.py \
      --run_type 0 \
      --model_name $MODEL_NAME \
      --data_path_list "$DATA_PATH_LIST" \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --label_len $LABEL_LEN \
      --dropout $DROPOUT \
      --freq $FREQ \
      --lradj $LRADJ \
      --learning_rate $LEARNING_RATE \
      --patience $PATIENCE \
      --batch_size $BATCH_SIZE \
      --train_epochs $TRAIN_EPOCHS \
      --d_model $D_MODEL \
      --n_heads $N_HEADS \
      --e_layers $E_LAYERS \
      --d_ff $D_FF \
      --xlstm_hidden $XLSTM_HIDDEN \
      --xlstm_layers $XLSTM_LAYERS \
      --rnn_type $RNN_TYPE \
      --enable_feature_selection True \
      --feature_selection_type gflu \
      --gflu_sparsity_init 0.5 \
      --gflu_learnable_t True \
      --enable_target_bypass False
    ;;
  
  3)
    echo "=========================================="
    echo "实验 3: 仅目标直通"
    echo "=========================================="
    python run.py \
      --run_type 0 \
      --model_name $MODEL_NAME \
      --data_path_list "$DATA_PATH_LIST" \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --label_len $LABEL_LEN \
      --dropout $DROPOUT \
      --freq $FREQ \
      --lradj $LRADJ \
      --learning_rate $LEARNING_RATE \
      --patience $PATIENCE \
      --batch_size $BATCH_SIZE \
      --train_epochs $TRAIN_EPOCHS \
      --d_model $D_MODEL \
      --n_heads $N_HEADS \
      --e_layers $E_LAYERS \
      --d_ff $D_FF \
      --xlstm_hidden $XLSTM_HIDDEN \
      --xlstm_layers $XLSTM_LAYERS \
      --rnn_type $RNN_TYPE \
      --enable_feature_selection False \
      --enable_target_bypass True
    ;;
  
  4)
    echo "=========================================="
    echo "实验 4: GFLU + 目标直通 (推荐)"
    echo "=========================================="
    python run.py \
      --run_type 0 \
      --model_name $MODEL_NAME \
      --data_path_list "$DATA_PATH_LIST" \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --label_len $LABEL_LEN \
      --dropout $DROPOUT \
      --freq $FREQ \
      --lradj $LRADJ \
      --learning_rate $LEARNING_RATE \
      --patience $PATIENCE \
      --batch_size $BATCH_SIZE \
      --train_epochs $TRAIN_EPOCHS \
      --d_model $D_MODEL \
      --n_heads $N_HEADS \
      --e_layers $E_LAYERS \
      --d_ff $D_FF \
      --xlstm_hidden $XLSTM_HIDDEN \
      --xlstm_layers $XLSTM_LAYERS \
      --rnn_type $RNN_TYPE \
      --enable_feature_selection True \
      --feature_selection_type gflu \
      --gflu_sparsity_init 0.5 \
      --gflu_learnable_t True \
      --enable_target_bypass True
    ;;
  
  5)
    echo "=========================================="
    echo "实验 5: Channel Attention + 目标直通"
    echo "=========================================="
    python run.py \
      --run_type 0 \
      --model_name $MODEL_NAME \
      --data_path_list "$DATA_PATH_LIST" \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --label_len $LABEL_LEN \
      --dropout $DROPOUT \
      --freq $FREQ \
      --lradj $LRADJ \
      --learning_rate $LEARNING_RATE \
      --patience $PATIENCE \
      --batch_size $BATCH_SIZE \
      --train_epochs $TRAIN_EPOCHS \
      --d_model $D_MODEL \
      --n_heads $N_HEADS \
      --e_layers $E_LAYERS \
      --d_ff $D_FF \
      --xlstm_hidden $XLSTM_HIDDEN \
      --xlstm_layers $XLSTM_LAYERS \
      --rnn_type $RNN_TYPE \
      --enable_feature_selection True \
      --feature_selection_type channel_attn \
      --channel_attn_reduction 4 \
      --enable_target_bypass True
    ;;
  
  6)
    echo "=========================================="
    echo "实验 6: VSN + 目标直通"
    echo "=========================================="
    python run.py \
      --run_type 0 \
      --model_name $MODEL_NAME \
      --data_path_list "$DATA_PATH_LIST" \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --label_len $LABEL_LEN \
      --dropout $DROPOUT \
      --freq $FREQ \
      --lradj $LRADJ \
      --learning_rate $LEARNING_RATE \
      --patience $PATIENCE \
      --batch_size $BATCH_SIZE \
      --train_epochs $TRAIN_EPOCHS \
      --d_model $D_MODEL \
      --n_heads $N_HEADS \
      --e_layers $E_LAYERS \
      --d_ff $D_FF \
      --xlstm_hidden $XLSTM_HIDDEN \
      --xlstm_layers $XLSTM_LAYERS \
      --rnn_type $RNN_TYPE \
      --enable_feature_selection True \
      --feature_selection_type vsn \
      --vsn_hidden 64 \
      --enable_target_bypass True
    ;;
  
  *)
    echo "错误: 无效的实验编号 $EXPERIMENT"
    echo "请选择 1-6 之间的实验编号"
    echo ""
    echo "实验列表:"
    echo "  1 - Baseline (所有开关关闭)"
    echo "  2 - 仅GFLU特征选择"
    echo "  3 - 仅目标直通"
    echo "  4 - GFLU + 目标直通 (推荐)"
    echo "  5 - Channel Attention + 目标直通"
    echo "  6 - VSN + 目标直通"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="

