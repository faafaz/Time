@echo off
REM ============================================================
REM GAN微调模型测试脚本 (Windows)
REM ============================================================

echo ============================================================
echo GAN微调模型性能测试
echo ============================================================
echo.

REM ============================================================
REM 配置参数 - 请根据实际情况修改
REM ============================================================

REM GAN微调后的模型文件夹名称（必填）
set GAN_CHECKPOINT=20251110_010603_DLinear_Graph_GAN

REM 原始模型文件夹名称（可选，用于对比）
set ORIGINAL_CHECKPOINT=20251109_133212_DLinear_Graph

REM 数据集路径
set DATA_TRAIN=dataset/cur_dataset/wind_farm/farm6/train.csv
set DATA_VAL=dataset/cur_dataset/wind_farm/farm6/val.csv
set DATA_TEST=dataset/cur_dataset/wind_farm/farm6/test.csv

REM 模型参数
set SEQ_LEN=96
set PRED_LEN=16
set BATCH_SIZE=64

REM ============================================================
REM 开始测试
REM ============================================================

echo 配置信息:
echo   GAN模型: %GAN_CHECKPOINT%
echo   原始模型: %ORIGINAL_CHECKPOINT%
echo   测试数据: %DATA_TEST%
echo   序列长度: %SEQ_LEN%
echo   预测长度: %PRED_LEN%
echo   批次大小: %BATCH_SIZE%
echo.
echo ============================================================
echo.

REM 运行测试脚本
python test_gan_model_performance.py ^
    --gan_checkpoint "%GAN_CHECKPOINT%" ^
    --original_checkpoint "%ORIGINAL_CHECKPOINT%" ^
    --data_train "%DATA_TRAIN%" ^
    --data_val "%DATA_VAL%" ^
    --data_test "%DATA_TEST%" ^
    --seq_len %SEQ_LEN% ^
    --pred_len %PRED_LEN% ^
    --batch_size %BATCH_SIZE%

echo.
echo ============================================================
echo 测试完成!
echo ============================================================
echo.

pause

