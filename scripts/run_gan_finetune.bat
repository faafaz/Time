@echo off
REM GAN微调脚本 (Windows)
REM 用法: scripts\run_gan_finetune.bat

REM 设置预训练模型路径
set PRETRAINED_CHECKPOINT=checkpoints\DLinear_Graph\20251109_233713_DLinear_Graph\checkpoint

REM 检查预训练模型是否存在
if not exist "%PRETRAINED_CHECKPOINT%" (
    echo 错误: 预训练模型不存在: %PRETRAINED_CHECKPOINT%
    echo 请先训练DLinear_Graph模型,或修改PRETRAINED_CHECKPOINT变量
    pause
    exit /b 1
)

echo =========================================
echo GAN微调 - DLinear_Graph模型
echo =========================================
echo 预训练模型: %PRETRAINED_CHECKPOINT%
echo =========================================

REM 运行GAN微调
python run_gan_finetune.py ^
    --pretrained_checkpoint %PRETRAINED_CHECKPOINT% ^
    --train_epochs 50 ^
    --gan_pretrain_epochs 10 ^
    --gan_lambda_adv 0.1 ^
    --gan_lambda_pred 1.0 ^
    --gan_g_lr 1e-6 ^
    --gan_d_lr 1e-4 ^
    --gan_n_critic 5 ^
    --gan_disc_hidden 64 ^
    --gan_disc_dropout 0.3 ^
    --batch_size 64 ^
    --learning_rate 1e-4 ^
    --patience 10 ^
    --gpu 0 ^
    --use_gpu True ^
    --gpu_type cuda

echo =========================================
echo GAN微调完成!
echo =========================================
pause

