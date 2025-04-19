#!/bin/bash

# === 单卡环境强制设置分布式变量 ===
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

# === 其他参数 ===
CODE_DIR="C:\apps\code\vistar"
CONFIG_FILE="${CODE_DIR}/configs/cd/dfc25_track2/default.yaml"
OUTPUT_DIR="${CODE_DIR}/dfc25_track2_a100_hky_att4"

SEED=42
AMP_TYPE="fp16"
GPU_ID=0

NUM_WORKERS=16
CUDNN_BENCHMARK=1

# ---- 训练流程控制 ----
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# =================================================================
#                        执行部分
# =================================================================

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 设置GPU可见性
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 启用cudnn加速
export CUDNN_BENCHMARK=${CUDNN_BENCHMARK}

# 启动训练任务
echo "[$(date)] 训练启动 | 日志保存至: ${LOG_FILE}" | tee -a ${LOG_FILE}

python -u ${CODE_DIR}/train_cd_model.py \
    ${CONFIG_FILE} \
    --path ${OUTPUT_DIR} \
    --manual_seed ${SEED} \
    --resume \
    --amp ${AMP_TYPE} \
    --last_gamma 0.1 \
    --auto_restart_thresh 0.9 \
    --save_freq 1 \
    --project "dfc25_track2" \
    --name "dfc25_track2_a100_hky2decoder_caformer" \
    --num_workers ${NUM_WORKERS} \
    2>&1 | tee -a ${LOG_FILE}

# 状态码检查
if [ $? -eq 0 ]; then
    echo "[$(date)] 训练正常结束" | tee -a ${LOG_FILE}
else
    echo "[$(date)] 训练异常终止！请检查日志："$(tail -n 20 ${LOG_FILE}) | tee -a ${LOG_FILE}
fi