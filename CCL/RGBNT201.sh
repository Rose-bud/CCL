#!/bin/bash

# ==========================================
# 用户需要修改的路径配置
# ==========================================
# 你的主训练 python 脚本名称
TRAIN_SCRIPT="train_seeds1.py" 

# 你的配置文件路径
CONFIG_FILE="/Lsy/IDEA/IDEA-main_2/DeMo-master_7best_1/configs/RGBNT201/DeMo.yml" 

# 基础输出目录
BASE_OUT_DIR="./loss/ablation_study"

# ==========================================
# 循环执行 3 种加权策略
# Strategy 1: CDDO / Standard SPL (你的方案)
# Strategy 2: Inverse Weighting (逆序加权)
# Strategy 3: Focal Loss Style (静态鲁棒目标)
# ==========================================

for strategy in 1 2 3
do
    echo "====================================================================="
    echo "🚀 [Start] Running Experiment with SOLVER.WEIGHT_STRATEGY = ${strategy}"
    echo "====================================================================="
    
    # 为不同策略创建不同的输出文件夹
    CURRENT_OUT_DIR="${BASE_OUT_DIR}/strategy_${strategy}"
    
    # 执行训练代码
    python ${TRAIN_SCRIPT} \
        --config_file ${CONFIG_FILE} \
        SOLVER.WEIGHT_STRATEGY ${strategy} \
        OUTPUT_DIR ${CURRENT_OUT_DIR}
        
    echo "✅ [Finished] Experiment with SOLVER.WEIGHT_STRATEGY = ${strategy} completed."
    echo "Logs and checkpoints are saved in: ${CURRENT_OUT_DIR}"
    echo -e "\n\n"
done

echo "🎉 All ablation experiments are done!"