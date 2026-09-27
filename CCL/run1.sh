#!/bin/bash

# ================= 配置区域 =================
PYTHON_EXEC="python"
TRAIN_SCRIPT="train_seeds1.py"
# 请确保将此处的 .yml 替换为你实际的配置文件名
CONFIG_FILE="/Lsy/IDEA/IDEA-main_2/DeMo-master_7best_1/configs/MSVR310/DeMo.yml" 
SUBSET_DIR="/Lsy/IDEA/IDEA-main_2/dataset_splits_301"

# 定义要遍历的相似度文件和文本模态开关
SIM_FILES=("high_similarity.txt" "medium_similarity.txt" "low_similarity.txt")
IS_TEXT_FLAGS=("True")
# ============================================

for is_text in "${IS_TEXT_FLAGS[@]}"; do
    for sim_file in "${SIM_FILES[@]}"; do
        
        # 提取 high, medium, low 用于命名
        SIM_PREFIX=$(echo $sim_file | cut -d'_' -f1) 
        
        # 动态生成隔离的输出目录，例如: ./RGBNT201_Ablation/isText_True_high
        CURRENT_OUTPUT_DIR="./RGBNT201_Ablation/isText_${is_text}_${SIM_PREFIX}"
        SUBSET_PATH="${SUBSET_DIR}/${sim_file}"

        echo "======================================================================"
        echo "Starting Experiment | isText: ${is_text} | Subset: ${SIM_PREFIX}"
        echo "Data Path: ${SUBSET_PATH}"
        echo "Output Target: ${CURRENT_OUTPUT_DIR}"
        echo "======================================================================"

        # 执行训练
        # 命令行末尾的 KEY VALUE 会被 cfg.merge_from_list(args.opts) 解析并覆盖配置
        $PYTHON_EXEC $TRAIN_SCRIPT \
            --config_file "$CONFIG_FILE" \
            --train_subset_txt "$SUBSET_PATH" \
            OUTPUT_DIR "$CURRENT_OUTPUT_DIR" \
            MODEL.isText "$is_text" \
            
    done
done

echo "All 6 ablation experiments completed successfully!"