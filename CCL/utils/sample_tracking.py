"""
样本追踪工具函数
用于保存和加载样本追踪数据
"""
import json
import os
import logging

logger = logging.getLogger("DeMo.tracking")

def save_tracking_data(epoch, data, selected_paths, output_dir="logs/sample_tracking"):
    """
    保存epoch的样本追踪数据到JSON文件
    
    Args:
        epoch: 当前epoch数
        data: 收集的样本数据字典 {img_path: {loss, w_diff, w, pid, camid}}
        selected_paths: 固定的1000个样本路径列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备保存的数据
    save_dict = {
        "epoch": epoch,
        "total_samples": len(data),
        "selected_samples_count": len(selected_paths) if selected_paths else 0,
        "sample_paths": selected_paths if selected_paths else list(data.keys()),
        "samples": {}
    }
    
    # 填充样本数据
    if selected_paths:
        # 第10, 15, 20...轮：可能检索不到所有样本
        for path in selected_paths:
            if path in data:
                save_dict["samples"][path] = data[path]
            else:
                # 检索不到的样本，字段留空
                save_dict["samples"][path] = {
                    "total_loss": None,
                    "w_diff": None,
                    "w": None,
                    "pid": None,
                    "camid": None
                }
    else:
        # 第5轮：保存所有收集的样本
        save_dict["samples"] = data
    
    # 保存JSON文件
    json_path = os.path.join(output_dir, f"train_epoch{epoch}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved tracking data for epoch {epoch} to {json_path}")
    logger.info(f"  Total samples in dict: {len(data)}")
    logger.info(f"  Samples with data: {sum(1 for v in save_dict['samples'].values() if v['total_loss'] is not None)}")
    
    return json_path


def load_tracking_data(epoch, input_dir="logs/sample_tracking"):
    """
    加载指定epoch的样本追踪数据
    
    Args:
        epoch: epoch数
        input_dir: 输入目录
        
    Returns:
        dict: 追踪数据字典，如果文件不存在返回None
    """
    json_path = os.path.join(input_dir, f"train_epoch{epoch}.json")
    
    if not os.path.exists(json_path):
        logger.warning(f"Tracking data file not found: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded tracking data for epoch {epoch} from {json_path}")
    return data


def get_sample_list_for_visualization(epoch=5, input_dir="logs/sample_tracking"):
    """
    获取用于可视化的样本路径列表
    
    Args:
        epoch: epoch数（默认从第5轮获取）
        input_dir: 输入目录
        
    Returns:
        list: 样本路径列表
    """
    data = load_tracking_data(epoch, input_dir)
    if data is None:
        return []
    
    return data.get("sample_paths", [])
