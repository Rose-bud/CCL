import argparse
import os

from config import cfg
from data.datasets.make_dataloader_text import make_dataloader
from engine.processor import do_inference
from modeling import make_model
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCL Testing")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--weight", default="", help="path to the trained checkpoint", type=str
    )
    parser.add_argument(
        "opts", help="Modify config options using the command-line",
        default=None, nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("CCL", output_dir, if_train=False)
    logger.info(args)
    if args.config_file:
        logger.info("Loaded configuration file %s", args.config_file)
        with open(args.config_file, "r", encoding="utf-8") as config_file:
            logger.info("\n%s", config_file.read())
    logger.info("Running with config:\n%s", cfg)

    if not (args.weight or cfg.TEST.WEIGHT):
        raise ValueError("Set --weight or TEST.WEIGHT to a trained checkpoint.")
    weight_path = os.path.abspath(os.path.expanduser(args.weight or cfg.TEST.WEIGHT))
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    _, _, val_loader, num_query, num_classes, camera_num, view_num = (
        make_dataloader(cfg)
    )
    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num
    )

    logger.info("Loading checkpoint from %s", weight_path)
    model.load_param(weight_path)
    model.eval()
    do_inference(cfg, model, val_loader, num_query, return_pattern=3)
