import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, num_classes, ppp=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    warmup=cfg.SOLVER.SPL_WARMUP
    is_text=cfg.MODEL.isText
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("CCL.train")
    logger.info('start training')
    if not hasattr(loss_fn, "finalize_epoch"):
        from layers.make_loss import make_loss
        loss_fn, _ = make_loss(cfg, num_classes=num_classes, logger=logger)
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM
    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        
        if is_text and epoch > warmup and hasattr(loss_fn, "get_epoch_weights"):
            previous_weights = loss_fn.get_epoch_weights(epoch - 1)
            model_to_update = model.module if hasattr(model, "module") else model
            if hasattr(model_to_update, "current_epoch_weights"):
                model_to_update.current_epoch_weights = previous_weights
        for n_iter, (img, vid, target_cam, target_view, img_path, text) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            text = {'rgb_text': text['rgb_text'].to(device),
                    'ni_text': text['ni_text'].to(device),
                    'ti_text': text['ti_text'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output = model(
                    img, text=text, label=target, cam_label=target_cam,
                    view_label=target_view, epoch=epoch, img_path=img_path,
                    ppp=ppp if ppp is not None else 1
                )
                loss = 0
                paired_outputs = len(output) - (len(output) % 2)
                for i in range(0, paired_outputs, 2):
                    loss = loss + loss_fn(
                        score=output[i], feat=output[i + 1], target=target,
                        target_cam=target_cam, epoch=epoch, index=ppp,
                        img_names=img_path
                    )
                if len(output) % 2 == 1:
                    loss = loss + output[-1]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))



        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        
        if hasattr(loss_fn, "finalize_epoch"):
            loss_fn.finalize_epoch(epoch)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger)
            else:
                if test_sign:
                    _, _ = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                              return_pattern=1)
                    _, _ = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                              return_pattern=2)
                mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                              return_pattern=3)
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    best_name = (
                        f"{cfg.MODEL.NAME}_{ppp}_best.pth"
                        if ppp is not None else f"{cfg.MODEL.NAME}_best.pth"
                    )
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, best_name))
                logger.info("~" * 50)
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                logger.info("~" * 50)
    return best_index


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, return_pattern=1):
    device = "cuda"
    logger = logging.getLogger("CCL.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)
    
    for img, pid, camid, camids, target_view, imgpath, text in val_loader:
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            text = {'rgb_text': text['rgb_text'].to(device),
                    'ni_text': text['ni_text'].to(device),
                    'ti_text': text['ti_text'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, text,cam_label=camids, view_label=target_view, return_pattern=return_pattern, img_path=imgpath)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator, epoch, logger, return_pattern=1):
    evaluator.reset()
    model.eval()
    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)
    for n_iter, (img, vid, camid, camids, target_view, _,text) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            text = {'rgb_text': text['rgb_text'].to(device),
                    'ni_text': text['ni_text'].to(device),
                    'ti_text': text['ti_text'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, text,cam_label=camids, view_label=target_view, return_pattern=return_pattern)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, scenceids, _))
            else:
                evaluator.update((feat, vid, camid, _))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("~" * 50)
    torch.cuda.empty_cache()
    return mAP, cmc
