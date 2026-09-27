# encoding: utf-8
import logging

import torch
import torch.nn.functional as F

from .center_loss import CenterLoss
from .triplet_loss import TripletLoss


def _sample_ids(img_names, batch_size):
    if img_names is None:
        return None
    if isinstance(img_names, (list, tuple)):
        names = list(img_names)
    else:
        names = [img_names[i] for i in range(len(img_names))]
    if len(names) != batch_size:
        raise ValueError(
            f"Expected {batch_size} image names, but received {len(names)}."
        )
    return [str(name.item() if hasattr(name, "item") else name) for name in names]


def make_loss(cfg, num_classes, logger=None):
    """Build the standard ReID loss with CDDO self-paced optimization."""
    logger = logger or logging.getLogger("CCL.train")
    sampler = cfg.DATALOADER.SAMPLER
    center_criterion = CenterLoss(
        num_classes=num_classes, feat_dim=2048, use_gpu=False
    )

    if sampler not in ("softmax", "softmax_triplet"):
        raise ValueError(
            "Expected DATALOADER.SAMPLER to be 'softmax' or "
            f"'softmax_triplet', but got {sampler!r}."
        )

    if "triplet" in cfg.MODEL.METRIC_LOSS_TYPE:
        margin = None if cfg.MODEL.NO_MARGIN else cfg.SOLVER.MARGIN
        triplet = TripletLoss(margin)
    elif sampler == "softmax_triplet":
        raise ValueError(
            "MODEL.METRIC_LOSS_TYPE must contain 'triplet' when using "
            "the softmax_triplet sampler."
        )

    warmup_epochs = int(cfg.SOLVER.SPL_WARMUP)
    quantile = float(cfg.SOLVER.CDDO_QUANTILE)
    if not 0.0 < quantile < 1.0:
        raise ValueError("SOLVER.CDDO_QUANTILE must be between 0 and 1.")

    state = {
        "rho": None,
        "epoch_losses": [],
        "sample_losses": {},
        "sample_weights": {},
    }

    def id_loss_per_sample(logits, target):
        if cfg.MODEL.IF_LABELSMOOTH != "on":
            return F.cross_entropy(logits, target, reduction="none")

        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            targets = torch.zeros_like(log_probs).scatter_(
                1, target.unsqueeze(1), 1
            )
            targets = (1.0 - 0.1) * targets + 0.1 / logits.size(1)
        return (-targets * log_probs).sum(dim=1)

    def triplet_loss_per_sample(features, target):
        _, dist_ap, dist_an = triplet(features, target)
        if triplet.margin is None:
            return F.softplus(dist_ap - dist_an)
        return F.relu(dist_ap - dist_an + triplet.margin)

    def combine_branches(values):
        if not isinstance(values, (list, tuple)):
            return values
        if len(values) == 1:
            return values[0]
        return 0.5 * values[0] + 0.5 * torch.stack(values[1:]).mean(dim=0)

    def record_epoch_losses(losses, img_names):
        detached = losses.detach().cpu()
        names = _sample_ids(img_names, detached.numel())
        if names is None:
            state["epoch_losses"].append(detached)
            return
        for name, value in zip(names, detached.tolist()):
            state["sample_losses"].setdefault(name, []).append(value)

    def loss_func(
        score,
        feat,
        target,
        target_cam=None,
        epoch=None,
        index=None,
        img_names=None,
        track_samples=True,
    ):
        del target_cam, index

        scores = list(score) if isinstance(score, (list, tuple)) else [score]
        id_losses = combine_branches(
            [id_loss_per_sample(logits, target) for logits in scores]
        )

        if sampler == "softmax":
            total_loss = id_losses
        else:
            features = list(feat) if isinstance(feat, (list, tuple)) else [feat]
            triplet_losses = combine_branches(
                [triplet_loss_per_sample(item, target) for item in features]
            )
            total_loss = (
                cfg.MODEL.ID_LOSS_WEIGHT * id_losses
                + cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet_losses
            )

        if epoch is not None and track_samples:
            record_epoch_losses(total_loss, img_names)

        if epoch is None or epoch <= warmup_epochs:
            return total_loss.mean()

        rho = state["rho"]
        if rho is None:
            raise RuntimeError(
                "CDDO rho is unavailable. finalize_epoch() must be called at "
                "the end of the final warm-up epoch."
            )

        rho_tensor = total_loss.new_tensor(rho)
        weights = torch.exp(-total_loss / rho_tensor).detach()
        regularizer = rho_tensor * (
            weights - weights * torch.log(weights.clamp_min(1e-12))
        )
        return (weights * total_loss).mean() + regularizer.mean()

    def finalize_epoch(epoch):
        """Finalize global CDDO statistics after all batches in an epoch."""
        if not state["epoch_losses"] and not state["sample_losses"]:
            state["sample_weights"] = {}
            return

        if state["sample_losses"]:
            mean_losses = {
                name: sum(values) / len(values)
                for name, values in state["sample_losses"].items()
            }
            all_losses = torch.tensor(list(mean_losses.values()))
        else:
            mean_losses = {}
            all_losses = torch.cat(state["epoch_losses"])
        if epoch == warmup_epochs:
            state["rho"] = max(
                float(torch.quantile(all_losses, quantile).item()), 1e-12
            )
            logger.info(
                "CDDO fixed rho at the %.0fth percentile of %d losses "
                "from final warm-up epoch %d: %.6f",
                quantile * 100,
                all_losses.numel(),
                epoch,
                state["rho"],
            )

        if mean_losses:
            minimum = min(mean_losses.values())
            maximum = max(mean_losses.values())
            scale = maximum - minimum
            if scale <= 1e-12:
                state["sample_weights"] = {
                    name: 1.0 for name in mean_losses
                }
            else:
                state["sample_weights"] = {
                    name: 1.0 - (value - minimum) / scale
                    for name, value in mean_losses.items()
                }

        state["epoch_losses"].clear()
        state["sample_losses"].clear()

    def get_epoch_weights(epoch=None):
        del epoch
        return state["sample_weights"] or None

    def get_rho():
        return state["rho"]

    loss_func.finalize_epoch = finalize_epoch
    loss_func.get_epoch_weights = get_epoch_weights
    loss_func.get_rho = get_rho
    return loss_func, center_criterion
