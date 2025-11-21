"""
ChestSearch Training Script for Radiology Findings Search.
This script trains the ChestSearch model on the GazeSearch dataset.
Based on the HAT training framework from https://github.com/cvlab-stonybrook/Scanpath_Prediction
"""
import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.chestsearch.builder import build
from src.chestsearch.evaluation import evaluate
from src.common.config import JsonConfig
from src.common.losses import focal_loss
from src.common.utils import transform_fixations

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ChestSearch model for radiology findings search')
    parser.add_argument('--hparams',
                        type=str,
                        default='src/configs/finding_search_box_mask_max6_split_shuffled.json',
                        help='Path to hyperparameters config file')
    parser.add_argument('--dataset-root',
                        type=str,
                        default='./data',
                        help='Root directory of the dataset')
    parser.add_argument('--eval-only',
                        action='store_true',
                        help='Perform evaluation only without training')
    parser.add_argument('--split',
                        type=int,
                        default=1,
                        help='Dataset split (default=1)')
    parser.add_argument('--eval-mode',
                        choices=['greedy', 'sample'],
                        type=str,
                        default='greedy',
                        help='Evaluation mode: greedy or sample scanpath (default=greedy)')
    parser.add_argument('--disable-saliency',
                        action='store_true',
                        help='Disable saliency metrics computation')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='GPU device ID (default=0)')
    return parser.parse_args()


def log_dict(writer, scalars, step, prefix):
    """Log scalar values to tensorboard."""
    for k, v in scalars.items():
        writer.add_scalar(prefix + "/" + k, v, step)


def compute_loss(model, batch, losses, loss_funcs, pa):
    """Compute loss for a training batch."""
    img = batch['true_state'].to(device)
    task_ids = batch['task_id'].to(device)
    is_last = batch['is_last'].to(device)
    IOR_weight_map = batch['IOR_weight_map'].to(device)

    # Transform fixations to input sequences
    inp_seq, inp_seq_high = transform_fixations(
        batch['normalized_fixations'],
        batch['is_padding'],
        hparams.Data,
        False,
        return_highres=True
    )
    inp_seq = inp_seq.to(device)
    inp_padding_mask = (inp_seq == pa.pad_idx)

    logits = model(img, inp_seq, inp_padding_mask, inp_seq_high.to(device), task_ids)

    bs = img.size(0)
    loss_dict = {}
    is_all_task = len(logits['pred_fixation_map'].size()) > 3

    # Next fixation prediction loss
    if "next_fix_pred" in losses:
        non_term_mask = torch.logical_not(is_last)

        if is_all_task:
            pred_fix_map = logits['pred_fixation_map'][torch.arange(bs), task_ids]
        else:
            pred_fix_map = logits['pred_fixation_map']

        if use_focal_loss:
            pred_fix_map = torch.sigmoid(pred_fix_map)

        if pred_fix_map.size(-1) != pa.im_w:
            pred_fix_map = F.interpolate(
                pred_fix_map.unsqueeze(1),
                size=(pa.im_h, pa.im_w)
            ).squeeze(1)

        tgt_fix_map = batch['target_fix_map'].to(device)
        pred_fix_map = pred_fix_map[non_term_mask]
        tgt_fix_map = tgt_fix_map[non_term_mask]

        loss_dict['next_fix_pred'] = loss_funcs['next_fix_pred'](
            pred_fix_map,
            tgt_fix_map,
            alpha=1,
            beta=4,
            weights=IOR_weight_map[non_term_mask]
        )

    # Termination prediction loss
    if "term_pred" in losses:
        if is_all_task:
            pred_termination = logits['pred_termination'][torch.arange(bs), task_ids]
        else:
            pred_termination = logits['pred_termination']
        loss_dict['term_pred'] = loss_funcs['term_pred'](
            pred_termination, is_last.float()
        )

    # Centermap prediction loss
    if "centermap_pred" in losses:
        pred_cm_map = torch.sigmoid(logits['pred_centermap'])
        tgt_cm_map = batch['centermaps'].to(device)
        pred_cm_map = F.interpolate(pred_cm_map, size=(pa.im_h, pa.im_w))
        loss_dict['centermap_pred'] = loss_funcs['centermap_pred'](
            pred_cm_map, tgt_cm_map
        )

    # Target map prediction loss
    if "target_map_pred" in losses:
        pred_target_map = torch.sigmoid(
            logits['pred_target_map'][torch.arange(bs), task_ids]
        )
        tgt_target_map = batch['label_coding'].to(device)
        pred_target_map = F.interpolate(
            pred_target_map.unsqueeze(1),
            size=(pa.im_h, pa.im_w)
        ).squeeze(1)
        loss_dict['target_map_pred'] = loss_funcs['target_map_pred'](
            pred_target_map, tgt_target_map
        )

    # Saliency prediction loss
    if "saliency_pred" in losses:
        pred_sal_map = logits['pred_saliency']
        tgt_sal_map = batch['saliency_map'].to(device)
        tgt_sal_map = F.interpolate(tgt_sal_map, size=pred_sal_map.shape[-2:])
        if pred_sal_map.size(1) > 1:
            tgt_sal_map = tgt_sal_map.repeat(1, pred_sal_map.size(1), 1, 1)
        loss_dict['saliency_pred'] = loss_funcs['saliency_pred'](
            pred_sal_map.squeeze(), tgt_sal_map.squeeze()
        )

    return loss_dict


def train_iter(model, optimizer, batch, losses, loss_weights, loss_funcs, pa):
    """Perform one training iteration."""
    assert len(losses) > 0, "No loss function assigned!"
    model.train()
    optimizer.zero_grad()

    loss_dict = compute_loss(model, batch, losses, loss_funcs, pa)

    # Compute weighted total loss
    loss = 0
    for k, v in loss_dict.items():
        loss += v * loss_weights[k]

    loss.backward()
    optimizer.step()

    # Convert losses to scalars
    for k in loss_dict:
        loss_dict[k] = loss_dict[k].item()

    return loss_dict


def get_eval_loss(model, eval_dataloader, losses, loss_funcs, pa):
    """Compute evaluation loss."""
    with torch.no_grad():
        model.eval()
        num_batches = 0
        avg_loss_dict = defaultdict(lambda: 0)

        for batch in tqdm(eval_dataloader, desc="Computing eval loss"):
            loss_dict = compute_loss(model, batch, losses, loss_funcs, pa)
            for k in loss_dict:
                avg_loss_dict[k] += loss_dict[k].item()
            num_batches += 1
            if num_batches > 1000:
                break

        for k in avg_loss_dict:
            avg_loss_dict[k] /= num_batches

        return avg_loss_dict


def run_evaluation():
    """Perform evaluation on validation set."""
    rst_tp = None
    pred_tp = None

    if hparams.Data.TAP in ['TP']:
        rst_tp, pred_tp = evaluate(
            model,
            device,
            valid_img_loader_tp,
            valid_gaze_loader_tp,
            hparams_tp.Data,
            bbox_annos,
            human_cdf,
            fix_clusters,
            prior_maps_tp,
            sss_strings,
            dataset_root,
            sps_test_tp,
            sample_action=sample_action,
            output_saliency_metrics=output_saliency_metrics,
            center_initial=True,
            log_dir=log_dir
        )
        print("TP Results:", rst_tp)

    return rst_tp, None, None


if __name__ == '__main__':
    args = parse_args()

    # Load hyperparameters
    hparams = JsonConfig(args.hparams)
    hparams.Model.name = 'chestsearch'
    hparams.Train.transfer_learn = 'none'

    # Load evaluation config
    dir = os.path.dirname(args.hparams)
    hparams_tp = JsonConfig(os.path.join(dir, 'eval_finding_search_box_mask_max6_split.json'))

    # Setup paths
    dataset_root = args.dataset_root
    if dataset_root[-1] == '/':
        dataset_root = dataset_root[:-1]

    output_saliency_metrics = not args.disable_saliency
    device = torch.device(f'cuda:{args.gpu_id}')
    sample_action = args.eval_mode == 'sample'

    # Build model and data loaders
    print("Building model and data loaders...")
    (model, optimizer, train_gaze_loader, val_gaze_loader, train_img_loader,
     valid_img_loader_tp, _, _,
     global_step, bbox_annos, human_cdf, fix_clusters, prior_maps_tp,
     _, _, sss_strings, valid_gaze_loader_tp,
     _, _, sps_test_tp,
     _, _, term_pos_weight, catIds) = build(
        hparams, dataset_root, device, False, args.eval_only, args.split
    )

    print(f"Model: ChestSearch")
    print(f"Number of training categories: {len(catIds)}")
    print(f"Starting from global step: {global_step}")

    # Evaluation-only mode
    if args.eval_only:
        print("Running evaluation only...")
        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_dir = hparams.Train.log_dir + '|eval_{}'.format(current_time_str)
        os.makedirs(log_dir, exist_ok=True)
        run_evaluation()
    else:
        # Training mode
        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_dir = hparams.Train.log_dir + '|{}'.format(current_time_str)

        # Setup tensorboard
        writer = SummaryWriter(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print("Log dir:", log_dir)

        log_folder_runs = "./runs/{}".format(log_dir.split('/')[-1])
        if not os.path.exists(log_folder_runs):
            os.makedirs(log_folder_runs)
        print("Log runs:", log_folder_runs)

        # Save configuration
        hparams.dump(log_dir, 'config.json')

        # Training parameters
        print_every = 20
        max_iters = hparams.Train.max_iters
        save_every = hparams.Train.checkpoint_every
        eval_every = hparams.Train.evaluate_every
        pad_idx = hparams.Data.pad_idx
        use_focal_loss = hparams.Train.use_focal_loss

        # Loss functions
        loss_funcs = {
            "next_fix_pred": focal_loss if use_focal_loss else torch.nn.BCEWithLogitsLoss(),
            "centermap_pred": focal_loss,
            "target_map_pred": focal_loss,
            "saliency_pred": torch.nn.BCEWithLogitsLoss(),
            "task_pred": torch.nn.CrossEntropyLoss(),
            "term_pred": torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(term_pos_weight, dtype=torch.float32)
            ),
        }

        # Loss weights
        loss_weights = {
            "next_fix_pred": 1.0,
            "centermap_pred": hparams.Train.centermap_pred_weight,
            "target_map_pred": hparams.Train.centermap_pred_weight,
            "saliency_pred": hparams.Train.saliency_pred_weight,
            "task_pred": hparams.Train.task_pred_weight,
            "term_pred": hparams.Train.term_pred_weight,
        }

        losses = hparams.Train.losses
        loss_dict_avg = dict(zip(losses, [0] * len(losses)))
        print("Loss weights:", loss_weights)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=hparams.Train.lr_steps, gamma=0.1
        )

        s_epoch = int(global_step / len(train_gaze_loader))
        last_time = datetime.datetime.now()

        # Training loop
        print(f"Starting training from epoch {s_epoch}...")
        for i_epoch in range(s_epoch, int(1e5)):
            scheduler.step()

            for i_batch, batch in enumerate(train_gaze_loader):
                loss_dict = train_iter(
                    model, optimizer, batch, losses,
                    loss_weights, loss_funcs, hparams.Data
                )

                for k in loss_dict:
                    loss_dict_avg[k] += loss_dict[k]

                # Print training stats
                if global_step % print_every == print_every - 1:
                    for k in loss_dict_avg:
                        loss_dict_avg[k] /= print_every

                    time = datetime.datetime.now()
                    eta = str((time - last_time) / print_every *
                             (max_iters - global_step))
                    last_time = time
                    time = str(time)

                    log_msg = "[{}], eta: {}, iter: {}, progress: {:.2f}%, epoch: {}, total loss: {:.3f}".format(
                        time[time.rfind(' ') + 1:time.rfind('.')],
                        eta[:eta.rfind('.')],
                        global_step,
                        (global_step / max_iters) * 100,
                        i_epoch,
                        np.sum(list(loss_dict_avg.values())),
                    )

                    for k, v in loss_dict_avg.items():
                        log_msg += " {}_loss: {:.3f}".format(k, v)

                    print(log_msg)
                    log_dict(writer, loss_dict_avg, global_step, 'train')
                    writer.add_scalar('train/lr',
                                    optimizer.param_groups[0]["lr"],
                                    global_step)

                    for k in loss_dict_avg:
                        loss_dict_avg[k] = 0

                # Evaluate
                if global_step % eval_every == eval_every - 1:
                    rst_tp, rst_ta, rst_fv = run_evaluation()
                    if rst_tp is not None:
                        log_dict(writer, rst_tp, global_step, "eval_TP")

                    writer.add_scalar('train/epoch',
                                    global_step / len(train_gaze_loader),
                                    global_step)
                    os.system(f"cp {log_dir}/events* {log_folder_runs}")

                # Save checkpoint
                if global_step % save_every == save_every - 1:
                    save_path = os.path.join(log_dir, f"ckp_{global_step}.pt")

                    if isinstance(model, torch.nn.DataParallel):
                        model_weights = model.module.state_dict()
                    else:
                        model_weights = model.state_dict()

                    torch.save(
                        {
                            'model': model_weights,
                            'optimizer': optimizer.state_dict(),
                            'step': global_step + 1,
                        },
                        save_path,
                    )
                    print(f"Saved checkpoint to {save_path}.")

                global_step += 1

                if global_step >= max_iters:
                    print("Reached max iterations. Exiting training!")
                    break
            else:
                continue
            break  # Break outer loop

        # Copy log files
        os.system(f"cp {log_dir}/events* {log_folder_runs}")
        print("Training complete!")
