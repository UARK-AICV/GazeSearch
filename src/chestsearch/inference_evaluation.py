import sys

sys.path.append("../common")

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from common import metrics, utils


def get_IOR_mask(norm_x, norm_y, h, w, r):
    bs = len(norm_x)
    x, y = norm_x * w, norm_y * h
    Y, X = np.ogrid[:h, :w]
    X = X.reshape(1, 1, w)
    Y = Y.reshape(1, h, 1)
    x = x.reshape(bs, 1, 1)
    y = y.reshape(bs, 1, 1)
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r
    return torch.from_numpy(mask.reshape(bs, -1))


def scanpath_decode(model, img, task_ids, pa, sample_action=False, center_initial=True):
    bs = img.size(0)
    with torch.no_grad():
        dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps = model.encode(img)
    if center_initial:
        normalized_fixs = torch.zeros(bs, 1, 2).fill_(0.5)
        action_mask = get_IOR_mask(
            np.ones(bs) * 0.5, np.ones(bs) * 0.5, pa.im_h, pa.im_w, pa.IOR_radius
        )
    else:
        normalized_fixs = torch.zeros(bs, 0, 2)
        action_mask = torch.zeros(bs, pa.im_h * pa.im_w)

    stop_flags = []
    for i in range(pa.max_traj_length):
        with torch.no_grad():
            if i == 0 and not center_initial:
                ys = ys_high = torch.zeros(bs, 1).to(torch.long)
                padding = torch.ones(bs, 1).bool().to(img.device)
            else:
                ys, ys_high = utils.transform_fixations(
                    normalized_fixs, None, pa, False, return_highres=True
                )
                padding = None

            out = model.decode_and_predict(
                dorsal_embs.clone(),
                dorsal_pos,
                dorsal_mask,
                high_res_featmaps,
                ys.to(img.device),
                padding,
                ys_high.to(img.device),
                task_ids,
            )
            prob, stop = out["pred_fixation_map"], out["pred_termination"]
            prob = prob.view(bs, -1)
            stop_flags.append(stop)

            if pa.enforce_IOR:
                # Enforcing IOR
                batch_idx, visited_locs = torch.where(action_mask == 1)
                prob[batch_idx, visited_locs] = 0

        if sample_action:
            m = Categorical(prob)
            next_word = m.sample()
        else:
            _, next_word = torch.max(prob, dim=1)

        next_word = next_word.cpu()
        norm_fy = (next_word // pa.im_w) / float(pa.im_h)
        norm_fx = (next_word % pa.im_w) / float(pa.im_w)
        normalized_fixs = torch.cat(
            [normalized_fixs, torch.stack([norm_fx, norm_fy], dim=1).unsqueeze(1)],
            dim=1,
        )

        new_mask = get_IOR_mask(
            norm_fx.numpy(), norm_fy.numpy(), pa.im_h, pa.im_w, pa.IOR_radius
        )
        action_mask = torch.logical_or(action_mask, new_mask)

    stop_flags = torch.stack(stop_flags, dim=1)
    # Truncate at terminal action
    trajs = []
    for i in range(normalized_fixs.size(0)):
        is_terminal = stop_flags[i] > 0.5
        if is_terminal.sum() == 0:
            ind = normalized_fixs.size(1)
        else:
            ind = is_terminal.to(torch.int8).argmax().item() + 1
        trajs.append(normalized_fixs[i, :ind])

    nonstop_trajs = [normalized_fixs[i] for i in range(normalized_fixs.size(0))]
    return trajs, nonstop_trajs


def actions2scanpaths(norm_fixs, patch_num, im_h, im_w):
    # convert actions to scanpaths
    scanpaths = []
    for traj in norm_fixs:
        task_name, img_name, condition, fixs = traj
        fixs = fixs.numpy()
        scanpaths.append(
            {
                "X": fixs[:, 0] * im_w,
                "Y": fixs[:, 1] * im_h,
                "name": img_name,
                "task": task_name,
                "condition": condition,
            }
        )
    return scanpaths


def compute_conditional_saliency_metrics(
    pa, model, gazeloader, task_dep_prior_maps, device
):
    n_samples, info_gain, nss, auc = 0, 0, 0, 0
    for batch in tqdm(gazeloader, desc="Computing saliency metrics"):
        img = batch["true_state"].to(device)
        task_ids = batch["task_id"].to(device)
        is_last = batch["is_last"]
        non_term_mask = torch.logical_not(is_last)
        if torch.sum(non_term_mask) == 0:
            continue
        # if pa.include_freeview:
        #     task_ids[batch['is_freeview']] = 18
        inp_seq, inp_seq_high = utils.transform_fixations(
            batch["normalized_fixations"],
            batch["is_padding"],
            pa,
            False,
            return_highres=True,
        )
        inp_seq = inp_seq.to(device)
        inp_padding_mask = inp_seq == pa.pad_idx

        gt_next_fixs = (
            batch["next_normalized_fixations"][:, -1] * torch.tensor([pa.im_w, pa.im_h])
        ).to(torch.long)
        prior_maps = torch.stack(
            [task_dep_prior_maps[task] for task in batch["task_name"]]
        ).cpu()
        with torch.no_grad():
            logits = model(
                img, inp_seq, inp_padding_mask, inp_seq_high.to(device), task_ids
            )
            pred_fix_map = logits["pred_fixation_map"]
            if len(pred_fix_map.size()) > 3:
                pred_fix_map = pred_fix_map[torch.arange(img.size(0)), task_ids]
            pred_fix_map = pred_fix_map.detach().cpu()
            # pred_fix_map = torch.nn.functional.interpolate(
            #     pred_fix_map.unsqueeze(1), size=(pa.im_h, pa.im_w), mode='bilinear').squeeze(1)

            probs = pred_fix_map
            # Normalize values to 0-1
            # probs -= probs.view(probs.size(0), 1, -1).min(dim=-1, keepdim=True)[0]
            probs /= probs.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        probs = probs[non_term_mask]
        prior_maps = prior_maps[non_term_mask]
        gt_next_fixs = gt_next_fixs[non_term_mask]
        info_gain += metrics.compute_info_gain(probs, gt_next_fixs, prior_maps)
        nss += metrics.compute_NSS(probs, gt_next_fixs)
        auc += metrics.compute_cAUC(probs, gt_next_fixs)
        n_samples += gt_next_fixs.size(0)

    info_gain /= n_samples
    nss /= n_samples
    auc /= n_samples

    return info_gain.item(), nss.item(), auc.item()


def sample_scanpaths(model, dataloader, pa, device, sample_action, center_initial=True):
    all_actions, nonstop_actions = [], []
    for i in range(10):
        for batch in tqdm(dataloader, desc=f"Generate scanpaths [{i}/10]:"):
            img = batch["im_tensor"].to(device)
            task_ids = batch["task_id"].to(device)
            img_names_batch = batch["img_name"]
            cat_names_batch = batch["cat_name"]
            cond_batch = batch["condition"]
            trajs, nonstop_trajs = scanpath_decode(
                model.module if isinstance(model, torch.nn.DataParallel) else model,
                img,
                task_ids,
                pa,
                sample_action,
                center_initial,
            )
            nonstop_actions.extend(
                [
                    (
                        cat_names_batch[i],
                        img_names_batch[i],
                        cond_batch[i],
                        nonstop_trajs[i],
                    )
                    for i in range(img.size(0))
                ]
            )

            all_actions.extend(
                [
                    (cat_names_batch[i], img_names_batch[i], cond_batch[i], trajs[i])
                    for i in range(img.size(0))
                ]
            )

        if not sample_action:
            break

    scanpaths = actions2scanpaths(all_actions, pa.patch_num, pa.im_h, pa.im_w)
    return scanpaths, nonstop_actions


def evaluate(
    model,
    device,
    valid_img_loader,
    gazeloader,
    pa,
    sample_action=True,
    center_initial=True,
    log_dir=None,
):
    print(
        "Eval on {} batches of images and {} batches of fixations".format(
            len(valid_img_loader), len(gazeloader)
        )
    )
    model.eval()

    scanpaths, nonstop_actions = sample_scanpaths(
        model, valid_img_loader, pa, device, sample_action, center_initial
    )

    return scanpaths
