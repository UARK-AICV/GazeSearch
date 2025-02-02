import sys

sys.path.append("../common")

import json
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.dataset import process_data
from common.utils import get_prior_maps

from .models import FoveatedObjectMemory, HumanAttnTransformer, HumanAttnTransformerV2


def build(hparams, dataset_root, device, is_pretraining, is_eval=False, split=1):
    dataset_name = hparams.Data.name
    data_path = hparams.Data.data_path

    # bounding box of the target object (for search efficiency evaluation)

    bbox_annos = np.load(
        join(dataset_root, "bbox_annos_filled_dummy.npy"), allow_pickle=True
    ).item()

    # load ground-truth expert scanpaths
    with open(join(dataset_root, data_path), "r") as json_file:
        human_scanpaths = [x for x in json.load(json_file) if x["split"] == "train"]
    with open(join(dataset_root, data_path), "r") as json_file:
        human_scanpaths_test = [
            x for x in json.load(json_file) if x["split"] == "valid"
        ]

    # exclude incorrect scanpaths
    if hparams.Data.exclude_wrong_trials:
        human_scanpaths = list(filter(lambda x: x["correct"] == 1, human_scanpaths))

    human_scanpaths_all = human_scanpaths + human_scanpaths_test

    human_scanpaths_tp = list(
        filter(lambda x: x["condition"] == "present", human_scanpaths_all)
    )

    if hparams.Data.subject > -1:
        print(f"excluding subject {hparams.Data.subject} data!")
        human_scanpaths = list(
            filter(lambda x: x["subject"] != hparams.Data.subject, human_scanpaths)
        )

    # process fixation data
    dataset = process_data(
        human_scanpaths,
        dataset_root,
        bbox_annos,
        hparams,
        human_scanpaths_all,
        sample_scanpath=hparams.Train.use_whole_scanpath if is_pretraining else False,
        min_traj_length_percentage=hparams.Train.min_scanpath_length_percentage
        if is_pretraining
        else 0,
        use_coco_annotation="centermap_pred" in hparams.Train.losses and (not is_eval),
    )
    n_tasks = len(dataset["catIds"])
    batch_size = hparams.Train.batch_size
    n_workers = hparams.Train.n_workers

    train_HG_loader = DataLoader(
        dataset["gaze_train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
    )
    print("num of training batches =", len(train_HG_loader))

    train_img_loader = DataLoader(
        dataset["img_train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
    )
    valid_img_loader_TP = DataLoader(
        dataset["img_valid_TP"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=True,
    )

    valid_HG_loader = DataLoader(
        dataset["gaze_valid"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=True,
    )
    valid_HG_loader_TP = DataLoader(
        dataset["gaze_valid_TP"],
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=True,
    )

    # Create model
    emb_size = hparams.Model.embedding_dim
    n_heads = hparams.Model.n_heads
    hidden_size = hparams.Model.hidden_dim
    tgt_vocab_size = hparams.Data.patch_count + len(hparams.Data.special_symbols)
    if hparams.Train.use_sinkhorn:
        assert (
            hparams.Model.separate_fix_arch
        ), "sinkhorn requires the model to be separate!"

    # Interesting, we have 3 models!!!
    if hparams.Model.name == "chestsearch":
        model = HumanAttnTransformer(
            hparams.Data,
            num_decoder_layers=hparams.Model.n_dec_layers,
            hidden_dim=emb_size,
            nhead=n_heads,
            ntask=n_tasks,
            tgt_vocab_size=tgt_vocab_size,
            num_output_layers=hparams.Model.num_output_layers,
            separate_fix_arch=hparams.Model.separate_fix_arch,
            train_encoder=hparams.Train.train_backbone,
            train_pixel_decoder=hparams.Train.train_pixel_decoder,
            use_dino=hparams.Train.use_dino_pretrained_model,
            dropout=hparams.Train.dropout,
            dim_feedforward=hidden_size,
            parallel_arch=hparams.Model.parallel_arch,
            dorsal_source=hparams.Model.dorsal_source,
            num_encoder_layers=hparams.Model.n_enc_layers,
            output_centermap="centermap_pred" in hparams.Train.losses,
            output_saliency="saliency_pred" in hparams.Train.losses,
            output_target_map="target_map_pred" in hparams.Train.losses,
            transfer_learning_setting=hparams.Train.transfer_learn,
            project_queries=hparams.Train.project_queries,
            is_pretraining=is_pretraining,
            output_feature_map_name=hparams.Model.output_feature_map_name,
        )
    elif hparams.Model.name == "FOM":
        model = FoveatedObjectMemory(
            hparams.Data,
            num_decoder_layers=hparams.Model.n_dec_layers,
            hidden_dim=emb_size,
            nhead=n_heads,
            ntask=n_tasks,
            num_output_layers=hparams.Model.num_output_layers,
            train_encoder=hparams.Train.train_backbone,
            dropout=hparams.Train.dropout,
            dim_feedforward=hidden_size,
            num_encoder_layers=hparams.Model.n_enc_layers,
        )
    elif hparams.Model.name == "HATv2":
        model = HumanAttnTransformerV2(
            hparams.Data,
            num_decoder_layers=hparams.Model.n_dec_layers,
            hidden_dim=emb_size,
            nhead=n_heads,
            ntask=n_tasks,
            num_output_layers=hparams.Model.num_output_layers,
            train_encoder=hparams.Train.train_backbone,
            dropout=hparams.Train.dropout,
            dim_feedforward=hidden_size,
            parallel_arch=hparams.Model.parallel_arch,
            dorsal_source=hparams.Model.dorsal_source,
            num_encoder_layers=hparams.Model.n_enc_layers,
            output_centermap="centermap_pred" in hparams.Train.losses,
            output_saliency="saliency_pred" in hparams.Train.losses,
            output_target_map="target_map_pred" in hparams.Train.losses,
        )
    else:
        print(f"No {hparams.Model.name} model implemented!")
        raise NotImplementedError
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams.Train.adam_lr, betas=hparams.Train.adam_betas
    )

    # Load weights from checkpoint when available
    if len(hparams.Model.checkpoint) > 0:
        print(
            f"loading weights from {hparams.Model.checkpoint} in {hparams.Train.transfer_learn} setting."
        )
        ckp = torch.load(join(hparams.Train.log_dir, hparams.Model.checkpoint))
        # weights_new = ckp['model'].copy()
        # for k, v in ckp['model'].items():
        #     if 'stages' in k:
        #         weights_new[k.replace('stages.', '')] = v
        #         weights_new.pop(k)
        # model.load_state_dict(weights_new)
        if hparams.Train.transfer_learn == "none" or hparams.Train.resume_training:
            model.load_state_dict(ckp["model"])
            optimizer.load_state_dict(ckp["optimizer"])
            global_step = ckp["step"]
        else:
            if "fix_ind_emb.weight" in ckp["model"]:
                # Pad or clip the fix_ind_emb to the correct size
                prev_value = ckp["model"]["fix_ind_emb.weight"]
                if prev_value.shape[0] < hparams.Data.max_traj_length:
                    ckp["model"]["fix_ind_emb.weight"] = torch.cat(
                        [
                            prev_value,
                            torch.zeros(
                                hparams.Data.max_traj_length - prev_value.shape[0],
                                prev_value.shape[1],
                            ).to(prev_value.device),
                        ],
                        dim=0,
                    )
                elif prev_value.shape[0] > hparams.Data.max_traj_length:
                    ckp["model"]["fix_ind_emb.weight"] = prev_value[
                        : hparams.Data.max_traj_length
                    ]
                if hparams.Train.transfer_learn == "finetune":
                    if n_tasks > 1:
                        generic_query = ckp["model"]["query_embed.weight"]
                        ckp["model"]["query_embed.weight"] = generic_query.repeat(
                            n_tasks, 1
                        )
                if hparams.Train.project_queries:
                    model.load_state_dict(ckp["model"], strict=False)
                else:
                    # Find shape-mismatched keys
                    weights_new = ckp["model"].copy()
                    for name, param in ckp["model"].items():
                        if (
                            name in model.state_dict()
                            and param.shape != model.state_dict()[name].shape
                        ):
                            print(
                                f"Shape mismatch for {name}: {param.shape} vs {model.state_dict()[name].shape}"
                            )
                            weights_new.pop(name)
                    print("loading model weights:", weights_new.keys())
                    print(model.load_state_dict(weights_new, strict=False))
            else:
                # Load backbone only
                print(model.encoder.load_state_dict(ckp["model"], strict=False))
            global_step = 0
            if hparams.Train.freeze_trained_params:
                for name, param in model.named_parameters():
                    if "fix_ind_emb" not in name and name in ckp["model"]:
                        param.requires_grad = False
    else:
        global_step = 0

    if hparams.Train.parallel:
        model = torch.nn.DataParallel(model)

    bbox_annos = dataset["bbox_annos"]
    human_cdf = dataset["human_cdf"]
    fix_clusters = dataset["fix_clusters"]

    prior_maps_tp = get_prior_maps(
        human_scanpaths_tp, hparams.Data.im_w, hparams.Data.im_h
    )
    keys = list(prior_maps_tp.keys())
    for k in keys:
        prior_maps_tp[k] = torch.tensor(prior_maps_tp.pop(k)).to(device)

    sss_strings = None

    sps_test_tp = list(filter(lambda x: x["split"] == "valid", human_scanpaths_tp))

    is_lasts = [x[5] for x in dataset["gaze_train"].fix_labels]
    term_pos_weight = len(is_lasts) / np.sum(is_lasts) - 1
    print("termination pos weight: {:.3f}".format(term_pos_weight))

    return (
        model,
        optimizer,
        train_HG_loader,
        valid_HG_loader,
        train_img_loader,
        valid_img_loader_TP,
        None,
        None,
        global_step,
        bbox_annos,
        human_cdf,
        fix_clusters,
        prior_maps_tp,
        None,
        None,
        sss_strings,
        valid_HG_loader_TP,
        None,
        None,
        sps_test_tp,
        None,
        None,
        term_pos_weight,
        dataset["catIds"],
    )
