from os.path import join

import numpy as np
from torchvision import transforms

from .data import FFN_IRL, FFN_Human_Gaze, SPTrans_Human_Gaze, VDTrans_Human_Gaze
from .utils import compute_search_cdf, preprocess_fixations


def process_data(
    target_trajs,
    dataset_root,
    target_annos,
    hparams,
    target_trajs_all,
    is_testing=False,
    sample_scanpath=False,
    min_traj_length_percentage=0,
    use_coco_annotation=False,
    out_of_subject_eval=False,
):
    print(
        "using",
        hparams.Train.repr,
        "dataset:",
        hparams.Data.name,
        "TAP:",
        hparams.Data.TAP,
    )
    coco_annos = None

    size = (hparams.Data.im_h, hparams.Data.im_w)
    transform_train = transforms.Compose(
        [
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_target_trajs_all = list(
        filter(lambda x: x["split"] == "valid", target_trajs_all)
    )

    # Thang: we may need this
    #
    # if hparams.Data.name == 'OSIE' and hparams.Data.im_h == 600:
    #     fix_clusters = np.load(f'{dataset_root}/clusters_600x800.npy',
    #                            allow_pickle=True).item()
    # else: # 320x512
    #     fix_clusters = np.load(f'{dataset_root}/clusters.npy',
    #                            allow_pickle=True).item()
    # for _, v in fix_clusters.items():
    #     if isinstance(v['strings'], list):
    #         break
    #     # remove other subjects' data if "subject" is specified
    #     if hparams.Data.subject > -1:
    #         try:
    #             v['strings'] = [v['strings'][hparams.Data.subject]]
    #         except:
    #             v['strings'] = []
    #     else:
    #         v['strings'] = list(v['strings'].values())
    is_coco_dataset = True
    # if is_coco_dataset:
    #     scene_labels = np.load(f'{dataset_root}/scene_label_dict.npy',
    #                            allow_pickle=True).item()
    # else:
    #     scene_labels = None
    scene_labels = None

    target_init_fixs = {}
    for traj in target_trajs_all:
        key = traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
        if is_coco_dataset:
            # Force center initialization for COCO-Search18
            target_init_fixs[key] = (0.5, 0.5)
        else:
            target_init_fixs[key] = (
                traj["X"][0] / hparams.Data.im_w,
                traj["Y"][0] / hparams.Data.im_h,
            )
    if hparams.Train.zero_shot:
        catIds = np.load(
            join(dataset_root, "all_target_ids.npy"), allow_pickle=True
        ).item()
    else:
        cat_names = list(np.unique([x["task"] for x in target_trajs]))
        catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    human_mean_cdf = None
    if is_testing:
        # testing fixation data
        test_target_trajs = list(
            filter(lambda x: x["split"] == "valid", target_trajs_all)
        )
        assert len(test_target_trajs) > 0, "no testing data found!"
        test_task_img_pair = np.unique(
            [
                traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
                for traj in test_target_trajs
            ]
        )

        # print statistics
        traj_lens = list(map(lambda x: x["length"], test_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print(
            "average train scanpath length : {:.3f} (+/-{:.3f})".format(
                avg_traj_len, std_traj_len
            )
        )
        print("num of train trajs = {}".format(len(test_target_trajs)))

        if hparams.Data.TAP == "TP":
            human_mean_cdf, _ = compute_search_cdf(
                test_target_trajs, target_annos, hparams.Data.max_traj_length
            )
            print("target fixation prob (test).:", human_mean_cdf)

        # load image data

        test_img_dataset = FFN_IRL(
            dataset_root,
            target_init_fixs,
            test_task_img_pair,
            target_annos,
            transform_test,
            hparams.Data,
            catIds,
        )

        return {
            "catIds": catIds,
            "img_test": test_img_dataset,
            "bbox_annos": target_annos,
            "gt_scanpaths": test_target_trajs,
            "fix_clusters": None,
        }

    else:
        # training fixation data
        train_target_trajs = list(filter(lambda x: x["split"] == "train", target_trajs))
        # print statistics
        traj_lens = list(map(lambda x: x["length"], train_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print(
            "average train scanpath length : {:.3f} (+/-{:.3f})".format(
                avg_traj_len, std_traj_len
            )
        )
        print("num of train trajs = {}".format(len(train_target_trajs)))

        train_task_img_pair = np.unique(
            [
                traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
                for traj in train_target_trajs
            ]
        )
        train_fix_labels = preprocess_fixations(
            train_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            # truncate_num=hparams.Data.max_traj_length,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            min_traj_length_percentage=min_traj_length_percentage,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )

        # validation fixation data
        valid_target_trajs = list(
            filter(lambda x: x["split"] == "valid", target_trajs_all)
        )
        # print statistics
        traj_lens = list(map(lambda x: x["length"], valid_target_trajs))
        avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
        print(
            "average valid scanpath length : {:.3f} (+/-{:.3f})".format(
                avg_traj_len, std_traj_len
            )
        )
        print("num of valid trajs = {}".format(len(valid_target_trajs)))

        valid_task_img_pair = np.unique(
            [
                traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
                for traj in valid_target_trajs
            ]
        )

        if hparams.Data.TAP in ["TP", "TAP"]:
            tp_trajs = list(
                filter(
                    lambda x: x["condition"] == "present" and x["split"] == "valid",
                    target_trajs_all,
                )
            )
            human_mean_cdf, _ = compute_search_cdf(
                tp_trajs, target_annos, hparams.Data.max_traj_length
            )
            print("target fixation prob (valid).:", human_mean_cdf)

        valid_fix_labels = preprocess_fixations(
            valid_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            min_traj_length_percentage=min_traj_length_percentage,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )
        valid_target_trajs_TP = list(
            filter(lambda x: x["condition"] == "present", valid_target_trajs_all)
        )
        valid_fix_labels_TP = preprocess_fixations(
            valid_target_trajs_TP,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            min_traj_length_percentage=min_traj_length_percentage,
            discretize_fix=hparams.Data.discretize_fix,
            is_coco_dataset=is_coco_dataset,
        )

        valid_fix_labels_all = preprocess_fixations(
            valid_target_trajs_all,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath,
            min_traj_length_percentage=min_traj_length_percentage,
            discretize_fix=hparams.Data.discretize_fix,
            remove_return_fixations=hparams.Data.remove_return_fixations,
            is_coco_dataset=is_coco_dataset,
        )
        valid_task_img_pair_TP = np.unique(
            [
                traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
                for traj in valid_target_trajs_all
                if traj["condition"] == "present"
            ]
        )
        valid_task_img_pair_all = np.unique(
            [
                traj["task"] + "*" + traj["name"] + "*" + traj["condition"]
                for traj in valid_target_trajs_all
            ]
        )

        # load image data
        train_img_dataset = FFN_IRL(
            dataset_root,
            None,
            train_task_img_pair,
            target_annos,
            transform_train,
            hparams.Data,
            catIds,
            coco_annos=coco_annos,
        )
        valid_img_dataset_all = FFN_IRL(
            dataset_root,
            None,
            valid_task_img_pair_all,
            target_annos,
            transform_test,
            hparams.Data,
            catIds,
            coco_annos=None,
        )
        valid_img_dataset_TP = FFN_IRL(
            dataset_root,
            None,
            valid_task_img_pair_TP,
            target_annos,
            transform_test,
            hparams.Data,
            catIds,
            coco_annos=None,
        )

        if hparams.Model.name == "FoveatedFeatureNet":
            gaze_dataset_func = FFN_Human_Gaze
        elif hparams.Model.name == "VDTransformer":
            gaze_dataset_func = VDTrans_Human_Gaze
        elif (
            hparams.Model.name in ["chestsearch", "FOM", "HATv2"]
        ):  # interesting!! FFN = FoveatedFeatureNet. This work is based on so many other works.
            gaze_dataset_func = SPTrans_Human_Gaze
        else:
            raise NotImplementedError

        train_HG_dataset = gaze_dataset_func(
            dataset_root,
            train_fix_labels,
            target_annos,
            scene_labels,
            hparams.Data,
            transform_train,
            catIds,
            blur_action=True,
            coco_annos=coco_annos,
        )
        valid_HG_dataset = gaze_dataset_func(
            dataset_root,
            valid_fix_labels,
            target_annos,
            scene_labels,
            hparams.Data,
            transform_test,
            catIds,
            blur_action=True,
            coco_annos=None,
        )
        valid_HG_dataset_TP = gaze_dataset_func(
            dataset_root,
            valid_fix_labels_TP,
            target_annos,
            scene_labels,
            hparams.Data,
            transform_test,
            catIds,
            blur_action=True,
            coco_annos=None,
        )

        valid_HG_dataset_all = gaze_dataset_func(
            dataset_root,
            valid_fix_labels_all,
            target_annos,
            scene_labels,
            hparams.Data,
            transform_test,
            catIds,
            blur_action=True,
            coco_annos=None,
        )
        # if hparams.Data.TAP == ['TP', 'TAP']:
        #     cutFixOnTarget(target_trajs, target_annos)
        print(
            "num of training and eval fixations = {}, {}".format(
                len(train_HG_dataset), len(valid_HG_dataset)
            )
        )
        print(
            "num of training and eval images = {}, {} (TP), {} (TA), {}(FV)".format(
                len(train_img_dataset), len(valid_img_dataset_TP), len([]), len([])
            )
        )

        return {
            "catIds": catIds,
            "img_train": train_img_dataset,
            "img_valid_TP": valid_img_dataset_TP,
            "img_valid_TA": None,
            "img_valid_FV": None,
            "img_valid": valid_img_dataset_all,
            "gaze_train": train_HG_dataset,
            "gaze_valid": valid_HG_dataset,
            "gaze_valid_TP": valid_HG_dataset_TP,
            "gaze_valid_TA": None,
            "gaze_valid_FV": None,
            "bbox_annos": target_annos,
            "fix_clusters": None,
            "valid_scanpaths": valid_target_trajs_all,
            "human_cdf": human_mean_cdf,
        }
