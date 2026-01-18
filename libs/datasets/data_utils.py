import os
import copy
import random
import numpy as np
import random
import torch
import torchvision as tv
import itertools

def temporal_iou(pred_seg, gt_seg):
    """Compute temporal IoU between two segments [start, end]"""
    inter_start = max(pred_seg[0], gt_seg[0])
    inter_end = min(pred_seg[1], gt_seg[1])
    inter = max(0.0, inter_end - inter_start)
    union = max(pred_seg[1], gt_seg[1]) - min(pred_seg[0], gt_seg[0])
    return inter / union if union > 0 else 0.0

def build_confusion_matrix(predictions_by_video, groundtruths_by_video, label_map, iou_threshold=0.5):
    num_classes = len(label_map)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for vid_id in groundtruths_by_video:
        if vid_id not in predictions_by_video: continue
        preds = predictions_by_video.get(vid_id, [])
        gts = groundtruths_by_video[vid_id]['annotations']
        pred_flags = [False] * len(preds)  # Track matched predictions

        for gt in gts:
            gt_seg = gt['segment']
            gt_label = label_map[gt['label']]

            best_iou = 0.0
            best_idx = -1

            for i, pred in enumerate(preds):
                if pred_flags[i]:
                    continue
                pred_seg = [float(pred['t-start']), float(pred['t-end'])]
                iou = temporal_iou(pred_seg, gt_seg)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                pred_label = int(preds[best_idx]['label'])  # Or map if needed
                conf_matrix[gt_label][pred_label] += 1
                pred_flags[best_idx] = True
            else:
                # Missed detection
                conf_matrix[gt_label][gt_label] += 0  # Optionally track FN elsewhere

        # Optional: count false positives
        # for i, used in enumerate(pred_flags):
        #     if not used:
        #         pred_label = int(preds[i]['label'])
        #         conf_matrix[bg_idx][pred_label] += 1

    return conf_matrix


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    
    # if len(frames) == 0:
    #     print("The video should save {} frames, but only {} frames are saved. Total number is {}.".format(end_index - start_index + 1, len(frames), i + 1))
    #     return None
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def circ_slice(a, start, length):
    it = itertools.cycle(a)
    next(itertools.islice(it, start, start), None)
    return list(itertools.islice(it, length))


def get_transforms(is_training, size):
    normalize = tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if size == 224:
        resize_dim = 256 # 142 256
        crop_dim = 224 # 128 # 224
    if is_training == True:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim, antialias=True),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim, antialias=True),
                tv.transforms.CenterCrop(crop_dim),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    return transform


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def truncate_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    offset,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = data_dict['feats'].shape[1]
    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for idx in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len

        # corner case: no valid truncation, recalculate st and ed
        if idx == max_num_trials - 1 and seg_idx.sum().item() == 0:
            # re-calculate window based on the first segment
            st = int(torch.minimum(torch.as_tensor(feat_len - max_seq_len - 1, dtype=torch.float32), data_dict['segments'][0, 0]).item())
            ed = st + max_seq_len

        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict



def truncate_video(
    data_dict,
    max_seq_len,
    trunc_thresh,
    offset,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = data_dict['feats'].shape[0]
    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for idx in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len

        # corner case: no valid truncation, recalculate st and ed
        if idx == max_num_trials - 1 and seg_idx.sum().item() == 0:
            # re-calculate window based on the first segment
            st = int(torch.minimum(torch.as_tensor(feat_len - max_seq_len - 1, dtype=torch.float32), data_dict['segments'][0, 0]).item())
            ed = st + max_seq_len

        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict
