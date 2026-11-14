import os
import shutil
import time
import json
import pickle
from typing import Dict

import numpy as np
from functools import reduce

import torch

from .metrics import ANETdetection


def load_results_from_pkl(filename):
    # load from pickle file
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def load_results_from_json(filename):
    assert os.path.isfile(filename)
    with open(filename, "r") as f:
        results = json.load(f)
    # for activity net external classification scores
    if 'results' in results:
        results = results['results']
    return results

def results_to_dict(results):
    """convert result arrays into dict used by json files"""
    # video ids and allocate the dict
    vidxs = sorted(list(set(results['video-id'])))
    results_dict = {}
    for vidx in vidxs:
        results_dict[vidx] = []

    # fill in the dict
    for vidx, start, end, label, score in zip(
        results['video-id'],
        results['t-start'],
        results['t-end'],
        results['label'],
        results['score']
    ):
        results_dict[vidx].append(
            {
                "label" : int(label),
                "score" : float(score),
                "segment": [float(start), float(end)],
            }
        )
        
    return results_dict


def results_to_array(results):
    # video ids and allocate the dict
    vidxs = sorted(list(set(results['video-id'])))
    results_dict = {}
    for vidx in vidxs:
        results_dict[vidx] = {
            'label'   : [],
            'score'   : [],
            'segment' : [],
            # 'temp_idxs': [],
        }

    # fill in the dict
    for vidx, start, end, label, score in zip(
        results['video-id'],
        results['t-start'],
        results['t-end'],
        results['label'],
        results['score'],
    ):
        results_dict[vidx]['label'].append(int(label))
        results_dict[vidx]['score'].append(float(score))
        results_dict[vidx]['segment'].append(
            [float(start), float(end)]
        )

    for vidx in vidxs:
        label = np.asarray(results_dict[vidx]['label'])
        score = np.asarray(results_dict[vidx]['score'])
        segment = np.asarray(results_dict[vidx]['segment'])

        # the score should be already sorted, just for safety
        results_dict[vidx]['label'] = label
        results_dict[vidx]['score'] = score
        results_dict[vidx]['segment'] = segment
    
    return results_dict


# ---- DURPRIOR (TIFAD_DURPRIOR=<gamma>, eval-time, byte-inert unset):
# zero-shot duration-prior rescoring. Per test class u, fit a log-normal
# over SEEN-class GT durations (training subset of the annotation json --
# TIFAD_DURPRIOR_SUBSET, default 'Validation' (THUMOS quirk: 'Validation'
# IS the training data); ANet uses 'training';
# seen = annotation classes minus the current test classes, so the unseen
# class's own GT is never touched). TIFAD_DURPRIOR_SRC selects the
# mixture weights over seen classes: 'attr' (default; Jaccard similarity
# of attribute sets from TIFAD_DURPRIOR_ATTRS) or 'allseen' (uniform).
# Rescoring: score *= exp(-gamma * z^2 / 2), z = (log dur - mu_u)/sig_u
# -- p(dur)^gamma with the log-space Gaussian kernel (per-class constants
# cancel in per-class AP). Measured (famce_uw, 10 splits): fused@0.35
_DURPRIOR_SIG_FLOOR = 0.15
_DURPRIOR_CACHE = {}


def _durprior_params(test_names):
    """(mu, sig) float arrays aligned with test_names (cached)."""
    key = tuple(test_names)
    if key in _DURPRIOR_CACHE:
        return _DURPRIOR_CACHE[key]
    src = os.environ.get('TIFAD_DURPRIOR_SRC', 'attr')
    anno = os.environ.get(
        'TIFAD_DURPRIOR_JSON',
        '/data3/xiaodan8/thumos/annotations/thumos14.json')
    attr_file = os.environ.get(
        'TIFAD_DURPRIOR_ATTRS',
        '/data3/xiaodan8/tifad/configs/thumos14_ttb_attrs_v3.json')
    train_subset = os.environ.get('TIFAD_DURPRIOR_SUBSET', 'Validation')
    with open(anno) as f:
        db = json.load(f)['database']
    logdur = {}
    for v in db.values():
        if v['subset'] != train_subset:
            continue
        for a in v['annotations']:
            d = max(float(a['segment'][1]) - float(a['segment'][0]), 1e-3)
            logdur.setdefault(a['label'], []).append(np.log(d))
    seen = sorted(c for c in logdur if c not in set(test_names))
    assert seen, "[DURPRIOR] no seen classes left after removing test set"
    s_mu = np.array([np.mean(logdur[c]) for c in seen])
    s_var = np.array([np.var(logdur[c]) for c in seen])
    if src == 'attr':
        attrs = {k: set(v) for k, v in json.load(open(attr_file)).items()}
    else:
        assert src == 'allseen', f"[DURPRIOR] unknown src '{src}'"
    mu = np.zeros(len(test_names))
    sig = np.zeros(len(test_names))
    for i, u in enumerate(test_names):
        if src == 'attr' and u in attrs:
            w = np.array([
                len(attrs[u] & attrs.get(c, set()))
                / max(len(attrs[u] | attrs.get(c, set())), 1)
                for c in seen])
            if w.sum() <= 0:
                w = np.ones(len(seen))
        else:
            w = np.ones(len(seen))
        w = w / w.sum()
        m = float((w * s_mu).sum())
        var = float((w * (s_var + s_mu ** 2)).sum() - m ** 2)
        mu[i] = m
        sig[i] = max(np.sqrt(max(var, 0.0)), _DURPRIOR_SIG_FLOOR)
    _DURPRIOR_CACHE[key] = (mu, sig)
    print(f"[DURPRIOR] active (gamma={os.environ.get('TIFAD_DURPRIOR')}, "
          f"src={src}, {len(seen)} seen -> {len(test_names)} test classes, "
          f"median priors "
          f"{[round(float(np.exp(m)), 2) for m in mu]}s)", flush=True)
    return mu, sig


def postprocess_results(results, cls_score_file, prior_prob=0.15, num_pred=200, topk=2):
    # load results and convert to dict
    if isinstance(results, str):
        results = load_results_from_pkl(results)
    # array -> dict
    class_names = results['class_names']
    results = results_to_array(results)

    # load external classification scores
    if '.json' in cls_score_file:
        with open(cls_score_file, "rt") as f:
            data = json.load(f)
        cls_scores = data['results']
        full_class_names = data['class']
        class_name_to_idx = {
            name: i
            for i, name in enumerate(full_class_names)
        }
    else:
        cls_scores = load_results_from_pkl(cls_score_file)
    
    # dict for processed results
    processed_results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': [],
    }

    # process each video
    for i, (vid, result) in enumerate(results.items()):
        # pick top k cls scores and idx
        curr_cls_scores = np.asarray(cls_scores[vid])
        class_indices = np.asarray([class_name_to_idx[name] for name in class_names[i]])
        # class_indices = np.asarray([ 0,  2,  3,  9, 10, 11, 14, 16, 17, 18])   # split 0
        # import ipdb; ipdb.set_trace()

        curr_cls_scores = curr_cls_scores[class_indices]
        
        topk = max((curr_cls_scores > prior_prob).sum(), 1)
        topk_cls_indices = np.argsort(curr_cls_scores)[::-1][:topk]
        
        valid_mask = [result['label'] == topk_cls_idx for topk_cls_idx in topk_cls_indices]
        valid_mask = reduce(np.logical_or, valid_mask)
        
        pred_score = result['score'][valid_mask]
        pred_segment = result['segment'][valid_mask]
        pred_label = result['label'][valid_mask]

        # ---- DURPRIOR rescoring (see _durprior_params above). Placed
        # BEFORE the geometric fusion multiplier (they commute) and before
        # the num_pred truncation (where the ordering matters). Byte-inert
        # when TIFAD_DURPRIOR is unset/0.
        import os as _osdp
        _dpg = float(_osdp.environ.get('TIFAD_DURPRIOR', '0') or 0)
        if _dpg > 0 and len(pred_score) > 0:
            _dp_mu, _dp_sig = _durprior_params(class_names[i])
            _dp_d = np.maximum(
                pred_segment[:, 1] - pred_segment[:, 0], 1e-3)
            _dp_lab = pred_label.astype(int)
            _dp_z = (np.log(_dp_d) - _dp_mu[_dp_lab]) / _dp_sig[_dp_lab]
            # ---- ASYM (TIFAD_DURPRIOR_ASYM=<gamma_long>, R3 2026-09-17):
            # over-extension-aware prior. The diagnosis found the top
            # detection 1.35x too LONG on average, so a stronger gamma
            # applies to candidates LONGER than the transferred mean
            # (z > 0) and the plain gamma to shorter ones. Byte-identical
            # to the symmetric prior when unset or equal to gamma.
            _dpa = _osdp.environ.get('TIFAD_DURPRIOR_ASYM', '')
            if _dpa:
                _dpg = np.where(_dp_z > 0, float(_dpa), _dpg)
            pred_score = pred_score * np.exp(-_dpg * 0.5 * _dp_z * _dp_z)

        # ---- soft geometric fusion (OVD-style), env-gated: score *=
        # cls_score[label]^alpha on top of the hard top-k filter. Inert at 0.
        import os as _osf
        _fa = float(_osf.environ.get('TIFAD_FUSE_ALPHA', '0') or 0)
        if _fa > 0 and len(pred_score) > 0:
            pred_score = pred_score * np.power(
                np.clip(curr_cls_scores[pred_label.astype(int)], 1e-6, None), _fa)

        inds = np.argsort(pred_score)[::-1][:num_pred]
        pred_score = pred_score[inds]
        pred_segment = pred_segment[inds]
        pred_label = pred_label[inds]
        
        num_segs = min(num_pred, len(pred_score))
        
        processed_results['video-id'].extend([vid] * len(pred_score))
        processed_results['t-start'].append(pred_segment[:, 0])
        processed_results['t-end'].append(pred_segment[:, 1])
        processed_results['label'].append(pred_label)
        processed_results['score'].append(pred_score)

    processed_results['t-start'] = np.concatenate(
        processed_results['t-start'], axis=0)
    processed_results['t-end'] = np.concatenate(
        processed_results['t-end'], axis=0)
    processed_results['label'] = np.concatenate(
        processed_results['label'],axis=0)
    processed_results['score'] = np.concatenate(
        processed_results['score'], axis=0)

    return processed_results
