#!/usr/bin/env python
"""Score C3 FineDiving predictions with the video-level evaluator (parsing2 env, actionformer6).

Builds the val dataset exactly as train_shard.py does (label_dict, video-level GT
per clip, held-out ids), converts the VLM's window-relative dives to absolute
seconds, applies the same soft-NMS per clip (libs.utils.nms.batched_nms with the
config's test_cfg), evaluates seen / held-out video mAP, and reports the
sub-action grounding table (predicted sub-action spans vs annotated spans).

  python scripts/vlm_hier_eval_fd.py --pred /data3/xiaodan8/tifad/logs/vlm_hier/fd_s1_pred.jsonl
"""
import argparse, json, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CFG = 'configs/finediving/v3/fd_champ_s1_a4000.yaml'
ATTR = 'configs/finediving/fd_dive_attribute_table_s1.json'
FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True); ap.add_argument('--src', default='all')
    ap.add_argument('--ids', default='', help='file of window ids: evaluate only these clips (predictions and ground truth), for the zero-shot subsample')
    a = ap.parse_args()
    keep = {l.strip() for l in open(a.ids) if l.strip()} if a.ids else None
    import torch
    from libs.core import load_config
    from libs.datasets import make_dataset
    from libs.utils.nms import batched_nms
    from libs.utils.metrics import ANETdetection
    cfg = load_config(CFG)
    held = [int(x[1:]) for x in json.load(open(ATTR))['held_out_classes']]
    cfg['dataset']['held_out_class_ids'] = held
    ds = make_dataset(cfg['dataset_name'], False, cfg['val_split'], cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
    gt = ds.get_video_level_ground_truth_df()
    if keep is not None:
        gt = gt[gt['video-id'].isin(keep)].reset_index(drop=True)
    verb = json.load(open('configs/finediving/fd_dive_verbalizer.json'))['actions']
    name2code = {v['name'].lower(): v['code'] for v in verb.values()}
    code2lab = ds.label_dict                                   # dive code -> int label
    rows = [json.loads(l) for l in open(a.pred)]
    if keep is not None:
        rows = [r for r in rows if r['video'] in keep]
    per_clip, n_inst, n_unk = collections.defaultdict(list), 0, 0
    subs_pred = {}                                             # clip -> best (greedy) dive's sub list, absolute
    for r in rows:
        for p in r['preds']:
            if a.src == 'greedy' and p.get('src') != 'greedy':
                continue
            n_inst += 1
            code = name2code.get(p['action'])
            if code is None or code not in code2lab:
                n_unk += 1; continue
            per_clip[r['video']].append((r['win_start'] + p['start'], r['win_start'] + p['end'], code2lab[code], float(p['conf'])))
            if p.get('src') == 'greedy' and r['video'] not in subs_pred:
                subs_pred[r['video']] = [(q['action'], r['win_start'] + q['start'], r['win_start'] + q['end']) for q in p.get('sub', [])]
    res = {'video-id': [], 't-start': [], 't-end': [], 'label': [], 'score': []}
    tc = cfg['test_cfg']
    for clip, dets in per_clip.items():
        segs = torch.tensor([[d[0], d[1]] for d in dets]); scores = torch.tensor([d[3] for d in dets]); labels = torch.tensor([d[2] for d in dets])
        s, sc, lb = batched_nms(segs, scores, labels, iou_threshold=0.5, min_score=tc['min_score'], max_seg_num=tc['max_seg_num'],
                                use_soft_nms=True, multiclass=True, voting_thresh=tc.get('voting_thresh', 0.75))
        for j in range(len(s)):
            res['video-id'].append(clip); res['t-start'].append(float(s[j, 0])); res['t-end'].append(float(s[j, 1])); res['label'].append(int(lb[j])); res['score'].append(float(sc[j]))
    # the evaluator keeps only clips that have at least one prediction (train_utils does the same; the detector
    # always emits proposals, so every clip is present there). A VLM that answers [] on a clip must not have that
    # clip's ground truth dropped: give such clips one dummy prediction at score ~0 (ranked last; negligible AP effect).
    n_dummy = 0
    for clip in gt['video-id'].unique():
        if clip not in per_clip:
            res['video-id'].append(clip); res['t-start'].append(0.0); res['t-end'].append(0.1); res['label'].append(int(gt['label'].iloc[0])); res['score'].append(1e-9); n_dummy += 1
    for k in ('t-start', 't-end', 'label', 'score'):
        res[k] = np.array(res[k])
    print(f'{n_dummy} clips without any prediction received a dummy (score 1e-9) so their ground truth stays in the evaluation')
    print(f'{len(rows)} windows, {n_inst} dive instances ({n_unk} unmatched names), {len(res["video-id"])} after soft-NMS, {len(per_clip)}/{gt["video-id"].nunique()} clips with a prediction')
    ev = ANETdetection(ant_file=None, split=None, tiou_thresholds=np.linspace(0.3, 0.7, 5), ground_truth_df=gt, dataset_name='finediving_vlm')
    _, mAP, _, ap_mat = ev.evaluate(res, verbose=True)
    # ap columns follow ev.activity_index (dense ids present after evaluation, in order); raw -> dense via label_to_col
    col = dict(getattr(ev, 'label_to_col', {})); dense_cols = list(ev.activity_index.keys())
    held_dense = {col[x] for x in held if x in col}
    held_set = [i for i, j in enumerate(dense_cols) if j in held_dense]; seen_set = [i for i, j in enumerate(dense_cols) if j not in held_dense]
    assert ap_mat.shape[1] == len(dense_cols), (ap_mat.shape, len(dense_cols))
    v_seen, v_held = float(ap_mat[:, seen_set].mean()), float(ap_mat[:, held_set].mean())
    print(f'RESULT C3 FineDiving s1 [{a.src}]' + (f' subset={len(keep)}' if keep is not None else '') + f': video mAP {mAP:.4f} | seen-class mAP {v_seen:.4f} ({len(seen_set)}) | held-out-class mAP {v_held:.4f} ({len(held_set)})')

    # ---- sub-action grounding: predicted (greedy) sub spans vs annotated spans, by position
    gt_subs = {}
    subverb = json.load(open('configs/finediving/fd_sub_verbalizer.json'))['actions']; raw2name = {v['raw_id']: v['name'] for v in subverb.values()}
    for line in open(f'{FD}/finediving_test_label.txt'):
        if line.strip():
            r = json.loads(line)
            for x in r['raw_value']:
                gt_subs[r['video']] = [(raw2name.get(q['action_id'], q['action_id']), sec(q['span'][0]), sec(q['span'][1])) for q in x['actions']]
    pos_t, name_ok, n_cnt_ok, n_clip = collections.defaultdict(list), [], 0, 0
    for clip, g in gt_subs.items():
        p = subs_pred.get(clip)
        if p is None:
            continue
        n_clip += 1; n_cnt_ok += int(len(p) == len(g))
        for k, (gn, ga, gb) in enumerate(g):
            if k < len(p):
                pos_t[f'{k+1}/{len(g)}'].append(tiou((p[k][1], p[k][2]), (ga, gb))); name_ok.append(int(p[k][0] == gn))
            else:
                pos_t[f'{k+1}/{len(g)}'].append(0.0); name_ok.append(0)
    allt = [t for v in pos_t.values() for t in v]
    print(f'SUB-ACTIONS (greedy dive per clip, {n_clip} clips): count correct {100*n_cnt_ok/max(1,n_clip):.1f} %, '
          f'tIoU mean {np.mean(allt):.3f}, >=0.5 {100*np.mean([t>=0.5 for t in allt]):.1f} %, name match {100*np.mean(name_ok):.1f} %')
    for k in sorted(pos_t, key=lambda s: (int(s.split('/')[1]), int(s.split('/')[0]))):
        v = pos_t[k]; print(f'  position {k}: mean tIoU {np.mean(v):.3f}, >=0.5 {100*np.mean([t>=0.5 for t in v]):.0f} %  (n={len(v)})')
    out = a.pred.replace('.jsonl', f'_eval_{a.src}.json')
    json.dump({'video_mAP': mAP, 'seen_mAP': v_seen, 'held_mAP': v_held, 'sub_tiou_mean': float(np.mean(allt)) if allt else None,
               'sub_tiou_ge05': float(np.mean([t >= 0.5 for t in allt])) if allt else None, 'sub_name_match': float(np.mean(name_ok)) if name_ok else None,
               'sub_count_correct': n_cnt_ok / max(1, n_clip), 'per_position': {k: float(np.mean(v)) for k, v in pos_t.items()}}, open(out, 'w'), indent=1)
    print('->', out)


if __name__ == '__main__':
    main()
