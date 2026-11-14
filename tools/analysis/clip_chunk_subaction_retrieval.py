#!/usr/bin/env python
"""Do the equal-split video chunks correspond to the text sub-actions?

A model-free test with CLIP ViT-B/32 as a neutral judge. For every FineDiving
test dive with M annotated sub-actions:
  * split the dive span into M equal chunks (what the model does) and, as the
    reference, use the M annotated sub-action spans (oracle chunks);
  * embed up to 4 frames per chunk from the pre-built frame store (mean CLIP
    image embedding) and the M sub-action root sentences (CLIP text);
  * retrieval: chunk k -> argmax_j cos(chunk_k, sentence_j) over the dive's own
    M sub-actions. Accuracy = fraction with j == k. Chance = mean 1/M.
If equal chunks retrieve their own sub-action well above chance, the chunk /
sub-action correspondence holds; the gap to the oracle
split is the cost of equal-length boundaries.
"""
import json, os, re, sys, statistics, collections
import numpy as np, torch
from PIL import Image

FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
A6 = '/data3/xiaodan8/actionformer6'
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))


def main():
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(16)
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval()
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    meta = json.load(open(f'{FD}/video_metadata_cache_test.json'))
    subname = {k: v['name'] for k, v in json.load(open(f'{A6}/configs/finediving/fd_sub_verbalizer.json'))['actions'].items()}
    root_of = {re.sub(r'^\([A-Z]+\) ', '', k): v[0] for k, v in json.load(open(f'{A6}/configs/finediving/fd_sub_ensemble.json')).items()}

    # text embeddings for the 28 sub-action root sentences
    names = sorted(root_of)
    with torch.no_grad():
        t = proc(text=[root_of[n] for n in names], return_tensors='pt', padding=True)
        temb = torch.nn.functional.normalize(model.get_text_features(**t), dim=-1)
    tidx = {n: i for i, n in enumerate(names)}

    store_idx = {}
    def frames_in(video):
        if video not in store_idx:
            d = f'{FD}/RGB/{video}'
            store_idx[video] = np.array(sorted(int(f[:-4]) for f in os.listdir(d))) if os.path.isdir(d) else np.array([])
        return store_idx[video]

    def embed_chunk(video, fps, total, a, b, k=4):
        idx = frames_in(video)
        if len(idx) == 0:
            return None
        want = [int(round((a + (i + 0.5) * (b - a) / k) * fps)) for i in range(k)]
        pick = sorted({int(idx[np.abs(idx - w).argmin()]) for w in want})
        nd = max(len(str(total)), 6)
        ims = [Image.open(f'{FD}/RGB/{video}/{p:0{nd}d}.jpg').convert('RGB') for p in pick]
        with torch.no_grad():
            x = proc(images=ims, return_tensors='pt')
            e = torch.nn.functional.normalize(model.get_image_features(**x), dim=-1).mean(0)
        return torch.nn.functional.normalize(e, dim=-1)

    res = collections.defaultdict(list)       # (setting, group) -> list of 0/1
    chance = []
    dives = 0
    for line in open(f'{FD}/finediving_test_label.txt'):
        if not line.strip():
            continue
        r = json.loads(line); video = r['video'].rsplit('_', 1)[0]
        if video not in meta:
            continue
        fps, total = meta[video]['fps'], meta[video]['total_frames']
        for x in r['raw_value']:
            s, e = map(sec, x['span']); subs = [(subname.get(q['action_id'], q['action_id']), *map(sec, q['span'])) for q in x['actions']]
            if any(n not in tidx for n, _, _ in subs):
                continue
            M = len(subs); L = e - s
            T = temb[[tidx[n] for n, _, _ in subs]]                   # (M, 512)
            for setting, spans in (('equal', [(s + k * L / M, s + (k + 1) * L / M) for k in range(M)]),
                                   ('oracle', [(a, b) for _, a, b in subs])):
                for k, (a, b) in enumerate(spans):
                    v = embed_chunk(video, fps, total, a, b)
                    if v is None:
                        continue
                    pred = int((T @ v).argmax())
                    g = '3-part' if M == 3 else '4-5-part'
                    res[(setting, g)].append(int(pred == k)); res[(setting, 'all')].append(int(pred == k))
            chance.append(1 / M); dives += 1
            if dives % 100 == 0:
                print(f'{dives} dives...', flush=True)
    print(f'\n{dives} test dives; chance (mean 1/M) = {100*statistics.mean(chance):.1f}%')
    for g in ('all', '3-part', '4-5-part'):
        eq, orc = res[('equal', g)], res[('oracle', g)]
        print(f'{g:9s} n={len(eq):5d}  chunk-k -> sub-action-k top-1:  equal split {100*statistics.mean(eq):.1f}%   '
              f'annotated spans {100*statistics.mean(orc):.1f}%')
    out = {k[0] + '/' + k[1]: statistics.mean(v) for k, v in res.items()}
    out.update({'chance': statistics.mean(chance), 'dives': dives, 'n': {k[0] + '/' + k[1]: len(v) for k, v in res.items()}})
    json.dump(out, open(f'{A6}/tools/analysis/clip_chunk_subaction_retrieval.json', 'w'), indent=1)   # py3.8: no dict |


if __name__ == '__main__':
    main()
