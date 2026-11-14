"""Build video_metadata_cache_{train,test}.json for activitynet_raw.

The ANet I3D features are uniform in TIME (one row per 0.64 s
= 16 frames at a virtual 25 fps, regardless of each video's native fps). The
cache therefore declares fps=25.0 for every video with total_frames derived
from the real duration, which makes frame index 16*k land exactly on feature
row k (feat_native_stride=16).
"""
import json

FC = '/data3/xiaodan8/Activitynet/frame_count_raw_video.json'
ANN = '/data3/xiaodan8/Activitynet/annotation/activitynet_{split}_label.txt'
OUT = '/data3/xiaodan8/Activitynet/video_metadata_cache_{split}.json'
VIRTUAL_FPS = 25.0

fc = json.load(open(FC))
for split in ('train', 'test'):
    cache = {}
    n_missing = 0
    for line in open(ANN.format(split=split)):
        v = json.loads(line)['video']
        m = fc.get(v)
        if m is None:
            n_missing += 1
            continue
        dur = float(m['video_seconds'])
        cache[v] = {
            'fps': VIRTUAL_FPS,
            'total_frames': int(round(dur * VIRTUAL_FPS)),
            'duration': dur,
        }
    json.dump(cache, open(OUT.format(split=split), 'w'))
    print(f"{split}: {len(cache)} videos cached, {n_missing} missing from frame-count json")
