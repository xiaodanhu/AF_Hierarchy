#!/usr/bin/env python
"""Convert the FineDiving release pickles into the CSVs the annotation pipeline reads.

The OpenDataLab release ships its annotations as pickles. The dataset-side
script `generate_annotations.py`, which builds `finediving_{train,test}_label.txt`,
reads CSVs instead, with two filenames that signal local processing
("corrected", "filtered"). Those CSVs were produced on the machine that ran the
experiments and were never part of the release, so this script regenerates them.

The filtering rule was recovered and verified: the sub-action table keeps every
id that actually occurs in the per-frame labels, excluding Background, in
ascending id order, renumbered from zero. That reproduces the table carried in
the reference CSV byte for byte, which is the check run by --verify.

Usage, from the FineDiving dataset root:
    python finediving_pkl_to_csv.py [--root .] [--verify <reference_csv>]
"""
import argparse, csv, os, pickle, sys

ANN = 'raw/FineDiving/Released_FineDiving_Dataset/Annotations'


def listify(x):
    """numpy array or sequence -> a Python list literal that ast.literal_eval reads."""
    try:
        return repr([int(v) for v in x.tolist()])
    except AttributeError:
        return repr([int(v) for v in x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='FineDiving dataset root')
    ap.add_argument('--ann', default=None, help='override the Annotations directory')
    ap.add_argument('--verify', default=None,
                    help='path to the reference copy of Sub_action_Types_Table_filtered.csv')
    a = ap.parse_args()

    ann = a.ann or os.path.join(a.root, ANN)
    coarse = pickle.load(open(os.path.join(ann, 'FineDiving_coarse_annotation.pkl'), 'rb'))
    fine = pickle.load(open(os.path.join(ann, 'FineDiving_fine-grained_annotation.pkl'), 'rb'))
    table = pickle.load(open(os.path.join(ann, 'Sub_action_Types_Table.pkl'), 'rb'))
    print(f'loaded {len(coarse)} coarse, {len(fine)} fine-grained, {len(table)} sub-action types')
    if set(coarse) != set(fine):
        print('WARNING: the coarse and fine-grained key sets differ', file=sys.stderr)

    # ---- sub-action table: keep the ids that occur, drop Background, renumber
    used = set()
    for v in fine.values():
        used.update(int(l) for l in v['frames_labels'])
    kept = [i for i in sorted(used) if table[i] != 'Background']
    p = os.path.join(ann, 'Sub_action_Types_Table_filtered.csv')
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'sub_action_type', 'new_id'])
        for new_id, i in enumerate(kept):
            w.writerow([i, table[i], new_id])
    print(f'wrote {p}  ({len(kept)} of {len(table)} sub-action types kept)')

    if a.verify:
        mine = open(p, 'rb').read().replace(b'\r\n', b'\n')
        ref = open(a.verify, 'rb').read().replace(b'\r\n', b'\n')
        print('verify against the reference copy:', 'IDENTICAL' if mine == ref else 'DIFFERS')
        if mine != ref:
            return 1

    # ---- coarse
    p = os.path.join(ann, 'FineDiving_coarse_annotation.csv')
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'clip_id', 'action_type', 'start_frame', 'end_frame'])
        for (vid, clip), v in sorted(coarse.items()):
            w.writerow([vid, clip, v['action_type'], v['start_frame'], v['end_frame']])
    print(f'wrote {p}')

    # ---- fine-grained. The name the pipeline expects carries "corrected"; what
    # that correction was on the original machine is not recorded, so this is the
    # straight conversion. Check the window counts (2252 train / 781 test) after
    # running generate_annotations.py.
    p = os.path.join(ann, 'FineDiving_fine-grained_annotation_corrected.csv')
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'clip_id', 'action_type', 'frames_labels', 'steps_transit_frames'])
        for (vid, clip), v in sorted(fine.items()):
            w.writerow([vid, clip, v['action_type'],
                        listify(v['frames_labels']), listify(v['steps_transit_frames'])])
    print(f'wrote {p}')

    # ---- train / test split, also shipped as pickles
    split_dir = os.path.join(os.path.dirname(ann.rstrip('/')), 'train_test_split')
    for name in ('train', 'test'):
        src = os.path.join(split_dir, f'{name}_split.pkl')
        if not os.path.exists(src):
            print(f'note: {src} not found, skipping')
            continue
        keys = pickle.load(open(src, 'rb'))
        p = os.path.join(split_dir, f'{name}_split.csv')
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['video', 'clip_id'])
            for vid, clip in keys:
                w.writerow([vid, clip])
        print(f'wrote {p}  ({len(keys)} clips)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
