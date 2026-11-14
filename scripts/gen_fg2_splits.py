"""Random seen/unseen splits for the l_d=2 (element-set) protocol.

Each split holds out 4 of the 14 element sets subject to: vault (p00) stays seen (sole set of its
apparatus) and every apparatus keeps >= 1 seen set. For each split k it writes
  configs/splits/finegym_phrase_attribute_table_s<k>.json   (copy of the phrase table, new held_out_classes)
  configs/splits/finegym_t2_duration_prior_s<k>.json         (scripts/gen_finegym_duration_prior.py --level phrase)
  configs/splits/finegym_t2_{champ,champ_stage2,champ_stage3_nested,champ_stage3_flat,tifad}_s<k>.yaml
Usage: python scripts/gen_fg2_splits.py --seeds 1 2 3
"""
import argparse, json, os, random, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = '/home/jupyter/xhu3/video/dataset/finegym/annotation/Dec16/gym99_train_label.txt'
PY = sys.executable

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3]); a = ap.parse_args()
    os.chdir(ROOT)
    verb = json.load(open('configs/finegym_zsl_verbalizer.json'))
    p2a = {k: v['activity'] for k, v in verb['phrases'].items()}
    names = {k: v['name'] for k, v in verb['phrases'].items()}
    table = json.load(open('configs/finegym_phrase_attribute_table.json'))
    candidates = sorted(p for p in p2a if p != 'p00')
    summary = {}
    for k, seed in enumerate(a.seeds, start=1):
        rng = random.Random(seed)
        while True:
            held = sorted(rng.sample(candidates, 4))
            seen_apps = {p2a[p] for p in p2a if p not in held}
            if seen_apps == set(p2a.values()):
                break
        t = dict(table); t['held_out_classes'] = held
        t['note'] = f"random split s{k} (seed {seed}): 4/14 element sets held out; vault kept seen; every apparatus keeps >=1 seen set"
        tp = f'configs/splits/finegym_phrase_attribute_table_s{k}.json'
        json.dump(t, open(tp, 'w'), indent=1)
        pp = f'configs/splits/finegym_t2_duration_prior_s{k}.json'
        subprocess.run([PY, 'scripts/gen_finegym_duration_prior.py', '--level', 'phrase', '--labels', LABELS,
                        '--attr-table', tp, '--out', pp, '--verbalizer', 'configs/finegym_zsl_verbalizer.json'],
                       check=True, stdout=subprocess.DEVNULL)
        for cfg in ('champ', 'champ_stage2', 'champ_stage3_nested', 'champ_stage3_flat', 'tifad'):
            s = open(f'configs/finegym_t2_{cfg}.yaml').read()
            s2 = s.replace('configs/finegym_phrase_attribute_table.json', tp).replace('configs/finegym_t2_duration_prior.json', pp)
            assert s2 != s, cfg
            if cfg.startswith('champ_stage3') or cfg == 'champ_stage2':
                # stage-2/3 configs are launched with --init_from; nothing else is split-specific
                pass
            open(f'configs/splits/finegym_t2_{cfg}_s{k}.yaml', 'w').write(s2)
        summary[f's{k}'] = {'seed': seed, 'held_out': held, 'names': [names[p] for p in held]}
        print(f"s{k} (seed {seed}): {held} = {[names[p] for p in held]}")
    json.dump(summary, open('configs/splits/SPLITS.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
