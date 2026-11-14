"""Pick 20 held-out c-actions such that every individual attribute value
appears in at least one *seen* (training) class — guarantees compositional
reachability for ZSL.

Run:
  python3 scripts/choose_held_out.py
Updates configs/finegym_attribute_table_v2.json in-place with the chosen list.
"""
import json
import random
import numpy as np

NUM_HELD_OUT = 20
SEED = 42
PATH = 'configs/finegym_attribute_table_v2.json'


def main():
    with open(PATH) as f:
        d = json.load(f)

    actions = d['actions']
    names = d['attribute_names']
    K = len(names)
    aids = sorted(actions.keys(), key=lambda x: int(x[1:]))
    M = np.array([actions[aid]['attrs'] for aid in aids], dtype=np.int8)  # 99 × K

    rng = random.Random(SEED)

    def reachability_ok(heldout_idx):
        seen_idx = [i for i in range(99) if i not in heldout_idx]
        seen_M = M[seen_idx]
        return (seen_M.sum(axis=0) > 0).all()

    for trial in range(10000):
        sample_idx = set(rng.sample(range(99), NUM_HELD_OUT))
        if reachability_ok(sample_idx):
            held_out = sorted([aids[i] for i in sample_idx], key=lambda x: int(x[1:]))
            print(f"trial {trial}: found reachable held-out set")
            print(f"  held_out: {held_out}")
            d['held_out_classes'] = held_out
            with open(PATH, 'w') as f:
                json.dump(d, f, indent=2)
            print(f"  wrote to {PATH}")
            heldout_M = M[list(sample_idx)]
            print(f"  per-attribute: seen-count, held-out-count")
            for j, name in enumerate(names):
                seen_count = (M[[i for i in range(99) if i not in sample_idx], j]).sum()
                ho_count = heldout_M[:, j].sum()
                if seen_count == 0:
                    print(f"    DEAD-IN-SEEN: {name}")
                elif ho_count > 0:
                    print(f"    {j:>2}  {name:30s}  seen={seen_count}  held-out={ho_count}")
            return
    print("FAIL: no reachable set found in 10000 trials")
    raise SystemExit(1)


if __name__ == '__main__':
    main()
