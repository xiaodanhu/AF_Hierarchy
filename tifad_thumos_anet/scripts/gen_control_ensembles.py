#!/usr/bin/env python
"""Text controls: do the gains come from the sub-action
semantics, or merely from having K extra sentences and K temporal slots?

Two ensembles derived from configs/thumos14_qwen_ensemble.json, keeping the
root sentence (index 0) of every class unchanged and replacing only the K=10
ordered sub-action sentences:

  thumos14_qwen_ensemble_cfree.json   content-free: the SAME ten generic,
      temporally ordered, class-agnostic sentences for every class. Keeps the
      token count and the K slots, removes all class-specific semantics.
  thumos14_qwen_ensemble_xswap.json   cross-class swap: class c receives the
      ten sub-action sentences of another class (a fixed derangement, seed 0).
      Keeps realistic sentences and the slots, breaks the match to the class.

(51.00), the semantics were doing the work.
"""
import json, random

SRC = 'configs/thumos14_qwen_ensemble.json'
GENERIC = [
    "The person stands ready before the action begins.",
    "The person starts the first movement of the action.",
    "The person continues the opening movement with more speed.",
    "The person reaches the first key moment of the action.",
    "The person is in the middle of the action.",
    "The person continues through the main part of the action.",
    "The person reaches the peak of the action.",
    "The person begins to finish the action.",
    "The person completes the final movement of the action.",
    "The person comes to rest after the action ends.",
]

src = json.load(open(SRC))
classes = list(src)
K = len(src[classes[0]]) - 1
assert all(len(v) == K + 1 for v in src.values()) and len(GENERIC) == K

cfree = {c: [src[c][0]] + GENERIC for c in classes}
json.dump(cfree, open('configs/thumos14_qwen_ensemble_cfree.json', 'w'), indent=1)

rng = random.Random(0)
while True:                                   # derangement: nobody keeps its own children
    perm = classes[:]; rng.shuffle(perm)
    if all(a != b for a, b in zip(classes, perm)):
        break
xswap = {c: [src[c][0]] + src[p][1:] for c, p in zip(classes, perm)}
json.dump(xswap, open('configs/thumos14_qwen_ensemble_xswap.json', 'w'), indent=1)
json.dump(dict(zip(classes, perm)), open('configs/thumos14_qwen_ensemble_xswap_map.json', 'w'), indent=1)

for name, d in (('cfree', cfree), ('xswap', xswap)):
    assert list(d) == classes and all(len(v) == K + 1 for v in d.values())
    assert all(d[c][0] == src[c][0] for c in classes), 'root sentences must be unchanged'
print(f'{len(classes)} classes, K={K}: wrote cfree (identical children for all classes) and xswap '
      f'(derangement, e.g. {classes[0]} <- {perm[0]})')
