"""scripts/seen_eval.py for the OTHER protocols (THUMOS 50/50, ANet 75/25 and
50/50): the identical measurement (unseen mAP + seen-vocab control + pooled
mean, fused and unfused), with the CELL'S OWN fusion settings instead of the
THUMOS-75/25 constants hard-wired in seen_eval.py (TIFAD_FUSE_ALPHA=0.35,
TIFAD_DURPRIOR=0.3).

Usage (everything after --fuse "..." is passed to seen_eval.py unchanged):
  python scripts/seen_eval_proto.py --fuse "TIFAD_FUSE_ALPHA=0.45 TIFAD_DURPRIOR=0.3" \
      --config configs/thumos14_i3d_50.yaml --ckpt ... --split n --envs "..." --out ...
  ANet: --fuse "TIFAD_FUSE_ALPHA=0.15 TIFAD_DURPRIOR=0.15 TIFAD_DURPRIOR_SRC=allseen
                TIFAD_DURPRIOR_SUBSET=training TIFAD_DURPRIOR_JSON=<anet json>"
The --fuse variables are set only while the FUSED numbers are computed and
removed for the unfused ones (seen_eval.evaluate); they are recorded under
meta.fused in the output json. seen_eval.py itself is left untouched so the
THUMOS-75/25 sweep (run_seen_eval.sh) keeps its exact behaviour.
"""
import os
import runpy
import shlex
import sys


def main():
    argv = sys.argv[1:]
    assert '--fuse' in argv, "seen_eval_proto.py needs --fuse \"K=V ...\""
    i = argv.index('--fuse')
    spec = argv[i + 1]
    del argv[i:i + 2]
    fused = {}
    for tok in shlex.split(spec):
        assert '=' in tok, f"bad --fuse token {tok!r}"
        k, v = tok.split('=', 1)
        fused[k] = v
    assert 'TIFAD_FUSE_ALPHA' in fused, "--fuse must set TIFAD_FUSE_ALPHA"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seen_eval.py')
    sys.argv = [path] + argv
    # seen_eval.py parses its CLI and applies --envs at import time; run it
    # as a plain module (not __main__) and swap the fusion constants IN
    # PLACE (evaluate() and the json meta hold the same dict object).
    g = runpy.run_path(path, run_name='seen_eval_lib')
    g['FUSED'].clear()
    g['FUSED'].update(fused)
    print(f"[seen_eval_proto] fused settings: {g['FUSED']}", flush=True)
    g['main']()


if __name__ == '__main__':
    main()
