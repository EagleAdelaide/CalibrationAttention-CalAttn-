#!/usr/bin/env bash
set -e

CFG=configs/cifar100_deits_calattn.yaml

for SEED in 0 1 2; do
  python - <<PY
import yaml
cfg=yaml.safe_load(open("$CFG"))
cfg["run"]["seed"]=$SEED
out="$CFG.seed$SEED.yaml"
yaml.safe_dump(cfg, open(out,"w"))
print(out)
PY

  CFG2=$CFG.seed$SEED.yaml
  python src/train.py --config $CFG2

  # Example path: adjust if you rename outputs
  CKPT="outputs/$(python -c "import yaml; c=yaml.safe_load(open('$CFG2')); print(c['run']['name'])")/seed${SEED}/best.pt"

  python src/calibrate_ts.py --config $CFG2 --ckpt $CKPT
  python src/calibrate_sats.py --config $CFG2 --ckpt $CKPT
done
