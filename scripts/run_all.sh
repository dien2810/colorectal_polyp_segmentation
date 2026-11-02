#!/usr/bin/env bash
# Full experiment pipeline (report Sections 6-7). Run from the project root on a GPU machine.
#   DATA_ROOT=/content/data/polyp bash scripts/run_all.sh
# Re-running is safe: training resumes from last.pt, finished runs are skipped.
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-data/polyp}
MODELS=${MODELS:-"unet multiresunet cenet transunet swinunet"}
REFERENCE=${REFERENCE:-cenet}          # model used for the p-value column (best CNN after week 4)
BEST=${BEST:-cenet}                    # best baseline used for the ablation (RQ3)
EXTRA_OPTS=${EXTRA_OPTS:-}              # e.g. "train.epochs=1" for a quick test

python scripts/prepare_splits.py --root "$DATA_ROOT"
python scripts/check_duplicates.py --root "$DATA_ROOT"

for m in $MODELS; do
  python scripts/train.py --config "configs/$m.yaml" --opts data.root="$DATA_ROOT" $EXTRA_OPTS
  python scripts/evaluate.py --run "outputs/runs/$m" --save-pred
done

for a in c3_domain_aug c2_boundary c2_c3; do
  python scripts/train.py --config "configs/$BEST.yaml" "configs/ablation/$a.yaml" --opts data.root="$DATA_ROOT" $EXTRA_OPTS
  python scripts/evaluate.py --run "outputs/runs/${BEST}_$a"
done

python scripts/stats_tests.py --models $MODELS --reference "$REFERENCE" \
  --ablation "$BEST" "${BEST}_c3_domain_aug" "${BEST}_c2_boundary" "${BEST}_c2_c3"
python scripts/benchmark.py --configs $(for m in $MODELS; do echo "configs/$m.yaml"; done)
python scripts/plot_dataset_examples.py --root "$DATA_ROOT"
python scripts/plot_report_figures.py --models $MODELS --reference "$REFERENCE" --root "$DATA_ROOT"
echo "All done: tables in outputs/tables, figures in report/figures"
