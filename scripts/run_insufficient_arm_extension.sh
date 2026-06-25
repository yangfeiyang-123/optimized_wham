#!/usr/bin/env bash
set -u
set -o pipefail

REPO="/data3/yangfeiyang/WorkSpace/optimized_wham"
IN="$REPO/BadmintonVideos/forehand_clear/wrong"
OUT="$REPO/output/forehand_clear/InsufficientArmExtension"
FPS="${FPS:-60}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-/data3/yangfeiyang/conda_envs/wham/bin/python}"
POSE_BACKEND="${POSE_BACKEND:-vitpose}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
TRACK_ID="${TRACK_ID:-merge}"
POSE_ITERATIONS="${POSE_ITERATIONS:-80}"

cd "$REPO" || exit 1
export PYTHONPATH="$REPO/third-party/ViTPose${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

STATUS="$OUT/batch_status.tsv"
BATCH_LOG="$OUT/batch.log"
printf "timestamp\tindex\ttotal\tvideo\tstatus\tstage\tlog\n" > "$STATUS"
: > "$BATCH_LOG"

mapfile -d '' VIDEOS < <(find "$IN" -maxdepth 1 -type f -name '*.mp4' -print0 | sort -z)
TOTAL="${#VIDEOS[@]}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

record_status() {
  local index="$1"
  local video="$2"
  local status="$3"
  local stage="$4"
  local log="$5"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(timestamp)" "$index" "$TOTAL" "$video" "$status" "$stage" "$log" | tee -a "$STATUS"
}

run_stage() {
  local stage="$1"
  local target="$2"
  local log="$3"
  shift 3

  if [[ -s "$target" ]]; then
    echo "[$(timestamp)] skip $stage; exists: $target" | tee -a "$log"
    return 0
  fi

  echo "[$(timestamp)] start $stage" | tee -a "$log"
  echo "\$ $*" >> "$log"
  "$@" >> "$log" 2>&1
  local rc="$?"
  if [[ "$rc" -ne 0 ]]; then
    echo "[$(timestamp)] failed $stage rc=$rc" | tee -a "$log"
    return "$rc"
  fi
  if [[ ! -s "$target" ]]; then
    echo "[$(timestamp)] failed $stage; missing target: $target" | tee -a "$log"
    return 2
  fi
  echo "[$(timestamp)] done $stage" | tee -a "$log"
  return 0
}

write_source_json() {
  local path="$1"
  local video="$2"
  local seq_out="$3"
  "$PYTHON" - "$path" "$video" "$seq_out" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
video = sys.argv[2]
seq_out = Path(sys.argv[3])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(
        {
            "video": video,
            "wham_pkl": str(seq_out / "wham_output.pkl"),
            "canonical_pkl": str(seq_out / "canonical_wham_output.pkl"),
            "world_grounded_pkl": str(seq_out / "world_grounded" / "optimized_canonical_wham_output.pkl"),
            "corrected_smpl_pkl": str(seq_out / "lower_body_corrected" / "corrected_smpl.pkl"),
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
PY
}

echo "[$(timestamp)] found $TOTAL videos under $IN" | tee -a "$BATCH_LOG"

INDEX=0
for video in "${VIDEOS[@]}"; do
  INDEX=$((INDEX + 1))
  base="$(basename "$video")"
  stem="${base%.*}"
  seq_out="$OUT/$stem"
  log="$seq_out/pipeline.log"

  mkdir -p "$seq_out"
  : > "$log"
  record_status "$INDEX" "$base" "START" "all" "$log" | tee -a "$BATCH_LOG" >/dev/null

  raw_pkl="$seq_out/wham_output.pkl"
  canonical_pkl="$seq_out/canonical_wham_output.pkl"
  fixed_report="$seq_out/fixed_beta_report.json"
  world_pkl="$seq_out/world_grounded/optimized_canonical_wham_output.pkl"
  corrected_pkl="$seq_out/lower_body_corrected/corrected_smpl.pkl"
  reference_manifest="$seq_out/reference_bundle/manifest.json"
  source_json="$seq_out/reference_bundle/source.json"

  if ! run_stage "raw_wham" "$raw_pkl" "$log" \
    "$PYTHON" demo.py \
      --video "$video" \
      --output_pth "$OUT" \
      --save_pkl \
      --pose-backend "$POSE_BACKEND"; then
    record_status "$INDEX" "$base" "FAILED" "raw_wham" "$log" | tee -a "$BATCH_LOG" >/dev/null
    continue
  fi

  if ! run_stage "fixed_beta" "$canonical_pkl" "$log" \
    "$PYTHON" scripts/canonicalize_wham_fixed_beta.py \
      --wham-pkl "$raw_pkl" \
      --out-pkl "$canonical_pkl" \
      --report "$fixed_report" \
      --beta-out "$seq_out/beta_fixed.npy" \
      --tracking-results "$seq_out/tracking_results.pth" \
      --device "$DEVICE" \
      --chunk-size "$CHUNK_SIZE" \
      --keep-percentile 80; then
    record_status "$INDEX" "$base" "FAILED" "fixed_beta" "$log" | tee -a "$BATCH_LOG" >/dev/null
    continue
  fi

  if ! run_stage "world_grounded" "$world_pkl" "$log" \
    "$PYTHON" scripts/world_grounded_smpl_optimizer.py \
      --input-pkl "$canonical_pkl" \
      --out-dir "$seq_out/world_grounded" \
      --fps "$FPS" \
      --track-id "$TRACK_ID" \
      --device "$DEVICE" \
      --root-smooth-axes y; then
    record_status "$INDEX" "$base" "FAILED" "world_grounded" "$log" | tee -a "$BATCH_LOG" >/dev/null
    continue
  fi

  if ! run_stage "lower_body_corrected" "$corrected_pkl" "$log" \
    "$PYTHON" scripts/optimize_smpl_lower_body.py \
      --input-pkl "$world_pkl" \
      --out-dir "$seq_out/lower_body_corrected" \
      --fps "$FPS" \
      --track-id "$TRACK_ID" \
      --max-root-y-shift 0.25 \
      --device "$DEVICE" \
      --enable-pose-pass \
      --pose-iterations "$POSE_ITERATIONS"; then
    record_status "$INDEX" "$base" "FAILED" "lower_body_corrected" "$log" | tee -a "$BATCH_LOG" >/dev/null
    continue
  fi

  if [[ ! -s "$source_json" ]]; then
    write_source_json "$source_json" "$video" "$seq_out" >> "$log" 2>&1
  fi

  if ! run_stage "reference_bundle" "$reference_manifest" "$log" \
    "$PYTHON" scripts/export_contact_preserving_reference.py \
      --input-pkl "$corrected_pkl" \
      --out-dir "$seq_out/reference_bundle" \
      --sequence "$stem" \
      --fps "$FPS" \
      --track-id "$TRACK_ID" \
      --quality-report "$seq_out/lower_body_corrected/validation_summary.json" \
      --source-json "$source_json" \
      --stance-enter-threshold 0.55 \
      --stance-exit-threshold 0.30 \
      --root-smooth-axes y; then
    record_status "$INDEX" "$base" "FAILED" "reference_bundle" "$log" | tee -a "$BATCH_LOG" >/dev/null
    continue
  fi

  record_status "$INDEX" "$base" "DONE" "all" "$log" | tee -a "$BATCH_LOG" >/dev/null
done

echo "[$(timestamp)] batch finished" | tee -a "$BATCH_LOG"
