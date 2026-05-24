#!/usr/bin/env bash
set -euo pipefail

# One-off cleanup helper for stale L22MG/LX521 baffle-modeling docs artifacts.
#
# Default mode is dry-run. Use --apply to remove stale untracked top-level
# docs/ artifacts. This script intentionally ignores tracked docs changes and
# refuses to run if any untracked docs entry is not classified.

APPLY=0
AUDIT=0

for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --audit) AUDIT=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: ./delete_stale.sh [--audit] [--apply]

Dry-run by default. Deletes stale untracked docs/ artifacts only when --apply
is supplied. Run ./move_baffle_related.sh --apply first if you want to keep
the relevant baffle reports under baffle_modeling/.
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")"

untracked_docs_top_entries() {
  python3 - <<'PY'
import subprocess
from pathlib import Path

raw = subprocess.check_output(
    ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all", "docs"]
)
entries = raw.decode("utf-8", "surrogateescape").split("\0")
tops = set()
for entry in entries:
    if not entry:
        continue
    status = entry[:2]
    path = entry[3:]
    if status == "??" and path.startswith("docs/"):
        parts = Path(path).parts
        tops.add("/".join(parts[:2]) if len(parts) >= 2 else path)
for path in sorted(tops):
    print(path)
PY
}

RELEVANT_EXACT=(
  "docs/l22mg-current-status"
  "docs/l22mg-juan-acceptance-scorecard"
  "docs/l22mg-measurement-geometry-provenance"
  "docs/l22mg-juan-top-populated-silent-self-contained-report"
  "docs/l22mg-bem-juan-top-populated-silent-wmax-model-agreement"
  "docs/l22mg-bem-juan-top-populated-silent-compact2-wmax-abcd"
  "docs/l22mg-bem-juan-top-populated-silent-modal-full2-wmax-abcd"
  "docs/l22mg-bem-juan-top-populated-silent-axisym-directivity-wmax-abcd"
  "docs/l22mg-juan-top-target-null-audit"
  "docs/l22mg-juan-top-front-rear-consistency"
  "docs/l22mg-juan-top-target-polar-quality"
  "docs/l22mg-target-robustness"
  "docs/l22mg-source-model-cross-validation"
  "docs/l22mg-source-candidate-decision"
  "docs/l22mg-offplane-source-ambiguity"
  "docs/l22mg-juan-top-lowfreq-source-support-sweep"
  "docs/l22mg-juan-top-passive-state-audit"
  "docs/l22mg-juan-top-passive-state-band-sensitivity"
  "docs/l22mg-juan-top-lowfreq-geometry-sensitivity"
  "docs/l22mg-bem-upgrade-path"
  "docs/l22mg-juan-top-q7near-300hz-mesh-detail-diagnostic"
)

is_relevant_exact() {
  local path="$1"
  local item
  for item in "${RELEVANT_EXACT[@]}"; do
    [[ "$path" == "$item" ]] && return 0
  done
  return 1
}

is_relevant() {
  local path="$1"
  is_relevant_exact "$path"
}

is_stale() {
  local path="$1"
  case "$path" in
    docs/andres-*|\
    docs/juan-naked-ir-peak-oneoff|\
    docs/l22mg-above5-*|\
    docs/l22mg-active-*|\
    docs/l22mg-andres-*|\
    docs/l22mg-asymmetric-*|\
    docs/l22mg-auto-*|\
    docs/l22mg-baffle-bem-*|\
    docs/l22mg-baffle-bem-*-smoke|\
    docs/l22mg-baffle-bem-driverface-1200|\
    docs/l22mg-baffle-bem-split-*|\
    docs/l22mg-baffle-sim*|\
    docs/l22mg-bem-juan-top-populated-silent-*|\
    docs/l22mg-bem-active-*|\
    docs/l22mg-bem-andres-*|\
    docs/l22mg-bem-auto-*|\
    docs/l22mg-bem-axisymmetric-*|\
    docs/l22mg-bem-convergence-*|\
    docs/l22mg-bem-cutout-*|\
    docs/l22mg-bem-feature-*|\
    docs/l22mg-bem-juan-top-axisymmetric-*|\
    docs/l22mg-bem-juan-top-h1659-*|\
    docs/l22mg-bem-linear-*|\
    docs/l22mg-bem-local-*|\
    docs/l22mg-bem-matrix-*|\
    docs/l22mg-bem-mesh-*|\
    docs/l22mg-bem-passive-*|\
    docs/l22mg-bem-q7-near-*|\
    docs/l22mg-bem-source-*|\
    docs/l22mg-bem-split-*|\
    docs/l22mg-bem-stl-*|\
    docs/l22mg-bem-wide-*|\
    docs/l22mg-contour-*|\
    docs/l22mg-coupled-*|\
    docs/l22mg-diagnostic-*|\
    docs/l22mg-direct-*|\
    docs/l22mg-driver-frame-*|\
    docs/l22mg-driver-stl|\
    docs/l22mg-driver-stl-metadata-*|\
    docs/l22mg-edge-*|\
    docs/l22mg-first-lobe-*|\
    docs/l22mg-high-angle-residual*|\
    docs/l22mg-hotband-variant-comparison.csv|\
    docs/l22mg-ir-peak-*|\
    docs/l22mg-juan-top-lowfreq-*|\
    docs/l22mg-juan-top-populated-silent-*|\
    docs/l22mg-juan-top-q7near-300hz-mesh-detail|\
    docs/l22mg-juan-top-target-null-gate-summary|\
    docs/l22mg-linkwitz-baffle-dipole-theory|\
    docs/l22mg-linear-solver-*|\
    docs/l22mg-lowfreq-*|\
    docs/l22mg-lx521-system-sanity|\
    docs/l22mg-matrix-free-*|\
    docs/l22mg-mesh-*|\
    docs/l22mg-mic-*|\
    docs/l22mg-passive-*|\
    docs/l22mg-physical-rear-filter|\
    docs/l22mg-published-null-context|\
    docs/l22mg-rear-sign-convention|\
    docs/l22mg-requirement-trace|\
    docs/l22mg-retimed-*|\
    docs/l22mg-source-cv-*|\
    docs/l22mg-source-family-high-angle-matrix|\
    docs/l22mg-source-geometry-diagnostics|\
    docs/l22mg-source-model-cross-validation-current-smoke|\
    docs/l22mg-source-model-audit*|\
    docs/l22mg-source-offplane-*|\
    docs/l22mg-source-smoothness-*|\
    docs/l22mg-source-support-*|\
    docs/l22mg-target-polar-quality|\
    docs/l22mg-validation-decomposition-first-lobe-target|\
    docs/l22mg-validation-decomposition-current|\
    docs/l22mg-validation-decomposition-profile-ring-compact-svd|\
    docs/l22mg-validation-decomposition-smoke|\
    docs/l22mg-validation-gate-*|\
    docs/l22mg-width-sweep-deliverable|\
    docs/l22mg-width-sweep-stale-*|\
    docs/l22mg-*smoke*|\
    docs/l22mg-*hotspot*|\
    docs/l22mg-*hotband*|\
    docs/l22mg-*probe*|\
    docs/l22mg-*-gates)
      return 0
      ;;
  esac
  return 1
}

relevant=()
stale=()
unknown=()

while IFS= read -r path; do
  if is_relevant "$path"; then
    relevant+=("$path")
  elif is_stale "$path"; then
    stale+=("$path")
  else
    unknown+=("$path")
  fi
done < <(untracked_docs_top_entries)

printf 'untracked docs classified: relevant=%d stale=%d unknown=%d\n' \
  "${#relevant[@]}" "${#stale[@]}" "${#unknown[@]}"

if (( AUDIT )); then
  printf '\n[relevant -> keep/move with move_baffle_related.sh]\n'
  printf '%s\n' "${relevant[@]}"
  printf '\n[stale -> delete]\n'
  printf '%s\n' "${stale[@]}"
  if ((${#unknown[@]})); then
    printf '\n[unknown -> classify before applying]\n'
    printf '%s\n' "${unknown[@]}"
  fi
fi

if ((${#unknown[@]})); then
  echo "Refusing to continue: unclassified untracked docs entries exist." >&2
  exit 1
fi

if ((${#stale[@]} == 0)); then
  echo "No stale untracked docs entries to delete."
  exit 0
fi

if (( APPLY == 0 )); then
  echo
  echo "Dry-run. Would delete these entries:"
  for path in "${stale[@]}"; do
    printf '  %s\n' "$path"
  done
  echo
  echo "Re-run with --apply to delete."
  exit 0
fi

for path in "${stale[@]}"; do
  rm -rf -- "$path"
done

echo "Deleted ${#stale[@]} stale untracked docs entries."
