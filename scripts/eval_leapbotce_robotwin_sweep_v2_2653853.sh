#!/usr/bin/env bash
# Thin wrapper: same cluster sweep driver, but the per-lane result copy takes
# the run dir the eval printed instead of globbing a lane-clock timestamp.
# 2026-11-08: the old lane copied a SIBLING lane's _result_clean.txt whenever
# two lanes' timestamps collided (v2 D0 wave 20/50 cells, D60/80 probe 13/20).
# Everything else is forwarded untouched — see
# baseline_delay/leapbotce_robotwin_sweep_lane_v2_2653853.sh for the fix and
# .bak_20261108 for the archived original.
set -u
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LANE_DRIVER="${LANE_DRIVER:-/dahuafs/userdata/2653853/code/baseline_delay/leapbotce_robotwin_sweep_lane_v2_2653853.sh}"
exec bash "${_HERE}/eval_leapbotce_robotwin_sweep_2653853.sh" "$@"
