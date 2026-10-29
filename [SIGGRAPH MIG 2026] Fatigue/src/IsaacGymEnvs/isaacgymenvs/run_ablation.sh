#!/usr/bin/env bash
# =============================================================================
# Resolution ablation - four rungs, tornado kick
# =============================================================================
# Each command is written out flat: no shell variables inside the argument list,
# no eval. Combining variables with eval makes the quoting get parsed twice,
# which is how a bare token reaches Hydra's override parser and produces
# "LexerNoViableAltException".
#
# `++key=value` adds the key if missing and overrides it if present, so you do
# not need to know which keys are already in the task YAML.
#
# Lists: double quotes, NO SPACES after the commas. Hydra's grammar rejects
# spaces inside brackets.
#
# Everything outside the per-rung flags is identical across runs: same policy
# checkpoint, same seed, same episode length, same motion file, 8 envs.
#
# Usage:  bash run_ablation.sh            # all four
#         bash run_ablation.sh loaddep    # one rung: uniform|loaddep|pergroup|fgs
# =============================================================================

set -e
RUN=${1:-all}

# -----------------------------------------------------------------------------
# rung (i) - per-DoF 3CC-r, uniform coefficients   [= Cheema et al. 2023]
# -----------------------------------------------------------------------------
if [ "$RUN" = "all" ] || [ "$RUN" = "uniform" ]; then
echo "=== rung (i) uniform 3CC-r ==="
python train.py seed=477 task=HumanoidAMPHandsFatigue \
  train=HumanoidAMPPPOLowGPFatigue headless=True num_envs=8 test=True \
  checkpoint=runs/wushu_fatigue/nn/wushu_fatigue_6000.pth \
  task.env.motion_file=amp_humanoid_wushu_kick.npy \
  task.env.episodeLength=2000 \
  task.env.useFatigue=True \
  task.env.visualizeFatigue=False \
  task.env.showSkyWall=False \
  ++task.env.useFGS=False \
  ++task.env.localGroupFatigue=False \
  ++task.env.loadDependentRate=False \
  ++task.env.fatigueF=2.0 \
  ++task.env.fatigueR=0.05 \
  ++task.env.fatigue_r=1.0 \
  ++task.env.logRC=True \
  ++task.env.logRCEnvs=8 \
  ++task.env.logRCPath=rc_log_uniform_3CCr.csv \
  ++task.env.logDuty=False \
  ++task.env.timeFatigue=False
fi

# -----------------------------------------------------------------------------
# rung (0) - load-dependent SINGLE compartment      [the reviewer's baseline]
# -----------------------------------------------------------------------------
# One pool per DoF, but its rate scales with instantaneous load: F rises and R
# falls as load increases, spanning the same range as the FGS fiber multipliers.
# Reproduces FGS's rate modulation WITHOUT a reserve to withhold.
if [ "$RUN" = "all" ] || [ "$RUN" = "loaddep" ]; then
echo "=== rung (0) load-dependent 3CC-r ==="
python train.py seed=477 task=HumanoidAMPHandsFatigue \
  train=HumanoidAMPPPOLowGPFatigue headless=True num_envs=8 test=True \
  checkpoint=runs/wushu_fatigue/nn/wushu_fatigue_6000.pth \
  task.env.motion_file=amp_humanoid_wushu_kick.npy \
  task.env.episodeLength=2000 \
  task.env.useFatigue=True \
  task.env.visualizeFatigue=False \
  task.env.showSkyWall=False \
  ++task.env.useFGS=False \
  ++task.env.localGroupFatigue=True \
  ++task.env.loadDependentRate=True \
  ++task.env.loadDepSpanF=0.689 \
  ++task.env.loadDepSpanR=-0.220 \
  ++task.env.loadDepCenter=0.5 \
  ++task.env.fatigue_F_groups="[0.848,3.394,3.394,1.697,1.697]" \
  ++task.env.fatigue_R_groups="[0.085,0.424,0.424,0.085,0.085]" \
  ++task.env.fatigue_r_groups="[1.0,1.0,1.0,1.0,1.0]" \
  ++task.env.logRC=True \
  ++task.env.logRCEnvs=8 \
  ++task.env.logRCPath=rc_log_load_dep.csv \
  ++task.env.logDuty=False \
  ++task.env.timeFatigue=False
fi

# -----------------------------------------------------------------------------
# rung (ii) - per-group coefficients, single compartment
# -----------------------------------------------------------------------------
if [ "$RUN" = "all" ] || [ "$RUN" = "pergroup" ]; then
echo "=== rung (ii) per-group ==="
python train.py seed=477 task=HumanoidAMPHandsFatigue \
  train=HumanoidAMPPPOLowGPFatigue headless=True num_envs=8 test=True \
  checkpoint=runs/wushu_fatigue/nn/wushu_fatigue_6000.pth \
  task.env.motion_file=amp_humanoid_wushu_kick.npy \
  task.env.episodeLength=2000 \
  task.env.useFatigue=True \
  task.env.visualizeFatigue=False \
  task.env.showSkyWall=False \
  ++task.env.useFGS=False \
  ++task.env.localGroupFatigue=True \
  ++task.env.loadDependentRate=False \
  ++task.env.fatigue_F_groups="[0.848,3.394,3.394,1.697,1.697]" \
  ++task.env.fatigue_R_groups="[0.085,0.424,0.424,0.085,0.085]" \
  ++task.env.fatigue_r_groups="[1.0,1.0,1.0,1.0,1.0]" \
  ++task.env.logRC=True \
  ++task.env.logRCEnvs=8 \
  ++task.env.logRCPath=rc_log_per_group.csv \
  ++task.env.logDuty=False \
  ++task.env.timeFatigue=False
fi

# -----------------------------------------------------------------------------
# rung (iii) - full FGS
# -----------------------------------------------------------------------------
# P and the multipliers are set explicitly so this reproduces the paper numbers
# regardless of what the file defaults happen to be after a sweep.
if [ "$RUN" = "all" ] || [ "$RUN" = "fgs" ]; then
echo "=== rung (iii) FGS ==="
python train.py seed=477 task=HumanoidAMPHandsFatigue \
  train=HumanoidAMPPPOLowGPFatigue headless=True num_envs=8 test=True \
  checkpoint=runs/wushu_fatigue/nn/wushu_fatigue_6000.pth \
  task.env.motion_file=amp_humanoid_wushu_kick.npy \
  task.env.episodeLength=2000 \
  task.env.useFatigue=True \
  task.env.visualizeFatigue=False \
  task.env.showSkyWall=False \
  ++task.env.useFGS=True \
  ++task.env.localGroupFatigue=True \
  ++task.env.loadDependentRate=False \
  ++task.env.fatigue_F_groups="[0.848,3.394,3.394,1.697,1.697]" \
  ++task.env.fatigue_R_groups="[0.085,0.424,0.424,0.085,0.085]" \
  ++task.env.fatigue_r_groups="[1.0,1.0,1.0,1.0,1.0]" \
  ++task.env.mu_percentages_per_dof="[[0.4,0.3,0.3]]" \
  ++task.env.mu_be_recruitment_thresholds="[33.3,66.6]" \
  ++task.env.F_mu_multipliers="[0.690,1.034,1.379]" \
  ++task.env.R_mu_multipliers="[1.099,0.989,0.879]" \
  ++task.env.r_mu_multipliers="[1.136,1.023,0.796]" \
  ++task.env.logRC=True \
  ++task.env.logRCEnvs=8 \
  ++task.env.logRCPath=rc_log_FGS.csv \
  ++task.env.logDuty=True \
  ++task.env.logDutyWindow=200 \
  ++task.env.logDutyPath=duty_log_FGS.csv \
  ++task.env.timeFatigue=False
fi

echo
echo "Done. Run check_exposure.py BEFORE interpreting any result."
echo "Rungs (i)-(iii) previously gave knee exposure 0.698-0.704; the"
echo "load-dependent rung must land in that range or the comparison is void."
