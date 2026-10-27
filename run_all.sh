#!/usr/bin/env bash
# Reproduces every experiment in the manuscript (about 3-4 h on two CPU cores).
set -e
export OMP_NUM_THREADS=1
python3 devdata.py            # commissioning logs (Greensboro)
python3 train_twin.py         # hybrid / physics / no-impedance twins (JAX)
python3 ddmpc.py              # one-step gradient-boosted models for DD-MPC
python3 calib_init.py         # initial conformal scores
python3 run_all.py main       # E2: 6 strategies x 3 sites x 3 scenarios x 4 seasons
python3 forecast.py           # E1: forecasting and state-estimation benchmark
NPROC=1 python3 run_all.py long
python3 run_all.py ablation
python3 run_all.py robust
python3 run_all.py sens
python3 analysis.py
python3 fig_arch.py && python3 fig_method.py
python3 fig_results.py forecast trajectory main_box breakdown ablation robust long
