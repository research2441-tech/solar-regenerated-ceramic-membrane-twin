# Fraction-resolved digital twin for solar-regenerated ceramic membranes

Code and result files for the manuscript on a fraction-resolved, physics-informed
digital twin with conformal risk control for scheduling solar regeneration of
rice-husk-ash photocatalytic ceramic membranes in decentralised greywater reuse.

Everything is in this one folder. Scripts read inputs from, and write outputs to,
the folder they sit in.

## What is here

| File | Role |
|---|---|
| `membrane_design.py` | Structure-property model of the RHA-kaolin support and separation layer; design selection |
| `weather.py` | Loads the four typical-meteorological-year files (hourly GHI and air temperature) |
| `influent.py` | Stochastic greywater flow and composition, water temperature, reuse demand |
| `plant.py` | Ground-truth plant simulator (10-min sub-steps, compressible cake, impedance, sensor noise) |
| `twin.py` | Fraction-resolved twin (NumPy/JAX), observation model, ensemble Kalman filter |
| `train_twin.py` | Multiple-shooting calibration by reverse-mode differentiation (JAX + Optax) |
| `mpc.py` | Risk-aware scheduler and action-conditional adaptive conformal calibration |
| `ddmpc.py` | Data-driven MPC baseline (gradient-boosted one-step models) |
| `controllers.py` | Rule-based baselines: CONV-MF, PC-FIX, PC-TRIG |
| `sim.py` | Closed-loop harness and run metrics (compliance, permeability, energy, cost, GHG) |
| `devdata.py`, `calib_init.py` | Commissioning logs and initial conformal scores (Greensboro only) |
| `forecast.py` | Forecasting and state-estimation benchmark (E1) |
| `run_all.py` | Closed-loop experiments: `main`, `ablation`, `robust`, `sens`, `long` |
| `analysis.py` | Paired Wilcoxon tests with Holm correction, rank-biserial effect sizes, Friedman ranks |
| `fig_*.py` | Figures of the manuscript |
| `*.jsonl`, `*.csv`, `data_stats.json` | Result files used in the manuscript (one JSON line per run) |
| `*.CSV`, `*.csv`, `*.epw` (weather) | Public TMY3 weather records (see below) |

## Reproducing

```bash
pip install -r requirements.txt
bash run_all.sh
```

The full set of runs takes about three to four hours on two CPU cores. A single
closed-loop run is quick to try:

```python
import pickle, sim, mpc
tp = pickle.load(open("twin_params.pkl", "rb"))      # created by train_twin.py
ci = pickle.load(open("calib_init.pkl", "rb"))       # created by calib_init.py
ctrl = mpc.TwinMPC(tp["hybrid"], calib_init=ci["hybrid"], seed=1)
recs, obs, plant, ctx = sim.run(ctrl, site="CHI", start_day=15, n_days=60, scen="shock", seed=1)
print(sim.summarize(recs, plant, ctx, ctrl.extra_W))
```

## Data sources

- `723170TYA.CSV` (Greensboro, NC) and `703165TY.csv` (Sand Point, AK) are TMY3 files
  distributed with the pvlib-python test data (BSD-3-Clause).
- `USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw` and
  `USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw` are from the EnergyPlus weather set.

Greywater influent is generated, not measured; its statistics follow published
greywater surveys cited in the manuscript.

## Scope

This is a simulation study. Membrane properties come from a structure-property model
calibrated to published ranges, and the plant is a simulator. Results are not field
measurements.

## License

MIT for the code. Weather files keep their original licences.
