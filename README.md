# UAV Telemetry Poisoning Audit

This repository holds the code and result files for the paper
**"Training-Data Auditing and Poisoning Robustness in UAV Telemetry-Based Intrusion Detection"**.

It covers:
- the audited Random Forest baseline on PX4 flight telemetry;
- four training-data poisoning attacks;
- five poison-detection signals and three recovery strategies;
- the network-track replication on UAVIDS-2025 and ISOT.

The datasets are **not** included. They are public, but their licences and access terms belong to their owners (see [Data](#data)).

## Repository layout

```
step3/                         telemetry track (UAV Attack Dataset)
  config.py                    every setting: windows (50 samples, stride 2), airframes, seeds, labels
  experiments/                 poisoning attacks, detectors, defences, victim model
  *.py                         audit scripts (protocol ladder, label audit, prevalence, false alarms, baselines)
  results/cases/poison_w50/    every result file reported for the telemetry track
  figures/cases/poison_w50/    figures generated from those results
step2/experiments/optimizers.py  metaheuristics reused by step3/experiments/run_defense_opt.py
framework/
  common/                      shared loaders, unified taxonomy, sealing, statistics
  stage0_sealing/              seals a held-out partition of each dataset before any experiment
  stage3_poisoning/            network-track replication (UAVIDS-2025, ISOT)
  results/stage0/              sealing manifest (the loaders read the sealed cut points from it)
  results/stage3/              network-track and named-metric result files
```

## Data

Download each dataset from its official source, then place it at the path shown.
The code expects these exact folder names, which are the names of the original downloads.

| Dataset | Source | Expected path |
|---|---|---|
| UAV Attack Dataset | IEEE DataPort, DOI [10.21227/00dg-0d12](https://doi.org/10.21227/00dg-0d12) | `data/UAV Attack Data-20260702T145753Z-3-001/UAV Attack Data/` (with `Simulated - OTU Survey/` inside) |
| UAVIDS-2025 | IEEE DataPort, DOI [10.21227/j5p4-zt27](https://doi.org/10.21227/j5p4-zt27) | `data/UAV IDS-2025-20260702T145125Z-3-001/UAV IDS-2025/UAVIDS-2025.csv` |
| ISOT Drone Dataset | [ISOT lab, University of Victoria](https://onlineacademiccommunity.uvic.ca/isot/2024/12/05/drone-datasets/) | `data/ISOT Drone Dataset-20260702T145051Z-3-012/ISOT Drone Dataset/Dataset/new_feature_csv/` (one sub-folder per class) |

Notes on the data:
- **Telemetry:** 16 of the dataset's 18 simulated flights are used. The flights Ping DoS on PLANE-SITL and GPS Spoofing on TAIL-SITL lack the EKF innovation log the detector needs. Live flights are not used.
- **ISOT:** the full public release is used (2,945,986 feature rows, 137 CSV files in 10 folders).
  - The release folder `Password Cracking` corresponds to the description's *Deauthentication Attack*.
  - The grouping into five unified classes is the authors' own; see `framework/common/taxonomy.py`.

## Environment

- Python 3.12.
- Install the packages with `pip install -r requirements.txt`. The versions listed are the ones used to produce the results.
- No GPU is needed. The telemetry victim model and detectors run on NumPy and scikit-learn.

## Running

Run every command from the repository root.

```bash
# 0. Seal held-out partitions (writes framework/results/stage0/; the manifest shipped here is the one used in the paper)
python framework/stage0_sealing/run.py

# 1. Telemetry track: poisoning, detection, defences, statistics, figures
python step3/run_all_p3.py
python step3/experiments/run_targeted_sweep.py
python step3/experiments/run_clean_label_sweep.py
python step3/experiments/run_trigger_placement.py
python step3/experiments/run_defense_clean_control.py

# 2. Audit and operational evaluations
python step3/protocol_ladder.py
python step3/published_ladder.py
python step3/groundtruth_labels.py
python step3/groundtruth_rawmask.py
python step3/prevalence_eval.py
python step3/fa_event_eval.py
python step3/fa_budget_eval.py
python step3/transition_window_check.py
python step3/simple_baseline.py
python step3/variance_sensitivity.py
python step3/recompute_stats_clustered.py

# 3. Network-track replication (five seeds, as reported) and the clean-model trigger baseline
python framework/stage3_poisoning/network_poison.py --seeds 5
python framework/stage3_poisoning/clean_backdoor_baseline.py
python framework/stage3_poisoning/run.py --legacy-only
```

A few practical points:
- **Restartable:** scripts resume from their cache (`step3/experiments/.cache_p3/`) and from rows already written.
- **Quick check:** set `P3_SMOKE=1` (telemetry) or `FW_QUICK=1` (framework) for a fast run that only checks the code works. Its outputs are not results.
- **Labels:** `step3/README_STEP3.md` describes the telemetry track in detail, in Arabic.

## Where each reported result comes from

| Paper section | Script | Result file |
|---|---|---|
| 3.1 / 4.1 protocol-rigor ladder | `step3/protocol_ladder.py`, `step3/published_ladder.py`, `step3/make_published_ladder.py` | `p3_protocol_ladder.csv`, `p3_published_ladder*.csv`, `p3_ladder_combined.csv` |
| Ladder confound and balancing checks | `step3/ladder_confound_check.py`, `step3/balance_sensitivity.py` | `p3_ladder_confound.csv`, `p3_balance_sensitivity.csv` |
| Label-integrity audit | `step3/groundtruth_labels.py`, `step3/groundtruth_rawmask.py` | `p3_groundtruth_agreement*.csv`, `_rawmask/` |
| Native prevalence and false alarms | `step3/prevalence_eval.py`, `step3/fa_event_eval.py`, `step3/fa_budget_eval.py` | `p3_prevalence_eval.csv`, `p3_fa_events.csv`, `p3_fa_budget.csv` |
| Transition windows | `step3/transition_window_check.py` | `p3_transition_windows.csv` |
| Simple-rule baselines, innovation-variance check | `step3/simple_baseline.py`, `step3/variance_sensitivity.py` | `p3_simple_baseline.csv`, `p3_variance_sensitivity.csv` |
| 4.2 poisoning, detection, defences | `step3/experiments/run_p3_benchmark.py` | `p3_clean_baseline.csv`, `p3_damage_curve.csv`, `p3_detection.csv`, `p3_defense.csv`, `p3_stats.json` |
| Targeted / clean-label sweeps, trigger placement | `run_targeted_sweep.py`, `run_clean_label_sweep.py`, `run_trigger_placement.py` | `p3_targeted_sweep.csv`, `p3_clean_label_sweep.csv`, `p3_trigger_placement.csv` |
| Defence on clean data; tuned defence | `run_defense_clean_control.py`, `run_defense_opt.py` | `p3_defense_clean_control.csv`, `p3_defense_opt.csv`, `p3_opt_stats.json` |
| Airframe-clustered statistics | `step3/recompute_stats_clustered.py` | `p3_stats_clustered.json` |
| 4.3 network-track replication | `framework/stage3_poisoning/network_poison.py` | `framework/results/stage3/s3_network_*.csv` |
| Clean-model trigger baseline (telemetry) | `framework/stage3_poisoning/clean_backdoor_baseline.py` | `s3_backdoor_clean_baseline*.csv` |

Telemetry result files are in `step3/results/cases/poison_w50/`. `_mlpvictim/` holds the MLP-victim comparison.

## Citation

If you use this code, please cite the paper. The full reference will be added on publication.

## Licence

This project is licensed under the MIT License (see LICENSE).
