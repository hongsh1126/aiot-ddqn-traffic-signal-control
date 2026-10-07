# AIoT-DDQN traffic-signal control under V2X constraints

Code accompanying the IEEE Access article:

> Sunghyuck Hong, "Edge-Intelligent AIoT-Based Traffic Signal Control Using Deep Reinforcement Learning Under V2X Communication Constraints: A Reproducible Lightweight Surrogate Study," *IEEE Access*, vol. 14, 2026. DOI: [10.1109/ACCESS.2026.3731935](https://doi.org/10.1109/ACCESS.2026.3731935)

A self-contained Python surrogate that couples smart-pole sensing, a parametric IEEE 802.11p V2X delay/loss model, and a Double Deep Q-Network (DDQN) signal controller, compared with fixed-time, actuated, MaxPressure and DQN baselines. It needs no SUMO/OMNeT++ and no external dataset, and runs on one CPU core.

## Reproducibility scope
Re-runs reproduce the outputs in `data/results_aiot.json` (the resubmission outputs). They do **not** bitwise reproduce every archived manuscript-source reinforcement-learning value, because RL scalars can vary across environments. The paper discloses this limitation (Sec. IV-J). The archived evidence is kept in `data/`.

## Layout
- `code/` - simulator (`traffic_sim.py`), controllers (`controllers.py`), experiment drivers (`run_experiments.py`, `run_multiseed.py`), figure scripts, and the `run` entry script
- `data/` - archived JSON/CSV outputs
- `environment/` - `requirements.txt` and a `Dockerfile` (Python 3.12, NumPy 2.3.5, SciPy, Matplotlib)

## Run
```bash
cd code
pip install numpy==2.3.5 scipy matplotlib
./run
```

## License
MIT (see `LICENSE`).
