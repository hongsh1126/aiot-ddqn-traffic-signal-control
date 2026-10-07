"""Independent repetitions of the archived Medium-demand protocol.

No controller or simulator behavior is modified. Evaluation retains epsilon
0.05 and online updates, as in the archived run_experiments.py.
"""
import csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np
from run_experiments import train_rl, run_single, WARMUP, EVAL

ROOT = Path(__file__).resolve().parent
names = ['Fixed-Time','Actuated','MaxPressure','DQN','IntelliLight','AIoT-DDQN']
seeds = list(range(31000, 31010))
rows = []
start = time.perf_counter()
for seed in seeds:
    for name in names:
        ctrl = train_rl(name, 300, seed=seed)
        result = run_single(ctrl, 300, WARMUP+EVAL, seed=seed+100)
        rows.append(dict(base_seed=seed, training_arrival_seed=seed+1,
                         evaluation_arrival_seed=seed+100, controller=name, **result))
    print(f'Completed seed {seed}', flush=True)
summary = {}
for name in names:
    summary[name] = {}
    for metric in ['avg_wait_s','queue_veh','throughput_vph','emission_g_per_veh']:
        values = [r[metric] for r in rows if r['controller']==name]
        summary[name][metric] = dict(mean=float(np.mean(values)), sd=float(np.std(values,ddof=1)),n=len(values))
payload = dict(protocol='Archived online-adapting evaluation, Medium 300 vph/lane; 1800 s training; 1000 s evaluation; 200 s warmup',
               seeds=seeds, elapsed_s=time.perf_counter()-start, python=platform.python_version(), numpy=np.__version__,
               source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['controllers.py','traffic_sim.py','run_experiments.py']},
               rows=rows,summary=summary)
(ROOT/'multiseed_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
with (ROOT/'multiseed_results.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,indent=2),flush=True)
