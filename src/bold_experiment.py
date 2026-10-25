"""
Shared machinery for the B⊕LD configuration files (bold_<reweight>_<scaling>_<beta>.py).

A configuration file only sets the B⊕LD options and calls cli(). Each one runs the same
study, in three phases, from bae/src:

    python <config>.py search <id>    η grid for instance / seed <id> (ids 1..5), choosing on
                                      the validation rows
    python <config>.py choose         best η per run, from the search results
    python <config>.py final <id>     chosen η for instance / seed <id> (ids 1..100), report
                                      train, validation and test error
    python <config>.py summary        tables
or all of it:  bash run_config.sh <config>.py

The study (the same for every configuration, so results are comparable):
    Mammals Europe at k = 5, 10, 20, 30            id = initialisation seed, fixed split
    nested synthetic, noise 0.01 .. 0.08, k = 5, 25, 125
                                                   id = instance (and initialisation seed)
    encoders: one layer, and one hidden layer of 256 AND units
    η grid 2^-6 .. 2^4, batch 32, 100 epochs, cosine schedule
Results: results/<config name>/{search, final}/<run>_..._id<id>.json, eta_choice.json, summary.md
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from bold import train_bold
from data import RESULTS_DIR, get_split
from evaluate import evaluate
from greedy import greedy_descent

RUNS = ([{"dataset": "mammals_europe", "k": k} for k in (5, 10, 20, 30)]
        + [{"dataset": "nested", "noise": p, "k": k} for p in (0.01, 0.02, 0.04, 0.08) for k in (5, 25, 125)])
ENCODERS = {"onelayer": [], "256and": [(256, "and")]}
ETA_GRID = [2.0 ** e for e in range(-6, 5)]
TRAINING = dict(batch=32, epochs=100, switch_at=0.25, schedule="cosine")
SEARCH_IDS = range(1, 6)
FINAL_IDS = range(1, 101)
SEARCH_WORKERS = 24            # parallel runs inside one search id (5 ids -> 120 cores)
FINETUNE = True                # final runs: also the exact greedy search from the B⊕LD result


def run_key(run: dict, enc: str) -> str:
    noise = f"_noise{run['noise']}" if run["dataset"] == "nested" else ""
    return f"{run['dataset']}{noise}_k{run['k']}_{enc}"


def _spec(run: dict, ident: int) -> dict:
    if run["dataset"] == "nested":
        return {"dataset": "nested", "instance": ident, "noise": run["noise"]}
    return {"dataset": run["dataset"]}


def _train_eval(config: dict, run: dict, enc: str, eta: float, ident: int, finetune: bool) -> dict:
    Xtr, Xva, Xte, info = get_split(_spec(run, ident))
    t0 = time.time()
    model, history = train_bold(Xtr, run["k"], hidden=ENCODERS[enc], reweight=config["reweight"],
                                window=config.get("window", 2), bp_scale=config["bp_scale"],
                                beta=config["beta"], eta=eta, seed=ident, **TRAINING)
    res, _ = evaluate(model, Xtr, Xva, Xte)
    out = {"config": config, "run": run, "encoder": enc, "eta": eta, "id": ident, "data": info,
           "train_sec": round(time.time() - t0, 1), "result": res, "history": history}
    if finetune:
        t1 = time.time()
        fh = greedy_descent(model, Xtr)
        ft, _ = evaluate(model, Xtr, Xva, Xte)
        out["finetuned"] = {**ft, "sec": round(time.time() - t1, 1), "flips": fh[-1]["flips"]}
    return out


def _save(path: str, out: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)


def _job(args):
    config, run, enc, eta, ident, finetune, path = args
    t = time.time()
    try:
        out = _train_eval(config, run, enc, eta, ident, finetune)
    except Exception as err:
        out = {"config": config, "run": run, "encoder": enc, "eta": eta, "id": ident, "error": repr(err)}
    _save(path, out)
    msg = out.get("error") or (f"train {out['result']['polished']['train']['per_one']:.4f}  "
                               f"val {out['result']['polished']['val']['per_one']:.4f}")
    return f"{os.path.basename(path)}  {msg}  {time.time() - t:.0f}s"


def search(name: str, config: dict, ident: int):
    jobs = []
    for run in RUNS:
        for enc in ENCODERS:
            for eta in ETA_GRID:
                path = os.path.join(RESULTS_DIR, name, "search", f"{run_key(run, enc)}_eta{eta:g}_id{ident:03d}.json")
                if not os.path.exists(path):
                    jobs.append((config, run, enc, eta, ident, False, path))
    print(f"{name} search id {ident}: {len(jobs)} runs to do", flush=True)
    with Pool(SEARCH_WORKERS) as pool:
        for line in pool.imap_unordered(_job, jobs):
            print(line, flush=True)


def _load(folder: str):
    out = []
    for p in glob.glob(os.path.join(folder, "*.json")):
        with open(p) as f:
            r = json.load(f)
        if "error" not in r:
            out.append(r)
    return out


def choose(name: str) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for r in _load(os.path.join(RESULTS_DIR, name, "search")):
        by[run_key(r["run"], r["encoder"])][r["eta"]].append(r["result"]["polished"]["val"]["per_one"])
    choice = {}
    for key, etas in by.items():
        means = {eta: st.mean(v) for eta, v in etas.items()}
        best = min(means, key=means.get)
        choice[key] = {"eta": best, "val_per_one": means[best], "n_ids": len(etas[best]),
                       "grid": {f"{eta:g}": round(v, 5) for eta, v in sorted(means.items())}}
    path = os.path.join(RESULTS_DIR, name, "eta_choice.json")
    _save(path, choice)
    print(f"chose η for {len(choice)} runs -> {path}")
    return choice


def final(name: str, config: dict, ident: int):
    path_c = os.path.join(RESULTS_DIR, name, "eta_choice.json")
    if not os.path.exists(path_c):
        raise SystemExit(f"no {path_c}: run the search and 'choose' first")
    with open(path_c) as f:
        choice = json.load(f)
    t0 = time.time()
    for run in RUNS:
        for enc in ENCODERS:
            key = run_key(run, enc)
            path = os.path.join(RESULTS_DIR, name, "final", f"{key}_id{ident:03d}.json")
            if os.path.exists(path) or key not in choice:
                continue
            print(_job((config, run, enc, choice[key]["eta"], ident, FINETUNE, path)), flush=True)
    print(f"{name} final id {ident} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def _ms(xs, d=4):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    return f"{xs[0]:.{d}f}" if len(xs) == 1 else f"{st.mean(xs):.{d}f} ± {st.stdev(xs):.{d}f}"


def summary(name: str, config: dict):
    rs = _load(os.path.join(RESULTS_DIR, name, "final"))
    choice_path = os.path.join(RESULTS_DIR, name, "eta_choice.json")
    choice = json.load(open(choice_path)) if os.path.exists(choice_path) else {}
    by = defaultdict(list)
    for r in rs:
        by[run_key(r["run"], r["encoder"])].append(r)
    lines = [f"# {name}", "", f"Configuration: `{json.dumps(config)}`", "",
             "Error per 1 in X after polishing C, mean ± sd over ids. η chosen by validation error "
             "(search ids 1–5).", "",
             "| run | η | n | train | validation | test | oracle (test) | dead bits | finetuned test | train s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key in [run_key(r, e) for r in RUNS for e in ENCODERS]:
        g = by.get(key)
        if not g:
            continue
        res = [r["result"] for r in g]
        ft = [r.get("finetuned") for r in g]
        lines.append("| " + " | ".join([
            key, f"{choice.get(key, {}).get('eta', float('nan')):g}", str(len(g)),
            _ms([x["polished"]["train"]["per_one"] for x in res]),
            _ms([x["polished"]["val"]["per_one"] for x in res]),
            _ms([x["polished"]["test"]["per_one"] for x in res]),
            _ms([x["oracle"]["test"]["per_one"] for x in res]),
            f"{st.mean(x['code']['dead'] for x in res):.1f}",
            _ms([f["polished"]["test"]["per_one"] if f else None for f in ft]),
            f"{st.mean(r['train_sec'] for r in g):.0f}"]) + " |")
    if choice:
        etas = [f"{e:g}" for e in ETA_GRID]
        lines += ["", "Validation error over the η grid (mean over search ids):", "",
                  "| run | " + " | ".join(etas) + " |", "|---|" + "---|" * len(etas)]
        for key, c in choice.items():
            lines.append(f"| {key} | " + " | ".join(f"{c['grid'][e]:.3f}" if e in c["grid"] else "–" for e in etas) + " |")
    text = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, name, "summary.md")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(text)


def cli(config_file: str, config: dict):
    name = os.path.splitext(os.path.basename(config_file))[0]
    usage = f"usage: python {name}.py search <id> | choose | final <id> | summary"
    if len(sys.argv) < 2:
        raise SystemExit(usage)
    mode = sys.argv[1]
    if mode in ("search", "final"):
        if len(sys.argv) != 3:
            raise SystemExit(usage)
        (search if mode == "search" else final)(name, config, int(sys.argv[2]))
    elif mode == "choose":
        choose(name)
    elif mode == "summary":
        summary(name, config)
    else:
        raise SystemExit(usage)
