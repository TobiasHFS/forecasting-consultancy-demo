"""Run both reproducible cases; trained outputs are saved inside this project.

Usage: python run_demo.py --case all
Presentation packaging is separate: see presentation/report_builder.mjs.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import time
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / '.cache' / 'matplotlib'))
# Limit shared machine load. Nothing needs GPU, internet or a cloud account.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', 'bakery', 'beverage'], default='all')
    parser.add_argument('--fast', action='store_true', help='Optional smoke run; the delivered evidence uses the complete run.')
    args = parser.parse_args()
    results = {}
    cases = ['bakery', 'beverage'] if args.case == 'all' else [args.case]
    for case in cases:
        print(f'Start {case}: generating, fitting, forecasting and simulating.', flush=True)
        started = time.perf_counter()
        module = importlib.import_module(f'src.{case}')
        with threadpool_limits(limits=2):
            result = module.run(PROJECT / 'cases' / case, fast=args.fast)
        results[case] = {'result': result, 'runtime_seconds': round(time.perf_counter() - started, 2)}
        print(f'Completed {case} in {results[case]["runtime_seconds"]:.1f} seconds.', flush=True)
        if case == 'bakery':
            print('Start bakery censored-demand replication with independently fixed seed.', flush=True)
            replication = importlib.import_module('src.bakery_censored')
            replication_started = time.perf_counter()
            with threadpool_limits(limits=2):
                replication_result = replication.run(PROJECT / 'cases' / 'bakery' / 'censored_replication', fast=args.fast)
            results['bakery_censored_replication'] = {
                'result': replication_result,
                'runtime_seconds': round(time.perf_counter() - replication_started, 2)
            }
            print('Completed independently evaluated bakery replication.', flush=True)
    target = PROJECT / 'run_manifest.json'
    prior = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
    prior.update(results)
    prior['project_mode'] = 'synthetic simulation, not measured customer savings'
    prior['forecast_origin'] = '2026-10-14'
    prior['fast_run'] = args.fast
    target.write_text(json.dumps(prior, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
