"""Independent saved-evidence checks, separate from fitting and report rendering."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def bakery_check(case):
    observed = pd.read_csv(case / 'data' / 'observed_sales.csv')
    private = pd.read_csv(case / 'audit_private' / 'test_economic_replay.csv')
    econ = read_json(case / 'economic_results.json')
    config = read_json(ROOT / 'cases' / 'bakery' / 'config.json')
    assert not any('latent' in column or 'expected_demand' in column for column in observed.columns)
    assert not observed.duplicated(['date', 'store_code', 'product_code']).any()
    assert np.allclose(observed.sales_units + observed.leftover_units, observed.production_units)
    costs = np.array(config['costs_eur'])
    salvage = np.array(config['salvage'])
    products = private.product_code.to_numpy(int)
    results = {}
    for policy, summary in econ['policies'].items():
        quantity = private[f'{policy}_quantity'].to_numpy(float)
        demand = private.latent_demand_units.to_numpy(float)
        waste = np.maximum(quantity - demand, 0)
        shortage = np.maximum(demand - quantity, 0)
        contribution = private.price.to_numpy(float) - costs[products]
        loss = waste * (costs[products] - salvage[products]) + shortage * contribution
        realized = np.minimum(quantity, demand) * contribution - waste * (costs[products] - salvage[products])
        assert np.allclose(loss, private[f'{policy}_avoidable_loss_eur'])
        assert abs(loss.sum() - summary['avoidable_loss_eur']) < 1e-6
        assert abs(realized.sum() - summary['realized_contribution_eur']) < 1e-6
        results[policy] = {'recomputed_loss_eur': round(float(loss.sum()), 6), 'identity_passed': True}
    candidates = read_json(case / 'metrics.json')['models']
    forecast_file = case / 'data' / 'test_predictions.csv'
    forecasts = pd.read_csv(forecast_file)
    assert pd.to_datetime(forecasts.origin_date).lt(pd.to_datetime(forecasts.date)).all()
    assert pd.to_datetime(observed.date).max() <= pd.Timestamp('2026-10-14')
    vintages = pd.read_csv(case / 'data' / 'weather_forecast_vintages.csv')
    assert pd.to_datetime(vintages.origin_date).max() <= pd.Timestamp('2026-10-14')
    future = pd.read_csv(case / 'data' / 'future_forecast_2026-10-15_to_21.csv')
    assert future.origin_date.eq('2026-10-14').all()
    assert pd.to_datetime(future.date).min() == pd.Timestamp('2026-10-15')
    return {'policies': results, 'observed_rows': len(observed), 'test_rows': len(forecasts),
            'forecast_origin_checked': True, 'public_data_has_no_latent_labels': True}


def notebooks_check():
    import nbformat
    results = {}
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        assert all(cell.execution_count is not None for cell in code_cells)
        assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
        results[path.name] = {'executed_code_cells': len(code_cells), 'errors': 0}
    assert len(results) == 2
    return results


def beverage_check(case):
    trace = pd.read_csv(case / 'inventory_daily.csv')
    econ = read_json(case / 'economic_results.json')
    assert not trace.duplicated(['date', 'policy', 'key']).any()
    assert (trace[['opening_stock', 'arrivals', 'sales', 'closing_stock', 'on_order']] >= 0).all().all()
    assert np.allclose(trace.opening_stock + trace.arrivals - trace.sales, trace.closing_stock)
    assert np.allclose(trace.sales + trace.lost_units, trace.unbooked_demand + trace.booked_demand)
    assert np.allclose(trace.holding_eur, trace.capital_eur + trace.storage_eur)
    recomputed_net = trace.contribution_eur - trace.holding_eur - trace.order_cost_eur - trace.expediting_eur
    assert np.allclose(recomputed_net, trace.net_contribution_eur)
    totals = trace.groupby('policy').net_contribution_eur.sum()
    for policy in econ['policies']:
        assert abs(totals[policy['policy']] - policy['net_contribution_eur']) < 1e-6
    difference = float(totals['selected_model'] - totals['seasonal_minmax'])
    assert abs(difference - econ['paired_difference']['annual_gain_eur']) < 1e-6
    merged = trace.loc[trace.policy.eq('seasonal_minmax'), ['date', 'key', 'unbooked_demand', 'booked_demand']].merge(
        trace.loc[trace.policy.eq('selected_model'), ['date', 'key', 'unbooked_demand', 'booked_demand']],
        on=['date', 'key'], suffixes=['_base', '_selected'], validate='one_to_one')
    assert merged.unbooked_demand_base.eq(merged.unbooked_demand_selected).all()
    assert merged.booked_demand_base.eq(merged.booked_demand_selected).all()
    forecast = pd.read_csv(case / 'forecast_next_cycle.csv')
    assert pd.to_datetime(forecast.origin).lt(pd.to_datetime(forecast.date)).all()
    assert pd.to_datetime(forecast.origin).eq(pd.Timestamp('2026-10-14')).all()
    assert pd.to_datetime(forecast.date).min() == pd.Timestamp('2026-10-15')
    public = pd.read_csv(case / 'public_observations.csv')
    assert not any(column in public.columns for column in ['unbooked_demand', 'booked_demand', 'latent_demand', 'expected_mean'])
    publicly_observable_exhaustion = pd.to_datetime(public.date).dt.weekday.ne(6) & public.closing_stock.eq(0)
    assert public.stockout.eq(publicly_observable_exhaustion).all()
    return {'stock_balance_passed': True, 'same_latent_demand_for_both_policies': True,
            'no_double_counting_lost_contribution': True, 'independent_annual_delta_eur': difference,
            'public_stockout_uses_only_open_calendar_and_closing_stock': True,
            'future_origin_checked': True, 'observed_rows': len(public), 'trace_rows': len(trace)}


def main():
    checks = {
        'bakery_original': bakery_check(ROOT / 'cases' / 'bakery'),
        'bakery_censored_replication': bakery_check(ROOT / 'cases' / 'bakery' / 'censored_replication'),
        'beverage': beverage_check(ROOT / 'cases' / 'beverage'),
        'notebooks': notebooks_check(),
    }
    for case in ['bakery', 'beverage']:
        for filename in ['summary.json', 'metrics.json', 'economic_results.json']:
            assert (ROOT / 'cases' / case / filename).is_file(), f'Missing {case}/{filename}'
    checks['status'] = 'passed'
    (ROOT / 'validation_receipt.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
