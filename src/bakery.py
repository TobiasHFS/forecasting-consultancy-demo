"""Reproducible bakery forecasting demo. All companies and data are fictional.

The latent demand table is an audit-only simulator outcome; modelling functions
receive observed POS sales, production and stockout flags, never latent demand.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/"cases"/"bakery"/".cache"/"matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261014
STORES = ["Altstadt", "Bahnhof", "Wohnviertel"]
PRODUCTS = ["Kaisersemmel", "Breze", "Croissant", "Vollkornbrot", "Roggenbrot", "Plunder", "Belegtes_Broetchen", "Warmer_Snack"]
BASE = np.array([170, 108, 43, 23, 20, 34, 37, 25.], dtype=float)
PRICES = np.array([0.65, 1.15, 2.25, 4.90, 4.40, 2.40, 3.60, 4.20])
COSTS = np.array([0.22, 0.38, 0.78, 1.70, 1.50, 0.83, 1.48, 1.92])
SALVAGE = np.zeros(8)  # donation is not cash income
BATCH = np.array([10, 6, 4, 2, 2, 4, 4, 4])
WORK = np.array([1., 1.1, 1.7, 2.3, 2.2, 1.8, 2.0, 2.5])
CAPACITY = np.array([1120., 1030., 980.])
STORE_FACTOR = np.array([1.10, 0.92, 0.88])
CAT = ["store_code", "product_code", "weekday"]
FEATURES = CAT + ["doy_sin", "doy_cos", "trend", "holiday", "vacation", "promo", "price", "weather_forecast", "lead_days", "lag7", "lag14", "lag21", "lag28", "origin_mean7", "origin_mean28", "sameweekday_mean8", "sameweekday_std8", "recent_stockout_share"]
TRAIN_END = "2024-06-30"
VALID_END = "2024-12-31"
CALIB_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2026-06-30"
LAST_OBS = "2026-10-14"


def _dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """A deliberately simple public calendar, not a complete legal holiday list."""
    doy = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy()
    day = dates.day.to_numpy()
    holiday = ((month == 1) & (day == 1)) | ((month == 5) & (day == 1)) | ((month == 10) & (day == 3)) | ((month == 12) & np.isin(day, [25, 26]))
    # Fictional local vacation calendar, disclosed as such.
    vacation = ((month == 8) | ((month == 12) & (day >= 23)) | ((month == 4) & (day <= 12))).astype(int)
    return pd.DataFrame({"date": dates, "weekday": dates.dayofweek, "doy_sin": np.sin(2*np.pi*doy/365.25), "doy_cos": np.cos(2*np.pi*doy/365.25), "trend": (dates - pd.Timestamp("2023-01-01")).days / 365.25, "holiday": holiday.astype(int), "vacation": vacation})


def allocate_batches(qty: np.ndarray, product_codes: np.ndarray, unit_price: np.ndarray, capacity: float) -> np.ndarray:
    """Same transparent batch/capacity allocator for every policy; not a MILP optimum."""
    p = np.asarray(product_codes, dtype=int)
    q = np.maximum(0., np.asarray(qty, dtype=float))
    q = np.floor(q / BATCH[p] + .5) * BATCH[p]
    priority = (unit_price - COSTS[p]) / WORK[p]
    # Remove one whole batch at a time from the lowest contribution/work category.
    while np.sum(q * WORK[p]) > capacity + 1e-7:
        possible = np.flatnonzero(q >= BATCH[p])
        if not len(possible):
            break
        j = possible[np.argmin(priority[possible])]
        q[j] -= BATCH[p[j]]
    return q.astype(int)


def generate_data(seed:int=SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic data-generating process with POS censoring and archived weather vintages."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2026-10-21", freq="D")
    cal = calendar(dates)
    n = len(dates)
    seasonal_temp = 10.0 - 9.0*np.cos(2*np.pi*(dates.dayofyear.to_numpy()-12)/365.25)
    weather_anomaly = np.zeros(n)
    demand_common = np.zeros(n)
    for t in range(n):
        weather_anomaly[t] = .72*(weather_anomaly[t-1] if t else 0.) + rng.normal(0, 2.3)
        demand_common[t] = .34*(demand_common[t-1] if t else 0.) + rng.normal(0, .065)
    actual_temp = seasonal_temp + weather_anomaly
    vintage_rows = []
    vintage = np.zeros((n, 7))
    for t in range(n):
        for lead in range(1, 8):
            orig = t-lead
            previous_anomaly = weather_anomaly[orig] if orig >= 0 else 0.
            # Forecast generated from origin information, not future actual weather.
            f = seasonal_temp[t] + .72**lead*previous_anomaly + rng.normal(0, 1.1 + .22*lead)
            vintage[t, lead-1] = f
            if orig >= 0:
                vintage_rows.append((dates[orig], dates[t], lead, f))
    weekday_effect = np.array([.91, .91, .94, 1.00, 1.15, 1.27, .95])
    all_sales = np.zeros((n, 3, 8))
    all_stock = np.zeros_like(all_sales, dtype=bool)
    rows, hidden = [], []
    for t, date in enumerate(dates):
        wd = int(cal.iloc[t]["weekday"])
        hol = int(cal.iloc[t]["holiday"])
        vac = int(cal.iloc[t]["vacation"])
        # Promotions are announced at least 14 days ahead and therefore known.
        promo_product = ((date.isocalendar().week*3 + date.year) % 8)
        promo_on = (date.isocalendar().week % 5 == 0 and wd in [3, 4, 5])
        promo = (np.arange(8) == promo_product).astype(int)*int(promo_on)
        price = PRICES * np.where(promo, .92, 1.)
        for store in range(3):
            station_effect = np.ones(8)
            if store == 1:
                station_effect[6:] = .61 if wd >= 5 else 1.27
            if store == 2:
                station_effect[:2] = 1.16 if wd >= 5 else .97
            season = 1.0 + .075*np.cos(2*np.pi*(date.dayofyear-18)/365.25)
            years = (date - dates[0]).days/365.25
            drift = (1.0 + .025*years) * (1.045 if date >= pd.Timestamp("2025-10-01") and store == 2 else 1.)
            holiday_factor = .78 if hol else 1.
            vac_factor = (1-.13*vac) if store == 1 else (1-.025*vac)
            weather_response = np.array([-.003, -.003, -.007, -.001, -.001, -.006, .009, -.011])
            weather_true = np.maximum(.73, 1. + weather_response*(actual_temp[t]-10))
            promo_effect = 1. + promo*np.array([.20,.18,.26,.11,.12,.24,.21,.19])
            expected = BASE*STORE_FACTOR[store]*weekday_effect[wd]*station_effect*season*drift*holiday_factor*vac_factor*weather_true*promo_effect
            # Nonlinear, learnable interaction: snack/croissant at very cold commuter mornings.
            if store == 1 and wd < 5 and actual_temp[t] < 4:
                expected[[2,7]] *= 1.14
            expected *= np.exp(demand_common[t])
            extra = rng.lognormal(-.5*.085**2, .085, 8)
            demand = rng.poisson(np.maximum(expected*extra, .2))
            # A fictional existing planner uses known calendar, noisy forecasts and past POS sales.
            weather_known = np.maximum(.73, 1. + weather_response*(vintage[t,0]-10))
            planner_expected = BASE*STORE_FACTOR[store]*weekday_effect[wd]*station_effect*season*(1+.018*years)*holiday_factor*vac_factor*weather_known*(1+.15*promo)
            if t >= 28:
                historic = all_sales[[t-7,t-14,t-21,t-28], store, :]
                planner_expected = .76*planner_expected + .24*np.mean(historic, axis=0)
            # Some preventable judgment errors; independent of future realized latent demand.
            judgment = rng.normal(0, .055, 8)
            production_raw = planner_expected*(1.10+judgment)
            q = allocate_batches(production_raw, np.arange(8), price, CAPACITY[store])
            sales = np.minimum(demand, q)
            stockout = demand >= q
            leftovers = np.maximum(q-sales, 0)
            all_sales[t,store]=sales
            all_stock[t,store]=stockout
            for product in range(8):
                rows.append({"date":date,"store_code":store,"store":STORES[store],"product_code":product,"product":PRODUCTS[product],"sales_units":int(sales[product]),"production_units":int(q[product]),"leftover_units":int(leftovers[product]),"stockout_flag":bool(stockout[product]),"promo":int(promo[product]),"unit_price_eur":round(float(price[product]),4),"avoidable_unit_cost_eur":float(COSTS[product]),"salvage_unit_value_eur":float(SALVAGE[product]),"batch_units":int(BATCH[product]),"work_units_per_piece":float(WORK[product]),"store_capacity_work_units":float(CAPACITY[store])})
                hidden.append({"date":date,"store_code":store,"product_code":product,"latent_demand_units":int(demand[product]),"actual_weather_c":float(actual_temp[t]),"expected_demand_units":float(expected[product])})
    observed = pd.DataFrame(rows)
    audit = pd.DataFrame(hidden)
    weather = pd.DataFrame(vintage_rows, columns=["origin_date","target_date","lead_days","weather_forecast_c"])
    # Future POS outcomes never go into the public observed data table.
    observed = observed.loc[observed.date <= pd.Timestamp(LAST_OBS)].reset_index(drop=True)
    return observed, audit, weather


def feature_table(observed: pd.DataFrame, weather: pd.DataFrame, target_dates: pd.DatetimeIndex, fixed_origin=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct point-in-time features only from records at/before each origin."""
    records=[]
    cal = calendar(target_dates).set_index("date")
    wx = weather.set_index(["origin_date","target_date"])["weather_forecast_c"]
    groups = {(int(s),int(p)):g.sort_values("date").set_index("date") for (s,p),g in observed.groupby(["store_code","product_code"],sort=False)}
    for date in target_dates:
        origin = pd.Timestamp(fixed_origin) if fixed_origin is not None else date-pd.Timedelta(days=1)
        lead = int((date-origin).days)
        if lead < 1 or lead > 7:
            raise ValueError("This demo supports only 1–7 day point-in-time horizons")
        c=cal.loc[date]
        for store in range(3):
            for product in range(8):
                g=groups[(store,product)]
                prior = g.loc[(g.index <= origin) & (g.index >= origin-pd.Timedelta(days=27))]
                if len(prior)<28:
                    continue
                same_dates = [date-pd.Timedelta(days=7*i) for i in range(1,9)]
                assert max(same_dates) <= origin
                same = g.reindex(same_dates)
                if same.sales_units.isna().any():
                    continue
                usable = same.loc[~same.stockout_flag.astype(bool),"sales_units"]
                # No claim that missing demand has been recovered. Fallback is observed POS sales.
                if len(usable)<3:
                    usable=same.sales_units
                future_promo_product=((date.isocalendar().week*3+date.year)%8)
                promo=int(date.isocalendar().week%5==0 and date.dayofweek in [3,4,5] and product==future_promo_product)
                vals={"date":date,"origin_date":origin,"store_code":store,"product_code":product,"store":STORES[store],"product":PRODUCTS[product],"weekday":int(c.weekday),"doy_sin":float(c.doy_sin),"doy_cos":float(c.doy_cos),"trend":float(c.trend),"holiday":int(c.holiday),"vacation":int(c.vacation),"promo":promo,"price":float(PRICES[product]*(.92 if promo else 1)),"weather_forecast":float(wx.loc[(origin,date)]),"lead_days":lead,"lag7":float(same.iloc[0].sales_units),"lag14":float(same.iloc[1].sales_units),"lag21":float(same.iloc[2].sales_units),"lag28":float(same.iloc[3].sales_units),"origin_mean7":float(prior.iloc[-7:].sales_units.mean()),"origin_mean28":float(prior.sales_units.mean()),"sameweekday_mean8":float(usable.mean()),"sameweekday_std8":float(usable.std(ddof=0)),"recent_stockout_share":float(prior.stockout_flag.mean()),"latest_history_date":prior.index.max(),"latest_weekday_lag_date":max(same_dates)}
                if date in g.index:
                    current=g.loc[date]
                    vals.update({"sales_units":int(current.sales_units),"stockout_flag":bool(current.stockout_flag)})
                records.append(vals)
    frame=pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No rows with required history")
    assert (frame.latest_history_date <= frame.origin_date).all()
    assert (frame.latest_weekday_lag_date <= frame.origin_date).all()
    assert "latent_demand_units" not in frame.columns
    return frame[FEATURES].astype(float), frame


def metric_pack(y, point, low, high, censor=None) -> dict:
    y=np.asarray(y,float); point=np.asarray(point,float)
    low=np.asarray(low,float); high=np.asarray(high,float)
    mask=np.ones(len(y),bool) if censor is None else ~np.asarray(censor,bool)
    yy=y[mask]; pp=point[mask]
    def pin(q,p):
        e=yy-p[mask]
        return float(np.mean(np.maximum(q*e,(q-1)*e)))
    return {"n":int(mask.sum()),"mae":float(np.mean(np.abs(yy-pp))),"rmse":float(np.sqrt(np.mean((yy-pp)**2))),"wape":float(np.sum(np.abs(yy-pp))/np.maximum(np.sum(yy),1)),"bias_units":float(np.mean(pp-yy)),"relative_bias":float(np.sum(pp-yy)/np.maximum(np.sum(yy),1)),"pinball_p10":pin(.1,low),"pinball_p50":pin(.5,point),"pinball_p90":pin(.9,high),"interval_80_coverage":float(np.mean((yy>=low[mask])&(yy<=high[mask]))),"mean_interval_width":float(np.mean(high[mask]-low[mask]))}


def make_ridge(alpha:float):
    numeric=[c for c in FEATURES if c not in CAT]
    pre=ColumnTransformer([("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CAT),("numeric",StandardScaler(),numeric)])
    return make_pipeline(pre,Ridge(alpha=alpha))


def experience_predict(meta: pd.DataFrame, promo_ratio: np.ndarray) -> np.ndarray:
    # 8 same-weekday POS observations; promotions get an observed-data multiplier.
    codes=meta.product_code.to_numpy(int)
    return np.maximum(0,meta.sameweekday_mean8.to_numpy()*np.where(meta.promo.to_numpy()==1,promo_ratio[codes],1.))


def learn_promo_ratio(train:pd.DataFrame) -> np.ndarray:
    # Coarse observed-sales adjustment, not a causal elasticity estimate.
    ratios=[]
    for p in range(8):
        g=train.loc[(train.product_code==p)&(~train.stockout_flag)]
        rates=g.sales_units/np.maximum(g.sameweekday_mean8,1)
        a=rates[g.promo==1]; b=rates[g.promo==0]
        ratios.append(float(np.clip(a.median()/max(b.median(),.1),1.,1.4)) if len(a)>=15 else 1.)
    return np.array(ratios)


def calibrate(meta:pd.DataFrame, point:np.ndarray)->dict:
    """Observed-sales calibration only; stockout labels remain lower bounds."""
    output={}
    point=np.maximum(point,1.)
    for p in range(8):
        mask=(meta.product_code.to_numpy()==p)&(~meta.stockout_flag.to_numpy(bool))
        residual=(meta.sales_units.to_numpy()[mask]-point[mask])/np.sqrt(point[mask])
        output[str(p)]={"n":int(mask.sum()),"standardized_residuals":np.sort(residual).tolist(),"q10":float(np.quantile(residual,.1)),"q90":float(np.quantile(residual,.9))}
    return output


def quantile_predict(meta:pd.DataFrame, point:np.ndarray, calibration:dict, q:float|np.ndarray)->np.ndarray:
    point=np.maximum(point,0.)
    qs=np.full(len(point),q,float) if np.isscalar(q) else np.asarray(q,float)
    out=np.empty(len(point))
    for p in range(8):
        m=meta.product_code.to_numpy(int)==p
        residual=calibration[str(p)]["standardized_residuals"]
        quantiles=np.quantile(residual,qs[m])
        out[m]=point[m]+np.sqrt(np.maximum(point[m],1))*quantiles
    return np.maximum(out,0.)


def policy_quantities(meta:pd.DataFrame, point:np.ndarray, calibration:dict)->np.ndarray:
    p=meta.product_code.to_numpy(int)
    price=meta.price.to_numpy(float)
    under=np.maximum(price-COSTS[p],.01)
    over=np.maximum(COSTS[p]-SALVAGE[p],.01)
    critical=under/(under+over)
    desired=quantile_predict(meta,point,calibration,critical)
    result=np.zeros(len(meta),dtype=int)
    for (_date,store),g in meta.groupby(["date","store_code"],sort=False):
        ix=g.index.to_numpy(int)
        result[ix]=allocate_batches(desired[ix],p[ix],price[ix],CAPACITY[int(store)])
    return result


def economics(meta:pd.DataFrame, quantities:dict[str,np.ndarray], audit:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    m=meta[["date","store_code","product_code","price"]].merge(audit[["date","store_code","product_code","latent_demand_units"]],on=["date","store_code","product_code"],validate="one_to_one")
    demand=m.latent_demand_units.to_numpy(float)
    p=m.product_code.to_numpy(int)
    rows=m.copy()
    summaries={}
    for name,q in quantities.items():
        q=np.asarray(q,float)
        sold=np.minimum(q,demand)
        waste=np.maximum(q-demand,0.)
        shortage=np.maximum(demand-q,0.)
        contribution=np.maximum(m.price.to_numpy()-COSTS[p],0)
        waste_cost=waste*(COSTS[p]-SALVAGE[p])
        lost_contribution=shortage*contribution
        loss=waste_cost+lost_contribution
        realized_contribution=sold*contribution-waste_cost
        rows[f"{name}_quantity"]=q.astype(int)
        rows[f"{name}_waste_units"]=waste
        rows[f"{name}_shortage_units"]=shortage
        rows[f"{name}_waste_cost_eur"]=waste_cost
        rows[f"{name}_lost_contribution_eur"]=lost_contribution
        rows[f"{name}_avoidable_loss_eur"]=loss
        rows[f"{name}_realized_contribution_eur"]=realized_contribution
        summaries[name]={"avoidable_loss_eur":float(loss.sum()),"waste_cost_eur":float(waste_cost.sum()),"lost_contribution_eur":float(lost_contribution.sum()),"waste_units":int(waste.sum()),"shortage_units":int(shortage.sum()),"realized_contribution_eur":float(realized_contribution.sum()),"units_produced":int(q.sum())}
        # Algebra prevents double-counting an avoided shortage as both revenue and profit.
        assert np.allclose(demand*contribution-realized_contribution,loss)
    return rows,summaries


def paired_week_bootstrap(rows:pd.DataFrame, base:str, chosen:str,n_boot:int=2500)->dict:
    daily=rows.groupby("date")[[f"{base}_avoidable_loss_eur",f"{chosen}_avoidable_loss_eur"]].sum()
    delta=daily[f"{base}_avoidable_loss_eur"]-daily[f"{chosen}_avoidable_loss_eur"]
    # Contiguous seven-day blocks preserve shared shocks and weekday covariance.
    ids=np.arange(len(delta))//7
    blocks=pd.DataFrame({"block":ids,"delta":delta.to_numpy()}).groupby("block").delta.sum().to_numpy()
    rng=np.random.default_rng(SEED+117)
    draws=rng.choice(blocks,size=(n_boot,len(blocks)),replace=True).sum(axis=1)
    return {"paired_test_saving_eur":float(delta.sum()),"ci_95_eur":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],"blocks":int(len(blocks)),"bootstrap_draws":n_boot,"method":"paired contiguous 7-day block resampling; assumes blocks approximately representative; last block may be shorter; no market inference"}


def run_checks(observed:pd.DataFrame,audit:pd.DataFrame,weather:pd.DataFrame,features:pd.DataFrame,meta:pd.DataFrame,output:Path)->dict:
    assertions={}
    assert observed.date.max()==pd.Timestamp(LAST_OBS)
    assert len(observed)==len(observed.drop_duplicates(["date","store_code","product_code"]))
    assert (observed.sales_units+observed.leftover_units==observed.production_units).all()
    assert (observed.sales_units<=observed.production_units).all()
    assert not any("latent" in c or "actual_weather" in c for c in FEATURES)
    assert (meta.latest_history_date<=meta.origin_date).all()
    assert (meta.latest_weekday_lag_date<=meta.origin_date).all()
    assert (meta.date>meta.origin_date).all()
    assertions.update({"unique_observed_keys":True,"sales_production_balance":True,"future_pos_not_public":True,"all_feature_history_at_or_before_origin":True,"hidden_demand_excluded_from_feature_columns":True})
    sample_dates=pd.date_range("2025-10-02",periods=2)
    original_x,_=feature_table(observed,weather,sample_dates)
    poisoned=observed.copy()
    poisoned.loc[poisoned.date>=sample_dates.min(),"sales_units"]=999999
    modified_x,_=feature_table(poisoned,weather,pd.DatetimeIndex([sample_dates.min()]))
    pd.testing.assert_frame_equal(original_x.iloc[:24].reset_index(drop=True),modified_x.reset_index(drop=True))
    assertions["changing_target_and_future_sales_does_not_change_origin_features"]=True
    future_x,future_meta=feature_table(observed,weather,pd.date_range("2026-10-15",periods=7),fixed_origin=LAST_OBS)
    assert future_meta.latest_history_date.max()<=pd.Timestamp(LAST_OBS)
    assert future_meta.latest_weekday_lag_date.max()<=pd.Timestamp(LAST_OBS)
    assert len(future_meta)==7*3*8
    assertions["seven_day_forecast_uses_only_oct14_or_earlier_sales"]=True
    _dump(output/"verification.json",assertions)
    return assertions


def save_chart(fig,path:Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"),dpi=160,bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"),bbox_inches="tight")
    plt.close(fig)


def run(output_dir:Path,fast:bool=False)->dict:
    started=time.perf_counter()
    output_dir=Path(output_dir)
    for d in ["data","audit_private","model","charts"]:
        (output_dir/d).mkdir(parents=True,exist_ok=True)
    observed,audit,weather=generate_data()
    observed.to_csv(output_dir/"data/observed_sales.csv",index=False,date_format="%Y-%m-%d")
    weather.loc[weather.origin_date<=pd.Timestamp(LAST_OBS)].to_csv(output_dir/"data/weather_forecast_vintages.csv",index=False,date_format="%Y-%m-%d")
    audit.to_csv(output_dir/"audit_private/latent_demand.csv",index=False,date_format="%Y-%m-%d")
    x,meta=feature_table(observed,weather,pd.date_range("2023-03-01",LAST_OBS))
    checks=run_checks(observed,audit,weather,x,meta,output_dir)
    train=(meta.date<=TRAIN_END).to_numpy()
    valid=((meta.date>TRAIN_END)&(meta.date<=VALID_END)).to_numpy()
    calib=((meta.date>VALID_END)&(meta.date<=CALIB_END)).to_numpy()
    test=((meta.date>=TEST_START)&(meta.date<=TEST_END)).to_numpy()
    y=meta.sales_units.to_numpy(float)
    weights=np.where(meta.stockout_flag.to_numpy(bool),.20,1.)
    promo_ratio=learn_promo_ratio(meta.loc[train])
    candidate_rows=[]
    trained={}
    valid_preds={"seasonal_naive":meta.loc[valid,"lag7"].to_numpy(float),"weekday_experience":experience_predict(meta.loc[valid],promo_ratio)}
    valid_uncens=~meta.loc[valid,"stockout_flag"].to_numpy(bool)
    for name,p in valid_preds.items():
        candidate_rows.append({"model":name,"validation_mae_uncensored":float(np.mean(np.abs(y[valid][valid_uncens]-p[valid_uncens]))),"complexity":"transparent seasonal rule"})
    for alpha in ([100.] if fast else [10.,100.,500.]):
        name=f"ridge_alpha_{int(alpha)}"
        model=make_ridge(alpha)
        model.fit(x.loc[train],y[train],ridge__sample_weight=weights[train])
        p=np.maximum(model.predict(x.loc[valid]),0)
        trained[name]=model
        valid_preds[name]=p
        candidate_rows.append({"model":name,"validation_mae_uncensored":float(np.mean(np.abs(y[valid][valid_uncens]-p[valid_uncens]))),"complexity":"regularized linear regression with common known calendar/weather/history"})
    grid=[{"learning_rate":.07,"max_leaf_nodes":15,"max_iter":150,"l2_regularization":12.,"min_samples_leaf":70}]
    if not fast:
        grid.append({"learning_rate":.05,"max_leaf_nodes":23,"max_iter":230,"l2_regularization":25.,"min_samples_leaf":90})
    for i,params in enumerate(grid):
        name=f"hist_gradient_boosting_{i+1}"
        model=HistGradientBoostingRegressor(loss="squared_error",categorical_features=CAT,early_stopping=False,random_state=SEED,**params)
        model.fit(x.loc[train],y[train],sample_weight=weights[train])
        p=np.maximum(model.predict(x.loc[valid]),0)
        trained[name]=model
        valid_preds[name]=p
        candidate_rows.append({"model":name,"validation_mae_uncensored":float(np.mean(np.abs(y[valid][valid_uncens]-p[valid_uncens]))),"complexity":"regularized tabular gradient boosting"})
    candidate_df=pd.DataFrame(candidate_rows).sort_values("validation_mae_uncensored")
    simple_df=candidate_df.loc[~candidate_df.model.str.startswith("hist_gradient")]
    best_simple=str(simple_df.iloc[0].model)
    best_all=str(candidate_df.iloc[0].model)
    simple_mae=float(simple_df.iloc[0].validation_mae_uncensored)
    all_mae=float(candidate_df.iloc[0].validation_mae_uncensored)
    # Chosen once on validation. Less than 1% improvement does not buy complexity.
    chosen=best_all if (not best_all.startswith("hist_gradient") or all_mae<.99*simple_mae) else best_simple
    candidate_df["selected"]=candidate_df.model.eq(chosen)
    candidate_df.to_csv(output_dir/"model/validation_model_selection.csv",index=False)
    chosen_model=trained.get(chosen)
    # Refit chosen and a frozen best ridge after selection; the test remains untouched.
    refit_mask=train|valid
    best_ridge=str(candidate_df.loc[candidate_df.model.str.startswith("ridge")].iloc[0].model)
    fitted={}
    for name in set([chosen,best_ridge]):
        if name not in trained:
            continue
        model=trained[name]
        if name.startswith("ridge"):
            model.fit(x.loc[refit_mask],y[refit_mask],ridge__sample_weight=weights[refit_mask])
        else:
            model.fit(x.loc[refit_mask],y[refit_mask],sample_weight=weights[refit_mask])
        fitted[name]=model
        joblib.dump(model,output_dir/f"model/{name}_test_model.joblib")
    cal_meta=meta.loc[calib].reset_index(drop=True)
    test_meta=meta.loc[test].reset_index(drop=True)
    predictions={}
    calibrations={}
    for name in ["seasonal_naive","weekday_experience","ridge", "selected"]:
        if name=="seasonal_naive":
            cp=cal_meta.lag7.to_numpy(); tp=test_meta.lag7.to_numpy()
        elif name=="weekday_experience":
            cp=experience_predict(cal_meta,promo_ratio); tp=experience_predict(test_meta,promo_ratio)
        else:
            source_name=best_ridge if name=="ridge" else chosen
            if source_name=="seasonal_naive":
                cp=cal_meta.lag7.to_numpy();tp=test_meta.lag7.to_numpy()
            elif source_name=="weekday_experience":
                cp=experience_predict(cal_meta,promo_ratio);tp=experience_predict(test_meta,promo_ratio)
            else:
                cp=np.maximum(fitted[source_name].predict(x.loc[calib]),0)
                tp=np.maximum(fitted[source_name].predict(x.loc[test]),0)
        calibrations[name]=calibrate(cal_meta,np.asarray(cp))
        predictions[name]=np.asarray(tp,float)
    _dump(output_dir/"model/calibration.json",calibrations)
    metrics={"task":"next-day observed-sales forecast; demand interpretation is limited by stockouts","primary_evaluation":"non-stockout rows of untouched chronological test; selection effects remain","test_start":TEST_START,"test_end":TEST_END,"selected_model":chosen,"best_simple":best_simple,"frozen_best_ridge":best_ridge,"selection_rule":"choose complex model only if validation MAE on non-stockout rows improves at least 1% over best simpler candidate","test_stockout_fraction":float(test_meta.stockout_flag.mean()),"models":{}}
    pred_file=test_meta[["date","origin_date","store","product","store_code","product_code","sales_units","stockout_flag","price"]].copy()
    for name,point in predictions.items():
        low=quantile_predict(test_meta,point,calibrations[name],.1)
        high=quantile_predict(test_meta,point,calibrations[name],.9)
        metrics["models"][name]={"uncensored_observed_sales":metric_pack(test_meta.sales_units,point,low,high,test_meta.stockout_flag),"all_observed_sales_censor_biased":metric_pack(test_meta.sales_units,point,low,high)}
        pred_file[f"{name}_point"]=point
        pred_file[f"{name}_p10"]=low
        pred_file[f"{name}_p90"]=high
    # Hidden oracle errors are audit-only, computed after choices have been frozen.
    latent_test=test_meta[["date","store_code","product_code"]].merge(audit,on=["date","store_code","product_code"],validate="one_to_one").latent_demand_units
    for name,point in predictions.items():
        low=quantile_predict(test_meta,point,calibrations[name],.1);high=quantile_predict(test_meta,point,calibrations[name],.9)
        metrics["models"][name]["synthetic_latent_demand_oracle_only"]=metric_pack(latent_test,point,low,high)
    _dump(output_dir/"metrics.json",metrics)
    pred_file.to_csv(output_dir/"data/test_predictions.csv",index=False,date_format="%Y-%m-%d")
    segment_rows=[]
    for by,groups in [("store",test_meta.groupby("store")),("product",test_meta.groupby("product")),("weekday",test_meta.groupby("weekday"))]:
        for label,g in groups:
            idx=g.index.to_numpy(int)
            for name,point in predictions.items():
                pack=metric_pack(g.sales_units,point[idx],quantile_predict(g.reset_index(drop=True),point[idx],calibrations[name],.1),quantile_predict(g.reset_index(drop=True),point[idx],calibrations[name],.9),g.stockout_flag)
                segment_rows.append({"dimension":by,"segment":str(label),"model":name,**pack})
    pd.DataFrame(segment_rows).to_csv(output_dir/"data/segment_metrics.csv",index=False)
    day_errors=pred_file.copy()
    for name in predictions:
        day_errors[f"{name}_absolute_error"]=(day_errors[f"{name}_point"]-day_errors.sales_units).abs()
    day_errors.groupby("date")[[f"{n}_absolute_error" for n in predictions]].sum().to_csv(output_dir/"data/day_forecast_errors.csv",date_format="%Y-%m-%d")
    quantities={name:policy_quantities(test_meta,point,calibrations[name]) for name,point in predictions.items()}
    # Every alternative gets exactly the same feasibility constraints.
    for name,q in quantities.items():
        p=test_meta.product_code.to_numpy(int)
        assert (q % BATCH[p] == 0).all()
        constraint_frame=test_meta[["date","store_code"]].copy()
        constraint_frame["work"]=q*WORK[p]
        for (_date,store),g in constraint_frame.groupby(["date","store_code"],sort=False):
            assert g.work.sum()<=CAPACITY[int(store)]+1e-7
    checks["all_policies_respect_identical_batch_and_daily_capacity_constraints"]=True
    _dump(output_dir/"verification.json",checks)
    econ_rows,summary=economics(test_meta,quantities,audit)
    econ_rows.to_csv(output_dir/"audit_private/test_economic_replay.csv",index=False,date_format="%Y-%m-%d")
    daily=econ_rows.groupby("date")[[c for c in econ_rows if c.endswith("avoidable_loss_eur") or c.endswith("waste_cost_eur") or c.endswith("lost_contribution_eur")]].sum()
    daily.to_csv(output_dir/"data/day_economic_results.csv",date_format="%Y-%m-%d")
    savings={name:paired_week_bootstrap(econ_rows,name,"selected",500 if fast else 2500) for name in ["seasonal_naive","weekday_experience","ridge"]}
    # Strong deployment comparator fixed from validation, not chosen on favorable test economics.
    comparator="ridge" if best_simple.startswith("ridge") else best_simple
    gross=savings[comparator]["paired_test_saving_eur"]
    sens=[]
    for adoption in [1.,.75,.5,.25]:
        for monthly_fee in [0,99,199,349]:
            annual_fee=monthly_fee*12
            review_cost=52*.5*22  # explicit planning assumption, not measured
            net=gross*adoption-annual_fee-review_cost
            sens.append({"adoption_fraction":adoption,"monthly_software_or_service_eur":monthly_fee,"annual_additional_review_cost_eur":review_cost,"annualized_test_gross_saving_eur":gross*adoption,"annualized_test_net_saving_eur":net,"note":"scales a full-year synthetic test; implementation fee and profit taxes omitted; no real-world ROI proof"})
    pd.DataFrame(sens).to_csv(output_dir/"data/cost_adoption_sensitivity.csv",index=False)
    stress=[]
    # Frozen quantities under alternative costs and a +15% quantity perturbation.
    # Its direction is not pre-labelled adverse: censoring can make it help.
    for waste_multiplier in [.5,1.,1.5]:
        for lost_margin_multiplier in [.5,1.,1.5]:
            base_loss=econ_rows[f"{comparator}_waste_cost_eur"].sum()*waste_multiplier+econ_rows[f"{comparator}_lost_contribution_eur"].sum()*lost_margin_multiplier
            model_loss=econ_rows["selected_waste_cost_eur"].sum()*waste_multiplier+econ_rows["selected_lost_contribution_eur"].sum()*lost_margin_multiplier
            stress.append({"waste_cost_multiplier":waste_multiplier,"lost_margin_multiplier":lost_margin_multiplier,"gross_saving_eur":float(base_loss-model_loss),"note":"cost-weight stress at frozen quantities, no retuning"})
    overorder=quantities["selected"]*1.15
    stressed_q=np.zeros(len(test_meta),int)
    for (_date,store),g in test_meta.groupby(["date","store_code"],sort=False):
        idx=g.index.to_numpy(int)
        stressed_q[idx]=allocate_batches(overorder[idx],g.product_code.to_numpy(),g.price.to_numpy(),CAPACITY[int(store)])
    _,stressed_summary=economics(test_meta,{"bad_overorder":stressed_q},audit)
    negative={"description":"Separate unfavorable cost/adoption scenario: 25% of the ORIGINAL model's simulated gross difference, 349 EUR/month service and 30 minutes weekly review. This net value is not combined with the separate quantity stress.","quantity_stress_description":"Separate +15% desired-quantity perturbation, constrained to the same batches/capacity. It may improve or worsen loss; this audit is not a retuned production policy.","inflated_quantity_avoidable_loss_eur":stressed_summary["bad_overorder"]["avoidable_loss_eur"],"inflated_quantity_saving_vs_comparator_eur":summary[comparator]["avoidable_loss_eur"]-stressed_summary["bad_overorder"]["avoidable_loss_eur"],"25pct_adoption_net_saving_eur":gross*.25-349*12-52*.5*22,"not_test_tuned":True}
    econ={"evaluation_type":"fixed-information offline replay, not closed-loop deployment","demand_data":"synthetic latent audit outcome, never model label/feature/selection/calibration input","same_information":"each policy uses the same historical reference-policy sales and archived origin weather forecast; endogenous feedback from a deployed policy is not validated","daily_origin":"previous day, lead=1","test_days":int(test_meta.date.nunique()),"policies":summary,"saving_vs_baselines":savings,"strong_comparator_frozen_on_validation":comparator,"cost_adoption_sensitivity_file":"data/cost_adoption_sensitivity.csv","frozen_quantity_cost_stress":stress,"negative_case":negative,"economic_identity":"avoidable loss = surplus*(avoidable cost-salvage) + shortage*(price-avoidable cost); realized contribution = sold*(price-avoidable cost)-surplus*(avoidable cost-salvage)","capacity_and_batches":"identical product batches and store weighted-work capacity for all policies; simple common greedy allocator, not proven globally optimal","excluded":"existing fixed rent/labor, tax, implementation/integration fee, labor overtime, customer substitution, donated-goods noncash value, effects of early stockouts on future observed sales"}
    _dump(output_dir/"economic_results.json",econ)
    # Future model is refit separately after the frozen test is fully reported.
    future_train=(meta.date<=pd.Timestamp("2026-08-31")).to_numpy()
    future_cal=((meta.date>pd.Timestamp("2026-08-31"))&(meta.date<=LAST_OBS)).to_numpy()
    future_promo=learn_promo_ratio(meta.loc[future_train])
    future_model=None
    if chosen in trained:
        future_model=joblib.load(output_dir/f"model/{chosen}_test_model.joblib")
        if chosen.startswith("ridge"):
            future_model.fit(x.loc[future_train],y[future_train],ridge__sample_weight=weights[future_train])
        else:
            future_model.fit(x.loc[future_train],y[future_train],sample_weight=weights[future_train])
        fc_point=np.maximum(future_model.predict(x.loc[future_cal]),0)
    else:
        fc_point=meta.loc[future_cal,"lag7"].to_numpy() if chosen=="seasonal_naive" else experience_predict(meta.loc[future_cal],future_promo)
    future_calibration=calibrate(meta.loc[future_cal].reset_index(drop=True),fc_point)
    # Final future parameters use all observed outcomes available after Oct14 close.
    # Residuals above remain honest out-of-sample errors of the Aug31 parameter fit.
    # Reusing that calibration after final refit is an explicit transfer approximation.
    future_fit_mask=(meta.date<=LAST_OBS).to_numpy()
    if future_model is not None:
        if chosen.startswith("ridge"):
            future_model.fit(x.loc[future_fit_mask],y[future_fit_mask],ridge__sample_weight=weights[future_fit_mask])
        else:
            future_model.fit(x.loc[future_fit_mask],y[future_fit_mask],sample_weight=weights[future_fit_mask])
    else:
        future_promo=learn_promo_ratio(meta.loc[future_fit_mask])
    future_x,future_meta=feature_table(observed,weather,pd.date_range("2026-10-15",periods=7),fixed_origin=LAST_OBS)
    future_meta=future_meta.reset_index(drop=True)
    if future_model is not None:
        fp=np.maximum(future_model.predict(future_x),0)
    else:
        fp=future_meta.lag7.to_numpy() if chosen=="seasonal_naive" else experience_predict(future_meta,future_promo)
    future_out=future_meta[["date","origin_date","store","product","store_code","product_code","lead_days","price","promo","weather_forecast","latest_history_date","latest_weekday_lag_date"]].copy()
    future_out["forecast_units"]=fp
    future_out["p10_units"]=quantile_predict(future_meta,fp,future_calibration,.1)
    future_out["p90_units"]=quantile_predict(future_meta,fp,future_calibration,.9)
    future_out["recommended_production_units"]=policy_quantities(future_meta,fp,future_calibration)
    future_out["status"]=np.where(future_out.lead_days==1,"next-day quantity draft; no release justified by this negative economic test","7-day fixed-origin planning scenario; h>1 not validated as daily production policy")
    future_out.to_csv(output_dir/"data/future_forecast_2026-10-15_to_21.csv",index=False,date_format="%Y-%m-%d")
    future_out.loc[future_out.lead_days==1].to_csv(output_dir/"data/next_day_recommendations_2026-10-15.csv",index=False,date_format="%Y-%m-%d")
    if future_model is not None:
        joblib.dump(future_model,output_dir/"model/future_model.joblib")
    _dump(output_dir/"model/future_calibration.json",future_calibration)
    # Public charts: honest errors and operating tradeoffs, not only attractive totals.
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":13,"axes.labelsize":10})
    fig,ax=plt.subplots(figsize=(10,4.4))
    month=pd.to_datetime(pred_file.date).dt.to_period("M").astype(str)
    for name,color in [("weekday_experience","#94A3B8"),("ridge","#F59E0B"),("selected","#0F766E")]:
        e=(pred_file[f"{name}_point"]-pred_file.sales_units).abs()
        mask=~pred_file.stockout_flag
        monthly=pd.DataFrame({"month":month[mask],"error":e[mask]}).groupby("month").error.mean()
        ax.plot(np.arange(len(monthly)),monthly.values,label=name,color=color,marker="o",lw=2)
    ax.set_xticks(np.arange(len(monthly)),monthly.index,rotation=45,ha="right")
    ax.set_ylabel("MAE / Artikel–Filiale–Tag [Stück]")
    ax.set_title("Unberührter Jahrestest: echte Fehler auf nicht ausverkauften Tagen")
    ax.legend(ncol=3,frameon=False)
    save_chart(fig,output_dir/"charts/test_monthly_errors")
    ex=pred_file.loc[(pred_file.store_code==1)&(pred_file.product_code==0)&(pred_file.date>=pd.Timestamp("2026-05-01"))&(pred_file.date<=pd.Timestamp("2026-06-11"))]
    fig,ax=plt.subplots(figsize=(10,4.4))
    ax.fill_between(ex.date,ex.selected_p10,ex.selected_p90,color="#99F6E4",alpha=.6,label="Kalibriertes 80%-Intervall (Verkäufe)")
    ax.plot(ex.date,ex.sales_units,color="#334155",lw=1.3,label="Beobachtete Verkäufe")
    ax.plot(ex.date,ex.selected_point,color="#0F766E",lw=1.5,label="Ausgewählte Prognose")
    stock=ex.loc[ex.stockout_flag]
    ax.scatter(stock.date,stock.sales_units,marker="x",color="#DC2626",label="Ausverkauf: Nachfrage nicht vollständig beobachtet")
    ax.set_title("Bahnhof / Kaisersemmel: Prognose, Unsicherheit und Zensierung")
    ax.set_ylabel("Stück am nächsten Tag")
    ax.legend(fontsize=8,loc="upper left")
    save_chart(fig,output_dir/"charts/forecast_vs_observed")
    fig,ax=plt.subplots(figsize=(9,4.4))
    names=["seasonal_naive","weekday_experience","ridge","selected"]
    waste=np.array([summary[n]["waste_cost_eur"] for n in names]);short=np.array([summary[n]["lost_contribution_eur"] for n in names])
    ax.bar(names,waste/1000,label="Überschuss: vermeidbare Kosten",color="#F59E0B")
    ax.bar(names,short/1000,bottom=waste/1000,label="Fehlmenge: verlorener Beitrag",color="#0F766E")
    ax.set_ylabel("Vermeidbarer Verlust im synthetischen Jahr [Tsd. EUR]")
    ax.set_title("Gleiche latente Nachfrage, gleiche Chargen und Kapazität")
    ax.legend(fontsize=9,frameon=False)
    save_chart(fig,output_dir/"charts/economic_policy_comparison")
    fig,ax=plt.subplots(figsize=(9,4))
    agg=future_out.groupby("date")[["forecast_units","recommended_production_units"]].sum()
    ax.plot(agg.index,agg.forecast_units,color="#0F766E",marker="o",label="Erwartete Verkäufe")
    ax.plot(agg.index,agg.recommended_production_units,color="#F59E0B",marker="s",label="Chargen-/kapazitätsbegrenzter Mengenvorschlag")
    ax.axvspan(agg.index[1],agg.index[-1],color="#CBD5E1",alpha=.23,label="h > 1: Szenario, separat nicht validiert")
    ax.set_title("Festes Informationsdatum 14.10.2026: nächste Woche ohne Zukunftsverkäufe")
    ax.set_ylabel("Stück, alle Filialen/Artikel")
    ax.legend(fontsize=8,frameon=False)
    save_chart(fig,output_dir/"charts/future_forecast")
    config={"seed":SEED,"fictional_company":"IlmBrot – synthetischer Demonstrationsbetrieb (keine echte Firma)","stores":STORES,"products":PRODUCTS,"features":FEATURES,"target":"observed next-day POS sales; stockout rows are censored lower bounds, weight=0.20; not recovered true demand","data_period":["2023-01-01",LAST_OBS],"train_end":TRAIN_END,"validation_end":VALID_END,"calibration_end":CALIB_END,"untouched_test":[TEST_START,TEST_END],"future_origin":LAST_OBS,"daily_origin_cutoff":"after closing and finalized POS data, illustratively 20:00 previous day; not a 14:00 intraday forecast","future_model_fit_end":LAST_OBS,"future_residual_calibration":["2026-09-01",LAST_OBS],"future_note":"Residual calibration uses out-of-sample September–Oct14 predictions from an Aug31 fit; final model is then refit through Oct14. Transferring those empirical residuals after refit is approximate; future interval coverage is not known. Both are separate from the frozen test model.","horizon":"lead=1 for training/main validation/economic replay; lead=1..7 fixed-origin future scenarios","censored_training_weight":.2,"interval":"product-wise standardized residual empirical q10/q90 on non-stockout calibration rows; 80% nominal, no universal/exchangeability guarantee","parameters_candidates":grid,"salvage":SALVAGE.tolist(),"capacity_work_units":CAPACITY.tolist(),"batch_units":BATCH.tolist(),"costs_eur":COSTS.tolist(),"price_eur":PRICES.tolist(),"run_fast":fast}
    _dump(output_dir/"config.json",config)
    dictionary={"observed_sales.csv":{"date":"sale/production date, daily local units","sales_units":"POS units, min(latent demand, available production); NOT full demand on stockout days","production_units":"synthetic reference-policy morning units","leftover_units":"production minus sales; includes unsold donation/no cash salvage","stockout_flag":"availability exhaustion indicator, not amount of lost demand","promo":"announced promotion known 14 days ahead","unit_price_eur":"known selling price including announced promotion","avoidable_unit_cost_eur":"avoidable ingredients/energy/packaging cost per additionally produced unit; fixed rent/team salary excluded","batch_units":"integer feasible production increments","store_capacity_work_units":"common daily weighted-work limit"},"weather_forecast_vintages.csv":{"origin_date":"forecast information cutoff","target_date":"forecast weather target day","lead_days":"1..7","weather_forecast_c":"synthetic archived forecast generated from origin's weather state, not future realized weather"},"audit_private/latent_demand.csv":{"latent_demand_units":"hidden simulation oracle only; forbidden as model label/feature/tuning/calibration input","actual_weather_c":"hidden realized weather, forbidden as future feature","expected_demand_units":"hidden generating mean, forbidden as input"},"test_predictions.csv":"one previous-day forecast per store/product/test date; intervals, observed sales and stockout flag","next_day_recommendations_2026-10-15.csv":"24 concrete batch/capacity-constrained production suggestions at Oct14 origin","future_forecast_2026-10-15_to_21.csv":"168 fixed-origin forecast rows; h>1 are planning scenarios, not validated operational policies","segment_metrics.csv":"uncensored POS error/interval metrics by store/product/weekday","day_economic_results.csv":"synthetic offline loss totals by day/policy","cost_adoption_sensitivity.csv":"explicit service/adoption/review-cost assumptions applied to one full test year"}
    _dump(output_dir/"data_dictionary.json",dictionary)
    summary_out={"case":"bakery","company":"IlmBrot (vollständig synthetisch)","selected_model":chosen,"test_days":int(test_meta.date.nunique()),"test_rows":len(test_meta),"observed_rows":len(observed),"test_stockout_fraction":metrics["test_stockout_fraction"],"test_mae":metrics["models"]["selected"]["uncensored_observed_sales"]["mae"],"test_wape":metrics["models"]["selected"]["uncensored_observed_sales"]["wape"],"interval_coverage":metrics["models"]["selected"]["uncensored_observed_sales"]["interval_80_coverage"],"baseline_name":comparator,"baseline_mae":metrics["models"][comparator]["uncensored_observed_sales"]["mae"],"gross_simulated_year_saving_eur":gross,"gross_saving_ci_95_eur":savings[comparator]["ci_95_eur"],"negative_case_net_eur":negative["25pct_adoption_net_saving_eur"],"next_day_recommendation_units":int(future_out.loc[future_out.lead_days==1,"recommended_production_units"].sum()),"runtime_seconds":round(time.perf_counter()-started,2),"leakage_checks":checks,"charts":["charts/test_monthly_errors.png","charts/forecast_vs_observed.png","charts/economic_policy_comparison.png","charts/future_forecast.png"],"forecast_file":"data/future_forecast_2026-10-15_to_21.csv","next_day_file":"data/next_day_recommendations_2026-10-15.csv","simulation_only":True}
    _dump(output_dir/"summary.json",summary_out)
    _write_methods(output_dir,config,metrics,econ,summary_out,candidate_df)
    return summary_out


def _write_methods(output:Path,config:dict,metrics:dict,econ:dict,summary:dict,candidates:pd.DataFrame):
    body=f'''# Bäckereidemo: Nachfrageprognose und Produktionsentscheidung

IlmBrot ist eine erfundene Bäckerei mit drei Filialen und acht Produkten. Alle Daten, Preise, Kapazitäten und Ergebnisse sind synthetisch. Die Demo trainiert echte Regressionsmodelle; sie beweist keinen Kundennutzen und verwendet keine Daten einer realen regionalen Firma.

## Entscheidung und Modellziel

Die reale Entscheidung wäre: Welche Menge je Produkt/Filiale produzieren wir morgen? Informationsstichtag ist jeweils der Vortag **nach Ladenschluss und Abschluss der Kassendaten**, beispielsweise um 20 Uhr. Eine Vorhersage um 14 Uhr dürfte den vollständigen Tagesabsatz nicht kennen. Die wirtschaftliche Auswertung nutzt genau diesen Ein-Tages-Horizont. Die Mengenentwürfe für den 15.10.2026 stehen in `data/next_day_recommendations_2026-10-15.csv`; der negative wirtschaftliche Test begründet keine reale Freigabe des ausgewählten Boosting-Modells. Der zusätzliche Sieben-Tage-Ausblick hält sämtliche Informationen auf dem 14.10.2026 fest. Spätere Ist-Verkäufe werden nicht eingeschleust; längere Horizonte sind Planungsszenarien und haben keine separat behauptete Betriebsvalidierung.

Das trainierte Label ist **beobachteter POS-Verkauf**, nicht geheime vollständige Nachfrage. Bei Ausverkauf ist Verkauf eine Untergrenze. Diese Zeilen werden mit 0,20 statt 1,00 gewichtet; für residualbasierte Kalibrierung und die Hauptfehlerauswertung verwenden wir nicht ausverkaufte Zeilen. Das kann Selektionsverzerrung erzeugen. Wir behaupten keine Wiederherstellung verlorener Nachfrage. Im Jahrestest sind {metrics['test_stockout_fraction']:.1%} der Zeilen zensiert. Die nachträglichen Audit-Metriken gegen synthetische latente Nachfrage machen diese Einschränkung sichtbar.

## Wie die Daten entstehen

Die Generierung enthält Filial-/Produktunterschiede, Wochentag, Jahressaison, einen offengelegten vereinfachten Feiertags-/Ferienkalender, angekündigte Aktionen, Wetterabhängigkeit, gemeinsame korrelierte Nachfrageschocks, individuelle Mengenstreuung, leichte zeitliche Drift und eine nichtlineare Pendler-/Kaltwetterinteraktion. Wetterprognosen werden aus dem am Stichtag bekannten Wetterzustand plus Fehler erzeugt. Tatsächliches späteres Wetter darf nicht Modellinput sein. Eine historische Referenzplanung produziert in festen Chargen unter Tageskapazität. Verkauf ist das Minimum aus Nachfrage und Produktion; Restmengen und Ausverkauf werden erfasst. Die Erzeugungsparameter sind keine vom Modell gesehenen Nachfrageinformationen.

## Ehrliche zeitliche Trennung

- Training: März 2023 bis 30.06.2024; davor nur notwendige Lag-Historie.
- Auswahl: 01.07.–31.12.2024.
- Residualkalibrierung: 01.01.–30.06.2025.
- Unberührter Jahrestest: 01.07.2025–30.06.2026, 365 Tage.
- Für künftige Vorschläge zunächst separat bis 31.08.2026 fitten und September bis 14.10.2026 außerhalb dieses Fits vorhersagen. Diese echten Out-of-sample-Residuen bilden die jüngste Kalibrierung. Danach wird das Zukunftsmodell mit allen bis Ladenschluss 14.10.2026 bekannten Verkaufslabels neu fitten. Die Übertragung der vorher gemessenen Residualverteilung auf diesen letzten Refit ist eine offengelegte Näherung; die zukünftige Intervallabdeckung ist unbekannt.

Während des Tests sind frühere Test-Verkäufe als Lagwerte zulässig: Eine echte Vortagsentscheidung hätte diese Daten ebenfalls. Parameter, Modellwahl und Residualkalibrierung werden während des Tests nicht auf diesen Ergebnissen angepasst. Zufälliges Vermischen vergangener und zukünftiger Zeilen findet nicht statt.

## Baselines und trainiertes Modell

Verglichen werden derselbe Wochentag der Vorwoche, eine Acht-Wochen-Erfahrungsregel mit beobachteter Aktionskorrektur, Ridge-Regression und zwei regularisierte HistGradientBoosting-Kandidaten. Ridge und Boosting bekommen dieselben bekannten Kalender-, Preis-, Wetter- und Vergangenheitsinformationen. Aktionskorrektur ist deskriptiv, keine kausale Preiselastizität. Auswahlkriterium ist MAE auf nicht ausverkauften Validierungstagen; ein komplexeres Modell wird nur gewählt, wenn es mindestens 1 % besser als der beste einfachere Kandidat ist.

Ausgewählt: **{summary['selected_model']}**. Beste einfachere Vergleichsmethode: **{metrics['best_simple']}**. Auswahlentscheidungen und gemessene Validierungsfehler stehen in `model/validation_model_selection.csv`. Auch eine einfache Methode darf gewinnen. Das Jahrestestergebnis war kein Tuningziel.

## Unsicherheit und Prognosefehler

Pro Produkt kalibrieren wir standardisierte Residuen mit der Quadratwurzel des prognostizierten Mengenlevels. Die empirischen 10-/90-%-Quantile bilden ein nominelles 80-%-Intervall. Diese Intervalle sind aus beobachteten, nicht zensierten Verkäufen kalibriert; zeitliche Drift, Selektion und unterschiedliche Prognosehorizonte verhindern eine universelle Deckungsgarantie. Die tatsächliche Jahrestestdeckung beträgt {summary['interval_coverage']:.1%}. Der Test-MAE beträgt {summary['test_mae']:.2f} Stück je Artikel/Filiale/Tag; WAPE {summary['test_wape']:.1%}. MAE, RMSE, WAPE, Bias, Pinball und Intervallabdeckung liegen für alle Modelle und zusätzlich als Audit gegen latente Nachfrage vor.

## Aus Prognosen werden Mengenentscheidungen

Eine höhere Menge kann Ausverkauf vermeiden, aber Reste verursachen. Der kostenbezogene Quantilbedarf lautet: `q = verlorener Stückbeitrag / (verlorener Stückbeitrag + vermeidbarer Überschuss-Stückkosten)`. Dafür verwenden wir Verkaufspreis minus vermeidbare Herstellungskosten für Fehlmengen und Herstellungskosten minus tatsächlich monetarisierbaren Restwert für Überschuss. Restwert ist hier null; eine Spende ist kein Erlös. Danach runden wir auf dieselben Produktchargen und begrenzen auf dieselbe gewichtete Tageskapazität. Ein transparenter gemeinsamer Greedy-Allocator ist keine nachgewiesen global optimale Produktionsplanung.

## Was die Wirtschaftsauswertung beweist und was nicht

Das ist ein **fixed-information offline replay**. Jede Politik sieht dieselben Verkäufe der historischen Referenzpolitik. Unterschiedliche Produktionsvorschläge werden anschließend auf derselben latenten synthetischen Nachfragetrajektorie abgerechnet. Anders produzierte Mengen würden echte Folgeverkäufe und spätere Lagwerte verändern; dieses endogene Feedback ist hier nicht geschlossen simuliert. Für einen Kundennachweis wäre ein kontrollierter realer Einsatz nötig.

`Vermeidbarer Verlust = Überschuss × (Kosten − Restwert) + Fehlmenge × (Preis − Kosten)`.

Die erzielten Stückbeiträge und Überschusskosten erfüllen diese Identität in automatischen Prüfungen. Umsatz einer vermiedenen Fehlmenge wird nicht zusätzlich als Gewinn gezählt. Bereits bestehende Miete/Gehälter, Steuern, Integration, Umsetzungshonorar, Überstunden, Kundensubstitution, Retourenlogistik und Qualitätseffekte sind nicht enthalten.

Gegen die vor Testauswertung festgelegte starke Methode **{summary['baseline_name']}** ergibt das synthetische Testjahr {summary['gross_simulated_year_saving_eur']:,.2f} EUR Brutto-Unterschied; gepaartes Sieben-Tage-Blockbootstrap: {summary['gross_saving_ci_95_eur'][0]:,.2f} bis {summary['gross_saving_ci_95_eur'][1]:,.2f} EUR. Das ist eine Unsicherheitsbeschreibung dieser synthetischen Trajektorie, keine Markt-/Kundenkonfidenz. `economic_results.json` enthält getrennte Überschuss-/Fehlmengenwerte und alle Vergleichspolitiken. Kleine Verbesserungen gegenüber der starken Methode können die Entwicklungskosten nicht rechtfertigen.

## Sensitivität und negativer Fall

`data/cost_adoption_sensitivity.csv` zeigt 25–100 % Umsetzung, 0/99/199/349 EUR monatliche Betreuung und ausdrücklich angenommene 30 Minuten zusätzliche Prüfung pro Woche zu 22 EUR/Stunde. Einmalige Einführung fehlt und müsste zusätzlich finanziert werden. Der ungünstige Kosten-/Umsetzungsfall mit 25 % Umsetzung der ursprünglichen Bruttodifferenz und 349 EUR monatlicher Betreuung ergibt nach Prüfungskosten {summary['negative_case_net_eur']:,.2f} EUR im synthetischen Jahr. Eine separate +15-%-Mengenstörung sowie Kosten-/Margenstress werden gezeigt. Die Mengenstörung darf helfen oder schaden und wird nicht künstlich als negativer Fall bezeichnet; hier kann höhere Produktion die durch Zensierung unterschätzte Nachfrage besser treffen. Ihre Auswertung ist kein neu validiertes oder für bessere Testgewinne nachgestimmtes Modell. Der genannte Nettoverlust verwendet ausschließlich die ursprüngliche Politik, nicht die Mengenstörung.

## Für den ersten echten Bäckereikunden nötig

1. POS-Zeilen mindestens nach Datum, Uhrzeit, Artikel, Filiale, verkaufter Menge, Preis/Rabatt, Storno und stabiler Artikelkennung.
2. Produktions-/Liefermengen, Retouren/Restmengen, Bestände bzw. Ausverkaufszeit und Öffnungskalender.
3. Zum Stichtag bekannte Vorbestellungen, Aktionen, geplante Produktwechsel und tatsächlich archivierte Wetterprognosen, falls Wetter verwendet werden soll.
4. Real vermeidbare Stückkosten, Losgrößen, Haltbarkeit, Ofen-/Personal-/Transportkapazität und freigegebene Qualitätsgrenzen.
5. Heutige manuelle oder Software-Planmengen als echte Konkurrenz. Wer entscheidet und wie kann eine Prognose diese Entscheidung ändern?

Fehlen Ausverkaufs-/Retourendaten, zuerst klein mit Erfassung beginnen. Wenige wiederkehrende Artikel, Filialwechsel, kurze Saisonhistorie und bestehende Spezialsoftware können ein Individualmodell unwirtschaftlich machen. Erst historische Zusatzleistung, dann parallele Vorschläge, dann begrenzter kontrollierter Einsatz. Die Bedienperson behält Mengenfreigabe; Rezeptur und Lebensmittelsicherheit werden nicht modellgesteuert verändert.

## Reproduzierbare Dateien

- `config.json`: Seed, Perioden, Modellinput und Annahmen.
- `data_dictionary.json`: Spalten und Bedeutung.
- `data/observed_sales.csv`: öffentlich vorzeigbare synthetische POS-Daten.
- `data/weather_forecast_vintages.csv`: echte Stichtagsstruktur der synthetischen Vorhersagen.
- `audit_private/latent_demand.csv`: reine Simulationswahrheit; kein Trainingseingang.
- `data/test_predictions.csv`, `segment_metrics.csv`, `day_forecast_errors.csv`: prüfbare Fehler.
- `audit_private/test_economic_replay.csv`: individuelle Gegenfaktualabrechnung der simulierten Politiken.
- `data/day_economic_results.csv`, `economic_results.json`: wirtschaftliche Ergebnisse.
- `model/*joblib`, `calibration.json`: trainierte Parameter und empirische Residualkalibrierung.
- `verification.json`: sinnvolle Konsistenz- und Zukunftsdatenprüfungen.
- `charts/*.png` und `*.svg`: präsentierbare und exportierbare Grafiken.

Bei Reproduktion werden feste Seeds verwendet. Laufzeit dieser Ausführung: {summary['runtime_seconds']:.1f} Sekunden. Versionsabhängige geringe numerische Abweichungen sind möglich.
'''
    (output/"METHODS.md").write_text(body,encoding="utf-8")
    card=f'''# Model Card – IlmBrot

- Modell: {summary['selected_model']}; echte Regression, kein Sprachmodell als numerischer Forecaster.
- Label: beobachtete nächste-Tages-Verkäufe, zensierte Tage mit reduziertem Gewicht.
- Nutzung: menschlich freigegebene Mengenempfehlung für kurze wiederkehrende Produktionsentscheidungen.
- Nicht geeignet: automatische Lebensmittelsicherheits-, Rezeptur-, Mitarbeiterleistungs- oder Bonitätsentscheidung.
- Test: voller zeitlich nachgelagerter Jahresblock; Modell/Residualverteilung davor eingefroren.
- Test-MAE: {summary['test_mae']:.3f}; WAPE: {summary['test_wape']:.3%}; nominell 80-%-Intervall, empirisch {summary['interval_coverage']:.3%} auf nicht ausverkauften Tagen.
- Grenzen: Verkaufszensierung, Selektion nicht ausverkaufter Tage, künstliche Daten, kein closed-loop Feedback, h>1 nicht als operative Strategie validiert, Integration/Einführungskosten fehlen.
- KI-Rolle in einem echten Projekt: Spezifikation, Code und Tests unterstützen; Zahlen deterministisch berechnen; Datenfreigabe und fachliche Prüfung bleiben erforderlich. Latente Simulationswahrheit darf auch ein KI-Assistent nicht als Input übernehmen.
- Betreiberverantwortung: Datenqualität/Drift beobachten, stichtagsgerechte Inputs kontrollieren, einfache Baselines weiterführen und Vorschläge bei Ausreißern ablehnen.
'''
    (output/"MODEL_CARD.md").write_text(card,encoding="utf-8")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"cases"/"bakery")
    parser.add_argument("--fast",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.fast),ensure_ascii=False,indent=2))
