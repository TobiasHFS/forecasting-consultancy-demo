"""Reproducible, deliberately synthetic beverage replenishment experiment.

The generator is the only component that sees latent demand. Training and order
decisions receive fulfilled POS sales, stockout flags and time-stamped plans.
All forecast origins are after closing. This module owns cases/beverage only.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import joblib

SEED = 5142026
START = pd.Timestamp("2023-01-01")
TODAY = pd.Timestamp("2026-10-14")
TRAIN_END = pd.Timestamp("2024-06-30")
VAL_END = pd.Timestamp("2024-12-31")
CAL_END = pd.Timestamp("2025-06-30")
TEST_END = pd.Timestamp("2026-06-30")
FUTURE_FIT_END = pd.Timestamp("2026-07-31")
FEATURES = ["sku_id", "store_id", "category_id", "horizon", "weekday", "open",
            "year_sin", "year_cos", "temperature_forecast", "rain_probability",
            "known_promotion", "last_observed_sales", "mean_7", "mean_28",
            "same_weekday_mean", "sales_sd_28", "stockout_fraction_28", "time_index"]


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, (pd.Timestamp, Path)): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def sku_dictionary():
    # Sale units are packs/crates, not individual bottles. VAT/deposits excluded.
    names = ["Mineralwasser 12er", "Sprudel 6er", "Pils 20er", "Alkoholfrei 20er",
             "Cola 6er", "Apfelschorle 6er", "Saft 6er", "Craftbier 12er"]
    costs = [4.2, 3.8, 11.5, 10.6, 6.0, 5.6, 8.1, 14.0]
    margins = [2.2, 2.0, 4.2, 4.0, 2.8, 2.8, 3.4, 5.0]
    packs = [8, 8, 6, 6, 8, 8, 6, 4]
    category = [0, 0, 1, 1, 2, 2, 3, 1]
    rows = []
    for store in range(2):
        for sku in range(8):
            rows.append(dict(key=store*8+sku, store_id=store, store=f"Filiale {store+1}",
                             sku_id=sku, sku=names[sku], category_id=category[sku],
                             unit_cost=costs[sku], unit_margin=margins[sku],
                             net_price=costs[sku]+margins[sku], casepack=packs[sku], lead_business_days=2))
    return pd.DataFrame(rows)


def add_business_days(day, n):
    result = pd.Timestamp(day)
    while n:
        result += pd.Timedelta(days=1)
        if result.weekday() < 5: n -= 1
    return result


def next_review(day):
    result = pd.Timestamp(day)+pd.Timedelta(days=1)
    while result.weekday() not in (2, 5): result += pd.Timedelta(days=1)
    return result


def protection_calendar(origin):
    """Sales days before the next review's replenishment arrives pre-opening."""
    cutoff=add_business_days(next_review(pd.Timestamp(origin)),2)
    return pd.date_range(pd.Timestamp(origin)+pd.Timedelta(days=1),cutoff,inclusive="left")


def observed_depletion(closing_stock, day):
    """POS-visible exhaustion; does not ask whether hidden demand was lost."""
    return (np.asarray(closing_stock)==0)&(pd.Timestamp(day).weekday()!=6)


@dataclass
class World:
    dates: pd.DatetimeIndex
    skus: pd.DataFrame
    weather: pd.DataFrame
    bookings: pd.DataFrame
    promotions: pd.DataFrame
    unbooked: np.ndarray
    booked: np.ndarray
    forecast_beta: np.ndarray
    rain_base: np.ndarray

    def __post_init__(self):
        self.booking_lookup = {}
        for rec in self.bookings.to_dict("records"):
            self.booking_lookup.setdefault(rec["date"], []).append(rec)
        self.promo_lookup = {}
        for rec in self.promotions.to_dict("records"):
            self.promo_lookup.setdefault(rec["date"], []).append(rec)

    def known_bookings(self, origin, target):
        values = np.zeros(16)
        for rec in self.booking_lookup.get(pd.Timestamp(target), []):
            if rec["announced_at"] <= pd.Timestamp(origin): values[rec["key"]] += rec["units"]
        return values

    def known_promo(self, origin, target):
        values = np.zeros(16)
        for rec in self.promo_lookup.get(pd.Timestamp(target), []):
            if rec["announced_at"] <= pd.Timestamp(origin): values[rec["key"]] = rec["discount"]
        return values

    def weather_forecast(self, origin_idx, target):
        # Expanding climatology is learned ONLY from weather observed by origin.
        origin = self.dates[origin_idx]
        horizon = (pd.Timestamp(target)-origin).days
        angle = 2*np.pi*(pd.Timestamp(target).dayofyear/365.25)
        target_x = np.array([1., np.sin(angle), np.cos(angle)])
        oa = 2*np.pi*(origin.dayofyear/365.25)
        origin_x = np.array([1., np.sin(oa), np.cos(oa)])
        beta = self.forecast_beta[origin_idx]
        anomaly = self.weather.temperature.iloc[origin_idx]-origin_x@beta
        temp = np.clip(target_x@beta+(0.72**horizon)*anomaly, -12, 36)
        wet = self.weather.rain.iloc[max(0, origin_idx-6):origin_idx+1].mean()
        rain = np.clip(.6*self.rain_base[origin_idx]+.4*wet*(.85**horizon), .05, .9)
        return float(temp), float(rain)

    def features(self, origin_idx, target, sales, stockouts):
        if origin_idx < 0: raise ValueError("No pre-origin observations")
        target = pd.Timestamp(target)
        origin = self.dates[origin_idx]
        if target <= origin: raise ValueError("Targets must be after the closing origin")
        # Read an explicit prefix, never future simulator values.
        lo = max(0, origin_idx-55)
        hist = np.array(sales[lo:origin_idx+1], copy=True)
        out = np.array(stockouts[lo:origin_idx+1], copy=True)
        hist[out] = np.nan
        open_past = self.dates[lo:origin_idx+1].weekday != 6
        hist[~open_past] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean7 = np.nanmean(hist[-7:], axis=0)
            mean28 = np.nanmean(hist[-28:], axis=0)
            same = np.nanmean(hist[self.dates[lo:origin_idx+1].weekday == target.weekday()], axis=0)
            sd = np.nanstd(hist[-28:], axis=0)
        mean7 = np.nan_to_num(mean7, nan=8.)
        mean28 = np.nan_to_num(mean28, nan=8.)
        same = np.where(np.isfinite(same), same, mean28)
        sd = np.nan_to_num(sd, nan=3.)
        temp, rain = self.weather_forecast(origin_idx, target)
        angle = 2*np.pi*target.dayofyear/365.25
        x = np.column_stack([
            self.skus.sku_id, self.skus.store_id, self.skus.category_id,
            np.repeat((target-origin).days, 16), np.repeat(target.weekday(), 16),
            np.repeat(int(target.weekday()!=6),16), np.repeat(np.sin(angle),16),
            np.repeat(np.cos(angle),16), np.repeat(temp,16), np.repeat(rain,16),
            self.known_promo(origin,target), sales[origin_idx], mean7, mean28, same,
            sd, out[-28:].mean(axis=0), np.repeat(origin_idx/365.25,16)])
        if not np.isfinite(x).all(): raise AssertionError("Nonfinite features")
        return x


def make_world():
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, TODAY)
    n = len(dates)
    skus = sku_dictionary()
    angle = 2*np.pi*(dates.dayofyear.to_numpy()/365.25)
    temp_season = 11.5-11.5*np.cos(angle-.5)
    anomaly = np.zeros(n)
    wet_latent = np.zeros(n)
    common = np.zeros(n)
    for i in range(1,n):
        anomaly[i] = .84*anomaly[i-1]+rng.normal(0,1.55)
        wet_latent[i] = .58*wet_latent[i-1]+rng.normal(0,.7)
        common[i] = .70*common[i-1]+rng.normal(0,.065)
    temperature = temp_season+anomaly
    rain_probability = 1/(1+np.exp(-(-.7+wet_latent)))
    rain = rng.binomial(1,rain_probability)
    weather = pd.DataFrame(dict(date=dates, temperature=temperature, rain=rain))
    # Archived synthetic weather forecasts have a causal origin. No future actual
    # temperature, latent weather mean or simulator demand mean is a feature.
    wx = np.column_stack([np.ones(n),np.sin(angle),np.cos(angle)])
    xtx = np.eye(3)*np.array([.01,25.,25.])
    xty = np.zeros(3)
    beta = np.zeros((n,3)); rain_base = np.zeros(n)
    for i in range(n):
        xtx += np.outer(wx[i],wx[i]); xty += wx[i]*temperature[i]
        beta[i] = np.linalg.solve(xtx,xty)
        rain_base[i] = rain[max(0,i-89):i+1].mean()
    # Some future contracts are already public today; future unregistered orders
    # remain unavailable at the origin, including historical backtest origins.
    bookings=[]; plan_dates=pd.date_range(START,TODAY+pd.Timedelta(days=10))
    for day in plan_dates:
        if day.weekday()==6: continue
        for store in range(2):
            if rng.random() < (.22 if day.weekday() in (4,5) else .035):
                lead=int(rng.choice([1,2,5,10,17,24],p=[.07,.10,.16,.24,.25,.18]))
                chosen=rng.choice([0,1,2,3,4],size=2,replace=False)
                size=int(rng.integers(20,85))
                for sku in chosen:
                    bookings.append(dict(date=day,announced_at=day-pd.Timedelta(days=lead),
                                         key=store*8+int(sku),units=size,kind="registrierte Fest-/Firmenbestellung"))
    bookings=pd.DataFrame(bookings)
    promotions=[]
    for start in pd.date_range(START+pd.Timedelta(days=7),TODAY+pd.Timedelta(days=10),freq="35D"):
        sku=int(rng.integers(0,8)); discount=float(rng.choice([.05,.10,.15]))
        for day in pd.date_range(start,start+pd.Timedelta(days=5)):
            for store in range(2):
                promotions.append(dict(date=day,announced_at=start-pd.Timedelta(days=14),
                                       key=store*8+sku,discount=discount))
    promotions=pd.DataFrame(promotions)
    booked=np.zeros((n,16),dtype=int)
    actual_promo=np.zeros((n,16))
    for rec in bookings.to_dict("records"):
        idx=(rec["date"]-START).days
        if 0 <= idx < n: booked[idx,rec["key"]]+=rec["units"]
    for rec in promotions.to_dict("records"):
        idx=(rec["date"]-START).days
        if 0 <= idx < n: actual_promo[idx,rec["key"]]=rec["discount"]
    base=np.array([15.,12.,8.,7.,6.,5.,4.,3.])
    unbooked=np.zeros((n,16),dtype=int)
    for i,day in enumerate(dates):
        if day.weekday()==6: continue
        for key in range(16):
            sku=key%8; store=key//8
            heat=max(temperature[i]-15,0)
            heat_factor=(1+.055*heat**1.20) if sku in (0,1) else (1+.035*heat if sku in (2,3,4,5) else 1.)
            weekend=(1.30 if day.weekday() in (4,5) else .92)
            if sku in (2,3,7) and day.weekday() in (4,5): weekend*=1.20
            seasonal=1+.11*np.sin(angle[i]-.5)
            regime=1.
            if day>=pd.Timestamp("2025-09-15"):
                regime=1.12 if sku in (0,1,3) else (.9 if sku==2 else 1.)
            mu=base[sku]*(1.17 if store==0 else .89)*heat_factor*weekend*seasonal*regime
            mu*=np.exp(common[i])*(1+2.0*actual_promo[i,key])*(.94 if rain[i] else 1.)
            # Overdispersion and shared innovations: even the correct conditional
            # mean could not predict each realized count.
            shape=42.
            unbooked[i,key]=rng.negative_binomial(shape,shape/(shape+mu))
    return World(dates,skus,weather,bookings,promotions,unbooked,booked,beta,rain_base)


def seasonal_prediction(x):
    # Strong, availability-aware seasonal comparator, rather than yesterday-only.
    result=.65*x[:,14]+.35*x[:,13]
    result*=np.clip(1+1.5*x[:,10],1,1.3)
    return np.where(x[:,5]>0,np.clip(result,0,None),0)


def predict(estimator, kind, x):
    if kind=="seasonal_minmax": result=seasonal_prediction(x)
    else: result=np.clip(estimator.predict(x),0,None)
    return np.where(x[:,5]>0,result,0.)


class ErrorPool:
    def __init__(self, residuals, start_weekday=0):
        self.residuals=np.asarray(residuals,float)
        self.start_weekday=int(start_weekday)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            self.q05=np.nanquantile(self.residuals,.05,axis=0)
            self.q90=np.nanquantile(self.residuals,.90,axis=0)
            self.q95=np.nanquantile(self.residuals,.95,axis=0)
        self.q05=np.nan_to_num(self.q05,nan=-5.)
        self.q90=np.nan_to_num(self.q90,nan=5.)
        self.q95=np.nan_to_num(self.q95,nan=8.)
        self.filled=np.where(np.isfinite(self.residuals),self.residuals,0.)
    def intervals(self, point):
        return np.clip(point+self.q05,0,None), np.clip(point+self.q95,0,None),np.clip(point+self.q90,0,None)
    def protection_quantile(self, points, origin_idx, service=.90, draws=100, open_mask=None):
        # Joint contiguous residual blocks retain temporal/cross-SKU dependence.
        # Sum paths, THEN take the aggregate quantile. Never sum daily quantiles.
        length=len(points); max_start=max(1,len(self.filled)-length+1)
        rng=np.random.default_rng(SEED+origin_idx*139)
        first_weekday=(START.weekday()+origin_idx+1)%7
        eligible=np.array([i for i in range(max_start) if (self.start_weekday+i)%7==first_weekday])
        starts=rng.choice(eligible if len(eligible) else np.arange(max_start),size=draws)
        noise=np.stack([self.filled[s:s+length] for s in starts])
        if noise.shape[1]!=length:
            noise=np.resize(noise,(draws,length,16))
        paths=np.maximum(0,np.asarray(points)[None,:,:]+noise)
        # Availability is a known calendar input, not inferred from a zero
        # forecast. Positive resampled residuals must not create Sunday demand;
        # conversely a zero forecast on an OPEN day retains uncertainty.
        if open_mask is None:
            open_mask=(first_weekday+np.arange(length))%7!=6
        open_mask=np.asarray(open_mask,dtype=bool)
        if open_mask.shape==(length,):
            paths[:,~open_mask,:]=0.
        elif open_mask.shape==np.asarray(points).shape:
            paths=np.where(open_mask[None,:,:],paths,0.)
        else:
            raise ValueError("Opening mask must match days or days-by-SKU points")
        return np.quantile(paths.sum(axis=1),service,axis=0)


def supplier_arrival(origin, key, stress=False):
    nominal=add_business_days(origin,2)
    # Order decision never receives this latent shock. Same origin/SKU key gives
    # both policies the same potential supplier shock.
    seed=SEED+int((origin-START).days)*997+(key%8)*7993
    rng=np.random.default_rng(seed)
    p=.23 if stress else .09
    delay=int(rng.choice([0,1,2],p=[1-p,.7*p,.3*p]))
    return add_business_days(nominal,delay),delay


def allocate_packs(wanted, free, packs, position, targets):
    allocated=np.zeros(16,dtype=int)
    remaining=np.maximum(wanted,0).astype(int)
    # Same rule for both policies: protect low relative coverage first, with
    # fixed item-key tie breaking. Capacity includes on-order reservations.
    while True:
        eligible=np.flatnonzero((remaining>=packs)&(packs<=free))
        if len(eligible)==0: break
        coverage=(position[eligible]+allocated[eligible])/np.maximum(targets[eligible],1.)
        key=int(eligible[np.argmin(coverage)])
        allocated[key]+=packs[key]; remaining[key]-=packs[key]; free-=packs[key]
    return allocated


def inventory_replay(world, start_idx, end_idx, kind="seasonal_minmax", estimator=None,
                     pool=None, reference_sales=None, reference_oos=None, initial_stock=None,
                     initial_pipeline=None, policy="reference", margin_scale=1.,
                     demand_shift=False, supplier_stress=False, bad_forecast=False):
    n=len(world.dates); skus=world.skus
    sales=np.zeros((n,16)); oos=np.zeros((n,16),bool)
    if reference_sales is not None:
        sales[:start_idx]=reference_sales[:start_idx]
        oos[:start_idx]=reference_oos[:start_idx]
    packs=skus.casepack.to_numpy(int); costs=skus.unit_cost.to_numpy(float)
    margins=skus.unit_margin.to_numpy(float)*margin_scale
    stock=(np.ones(16,dtype=int)*96 if initial_stock is None else np.array(initial_stock,int).copy())
    pipeline=[dict(o) for o in (initial_pipeline or [])]
    capacity=1650
    if stock.sum()+sum(o["qty"] for o in pipeline)>capacity: raise AssertionError("Initial capacity exceeded")
    daily=[]; orders=[]; snapshots={}
    for idx in range(start_idx,end_idx+1):
        day=world.dates[idx]; opening=stock.copy(); arrivals=np.zeros(16,dtype=int)
        active=[]
        for order in pipeline:
            if order["arrival"]<=day: arrivals[order["key"]]+=order["qty"]
            else: active.append(order)
        pipeline=active; stock+=arrivals
        ub=world.unbooked[idx].copy(); bd=world.booked[idx].copy()
        if demand_shift and day>=pd.Timestamp("2026-03-01"):
            ub=np.rint(ub*np.array([1.40 if k%8 in (0,1,3) else 1.15 for k in range(16)])).astype(int)
        served_booked=np.minimum(stock,bd); stock-=served_booked
        served_unbooked=np.minimum(stock,ub); stock-=served_unbooked
        fulfilled=served_booked+served_unbooked; lost=bd+ub-fulfilled
        observable_exhaustion=observed_depletion(stock,day)
        sales[idx]=served_unbooked; oos[idx]=observable_exhaustion
        onorder=np.zeros(16,dtype=int)
        for order in pipeline: onorder[order["key"]]+=order["qty"]
        new=np.zeros(16,dtype=int); targets=np.zeros(16)
        if day.weekday() in (2,5):
            next_arrival=add_business_days(next_review(day),2)
            protection_dates=protection_calendar(day)
            points=[]; known=[]
            for target in protection_dates:
                x=world.features(idx,target,sales,oos)
                pt=predict(estimator,kind,x)
                if bad_forecast: pt*=.70  # Deliberate failure mode, not tuning.
                points.append(pt); known.append(world.known_bookings(day,target))
            if pool is None:
                # Historical operator comparison; safety estimated from observed
                # sales only. Used to create the reference POS/censoring history.
                arr=np.asarray(points)
                targets=arr.sum(axis=0)+1.30*np.sqrt(len(arr))*np.maximum(x[:,15],3.)
            else:
                targets=pool.protection_quantile(points,idx,service=.90,
                                                 open_mask=protection_dates.weekday!=6)
            targets+=np.asarray(known).sum(axis=0)
            position=stock+onorder
            wanted=np.ceil(np.maximum(targets-position,0)/packs).astype(int)*packs
            new=allocate_packs(wanted,capacity-int(position.sum()),packs,position,targets)
            for key in np.flatnonzero(new):
                arrival,delay=supplier_arrival(day,int(key),supplier_stress)
                record=dict(key=int(key),qty=int(new[key]),origin=day,nominal_arrival=add_business_days(day,2),arrival=arrival)
                pipeline.append(record); onorder[key]+=new[key]
                orders.append(dict(policy=policy,origin=day,key=int(key),store_id=int(skus.store_id.iloc[key]),
                                   sku_id=int(skus.sku_id.iloc[key]),order_qty=int(new[key]),
                                   nominal_arrival=record["nominal_arrival"],actual_arrival=arrival,
                                   supplier_delay_business_days=delay,inventory_position_before=int(position[key]),
                                   protection_target=float(targets[key]),protection_end=protection_dates[-1],
                                   next_review_nominal_arrival=next_arrival))
        if stock.sum()+onorder.sum()>capacity: raise AssertionError("Shared capacity exceeded")
        order_cost=(10.+.40*np.count_nonzero(new)) if new.sum() else 0.
        promo=world.known_promo(day,day)
        retail_margin=(skus.net_price.to_numpy()*(1-promo)-costs)*margin_scale
        contribution=served_booked*margins+served_unbooked*retail_margin
        lost_contribution=(bd-served_booked)*margins+(ub-served_unbooked)*retail_margin
        capital=stock*costs*.08/365.25
        storage=stock*.0035
        holding=capital+storage
        for key in range(16):
            daily.append(dict(date=day,policy=policy,key=key,store_id=key//8,sku_id=key%8,
                              opening_stock=int(opening[key]),arrivals=int(arrivals[key]),
                              unbooked_demand=int(ub[key]),booked_demand=int(bd[key]),
                              unbooked_sales=int(served_unbooked[key]),booked_sales=int(served_booked[key]),
                              sales=int(fulfilled[key]),lost_units=int(lost[key]),stockout=bool(observable_exhaustion[key]),
                              closing_stock=int(stock[key]),on_order=int(onorder[key]),order_qty=int(new[key]),
                              inventory_value_eur=float(stock[key]*costs[key]),capital_eur=float(capital[key]),
                              storage_eur=float(storage[key]),holding_eur=float(holding[key]),
                              order_cost_eur=float(order_cost/16),expediting_eur=0.,
                              contribution_eur=float(contribution[key]),lost_contribution_eur=float(lost_contribution[key]),
                              net_contribution_eur=float(contribution[key]-holding[key]-order_cost/16)))
        if day in (CAL_END,TEST_END,TODAY):
            snapshots[str(day.date())]=(stock.copy(),[dict(o) for o in pipeline])
    return pd.DataFrame(daily),pd.DataFrame(orders),sales,oos,snapshots


def dataset(world, sales, oos):
    records=[]; features=[]
    for idx in range(56,len(world.dates)):
        day=world.dates[idx]
        for horizon in range(1,9):
            keys=[k for k in range(16) if 1+((idx*3+k)%8)==horizon]
            oi=idx-horizon
            matrix=world.features(oi,day,sales,oos)
            for key in keys:
                features.append(matrix[key])
                records.append(dict(date=day,origin=world.dates[oi],key=key,store_id=key//8,sku_id=key%8,
                                    actual_observed_unbooked_sales=float(sales[idx,key]),stockout=bool(oos[idx,key]),
                                    open=day.weekday()!=6,known_bookings=float(world.known_bookings(world.dates[oi],day)[key]),
                                    feature_max_observed_date=world.dates[oi]))
    return np.asarray(features),pd.DataFrame(records)


def metric(y,point,lower,upper,q90):
    y=np.asarray(y); error=point-y
    return dict(n=len(y),mae=float(np.mean(np.abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
                wape=float(np.sum(np.abs(error))/max(np.sum(y),1)),bias=float(np.mean(error)),
                pinball_q90=float(np.mean(np.maximum(.90*(y-q90),-.10*(y-q90)))),
                coverage_90=float(np.mean((y>=lower)&(y<=upper))),interval_width=float(np.mean(upper-lower)))


def error_pool(meta, mask, point):
    frame=meta.loc[mask,["date","key","stockout"]].copy()
    frame["residual"]=meta.loc[mask,"actual_observed_unbooked_sales"].to_numpy()-point
    frame.loc[frame.stockout,"residual"]=np.nan
    matrix=frame.pivot(index="date",columns="key",values="residual").reindex(columns=range(16))
    return ErrorPool(matrix.to_numpy(),matrix.index[0].weekday())


def train_models(x,meta,fast):
    valid=(~meta.stockout)&meta.open
    train=valid&(meta.date<=TRAIN_END)
    val=valid&(meta.date>TRAIN_END)&(meta.date<=VAL_END)&(meta.origin>=TRAIN_END)
    calib=(meta.date>VAL_END)&(meta.date<=CAL_END)&(meta.origin>=VAL_END)
    test=(meta.date>CAL_END)&(meta.date<=TEST_END)&(meta.origin>=CAL_END)
    y=meta.actual_observed_unbooked_sales.to_numpy()
    candidates=[]
    base=predict(None,"seasonal_minmax",x[val]); candidates.append(dict(model="seasonal_minmax",validation_mae=float(np.mean(np.abs(base-y[val]))),params={}))
    for alpha in (20.,100.,500.):
        prep=ColumnTransformer([("category",OneHotEncoder(handle_unknown="ignore",sparse_output=False),[0,1,2]),
                                ("numeric",StandardScaler(),list(range(3,len(FEATURES))))])
        ridge=make_pipeline(prep,Ridge(alpha=alpha))
        ridge.fit(x[train],y[train]); pr=predict(ridge,"ridge",x[val])
        candidates.append(dict(model="ridge",validation_mae=float(np.mean(np.abs(pr-y[val]))),params={"alpha":alpha},estimator=ridge))
    configurations=[dict(max_depth=3,min_samples_leaf=60,l2_regularization=20.,max_iter=160 if fast else 220,learning_rate=.07),
                    dict(max_depth=4,min_samples_leaf=45,l2_regularization=35.,max_iter=180 if fast else 260,learning_rate=.055)]
    for pars in configurations:
        model=HistGradientBoostingRegressor(loss="poisson",early_stopping=False,categorical_features=[0,1,2],random_state=SEED,**pars)
        model.fit(x[train],y[train]); pr=predict(model,"hist_gradient_boosting",x[val])
        candidates.append(dict(model="hist_gradient_boosting",validation_mae=float(np.mean(np.abs(pr-y[val]))),params=pars,estimator=model))
    winner=min(candidates,key=lambda r:r["validation_mae"])
    # Refit each best family only on train+validation; calibration/test untouched.
    models={"seasonal_minmax":None}; selections={}
    fit=valid&(meta.date<=VAL_END)
    for family in ("ridge","hist_gradient_boosting"):
        candidate=min([r for r in candidates if r["model"]==family],key=lambda r:r["validation_mae"])
        estimator=candidate["estimator"]
        estimator.fit(x[fit],y[fit]); models[family]=estimator; selections[family]=candidate["params"]
    pools={name:error_pool(meta,calib,predict(est,name,x[calib])) for name,est in models.items()}
    best_simple=min([r for r in candidates if r["model"] in ("ridge","seasonal_minmax")],key=lambda r:r["validation_mae"])["model"]
    selection=dict(selected_model=winner["model"],best_simple_model=best_simple,
                   selection_rule="Minimum MAE on uncensored open validation days; no test economics used. Origins earlier than each frozen fit/calibration cutoff are purged.",
                   candidates=[{k:v for k,v in r.items() if k!="estimator"} for r in candidates],
                   family_params=selections,fit_period=[str(START.date()),str(VAL_END.date())],
                   validation_period=[str((TRAIN_END+pd.Timedelta(days=1)).date()),str(VAL_END.date())],
                   calibration_period=[str((VAL_END+pd.Timedelta(days=1)).date()),str(CAL_END.date())],
                   test_period=[str((CAL_END+pd.Timedelta(days=1)).date()),str(TEST_END.date())],
                   early_stopping=False,random_row_cv=False,
                   comparator="Availability-aware seasonal/minmax with identical known contracts, shared capacity, casepacks, review dates and calibrated period safety.")
    return models,pools,selection,calib,test,valid


def summarize_inventory(frame,policy):
    daily=frame.groupby("date",sort=True).agg({"inventory_value_eur":"sum","stockout":"sum"})
    end=frame[frame.date==frame.date.max()]
    begin=frame[frame.date==frame.date.min()]
    costs=sku_dictionary().unit_cost.to_numpy()
    return dict(policy=policy,served_units=int(frame.sales.sum()),lost_units=int(frame.lost_units.sum()),
                lost_contribution_eur=float(frame.lost_contribution_eur.sum()),contribution_eur=float(frame.contribution_eur.sum()),
                holding_eur=float(frame.holding_eur.sum()),capital_eur=float(frame.capital_eur.sum()),
                storage_eur=float(frame.storage_eur.sum()),order_cost_eur=float(frame.order_cost_eur.sum()),expediting_eur=0.,
                net_contribution_eur=float(frame.net_contribution_eur.sum()),
                average_inventory_value_eur=float(daily.inventory_value_eur.mean()),
                initial_inventory_value_eur=float(np.sum(begin.opening_stock.to_numpy()*costs[begin.key.to_numpy()])),
                ending_inventory_value_eur=float(end.inventory_value_eur.sum()),
                ending_on_order_units=int(end.on_order.sum()),
                ending_on_order_commitment_eur=float(np.sum(end.on_order.to_numpy()*costs[end.key.to_numpy()])),
                stockout_days=int((daily.stockout>0).sum()),stockout_sku_days=int(frame.stockout.sum()))


def economic_comparison(base,chosen,draws=1000):
    b=base.groupby("date").net_contribution_eur.sum(); m=chosen.groupby("date").net_contribution_eur.sum()
    diff=(m-b).to_numpy(); n=len(diff); rng=np.random.default_rng(SEED+451)
    stats=[]; block=14
    for _ in range(draws):
        starts=rng.integers(0,n,size=math.ceil(n/block))
        indices=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
        stats.append(float(diff[indices].sum()))
    low,high=np.quantile(stats,[.025,.975])
    released=base.groupby("date").inventory_value_eur.sum().mean()-chosen.groupby("date").inventory_value_eur.sum().mean()
    gain=float(diff.sum()); setup=1800.; monthly_provider=150.; monthly_internal=50.
    return dict(annual_gain_eur=gain,bootstrap_ci_low_eur=float(low),bootstrap_ci_high_eur=float(high),
                inventory_released_eur=float(released),adoption_cost_eur=setup,monthly_overhead_eur=monthly_provider+monthly_internal,
                monthly_provider_fee_eur=monthly_provider,monthly_internal_effort_eur=monthly_internal,
                first_year_net_gain_eur=gain-setup-12*(monthly_provider+monthly_internal),
                bootstrap=dict(kind="paired circular moving block bootstrap",block_days=block,draws=draws,
                               interpretation="Conditional on this synthetic world, policies and costs; not customer ROI confidence. Seasonal/nonstationary uncertainty is not covered."),
                inventory_release_is_profit=False)


def charts(output,meta,predictions,trace,selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":140})
    charts_dir=output/"charts"; charts_dir.mkdir(exist_ok=True)
    selected_rows=predictions[(predictions.model==selected)&(predictions.key==0)].set_index("date")
    baseline_rows=predictions[(predictions.model=="seasonal_minmax")&(predictions.key==0)].set_index("date")
    show=selected_rows.loc["2026-05-01":"2026-06-30"]
    fig,ax=plt.subplots(figsize=(11,4.2))
    ax.plot(show.index,show.actual_observed_unbooked_sales,label="Beobachtete freie Verkäufe",color="#193847",lw=1.4)
    ax.plot(show.index,show.forecast_unbooked,label=f"Modell: {selected}",color="#007c83",lw=1.7)
    ax.plot(show.index,baseline_rows.loc[show.index].forecast_unbooked,label="Saisonaler Vergleich",color="#d08b32",lw=1.2)
    ax.fill_between(show.index,show.lower_90,show.upper_90,color="#007c83",alpha=.13,label="empirisches 90%-Band")
    ax.set(title="Getränkedemo · Mineralwasser, Filiale 1 · unangetasteter Test",ylabel="Verkaufseinheiten / Tag")
    ax.legend(ncol=2,fontsize=8); fig.tight_layout()
    for ext in ("png","svg"): fig.savefig(charts_dir/f"forecast_test.{ext}")
    plt.close(fig)
    daily=trace.groupby(["date","policy"]).agg(inventory_value_eur=("inventory_value_eur","sum"),lost_units=("lost_units","sum"),net=("net_contribution_eur","sum")).reset_index()
    fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for name,color in (("seasonal_minmax","#d08b32"),("selected_model","#007c83")):
        d=daily[daily.policy==name].set_index("date")
        axs[0].plot(d.index,d.inventory_value_eur.rolling(14,min_periods=1).mean(),label=name,color=color)
        axs[1].plot(d.index,d.lost_units.rolling(14,min_periods=1).sum(),label=name,color=color)
    axs[0].set(ylabel="Lagerwert, 14-Tage-Mittel (€)",title="Gleiche Nachfrage und Lieferstörungen; eigene Verkaufshistorie je Policy")
    axs[1].set(ylabel="Fehlmengen in 14 Tagen")
    axs[0].legend(); fig.tight_layout()
    for ext in ("png","svg"): fig.savefig(charts_dir/f"inventory_comparison.{ext}")
    plt.close(fig)


def assertions(world,x,meta,trace,forecast,sales,oos,orders,selection,public_observations):
    checks=[]
    balance=trace.opening_stock+trace.arrivals-trace.sales-trace.closing_stock
    assert (balance==0).all(); checks.append("Stock balance: opening + arrivals - served sales == closing for every SKU/day/policy.")
    assert (trace.sales<=trace.unbooked_demand+trace.booked_demand).all(); checks.append("Served units never exceed the same realized demand.")
    assert (trace.closing_stock>=0).all() and (trace.on_order>=0).all()
    capacity=trace.groupby(["date","policy"])[["closing_stock","on_order"]].sum().sum(axis=1)
    assert (capacity<=1650).all(); checks.append("Shared stock + reserved pipeline capacity <= 1650.")
    cp=world.skus.casepack.to_numpy()[trace.key.to_numpy()]
    assert (trace.order_qty.to_numpy()%cp==0).all(); checks.append("All order quantities honor supplier casepacks.")
    assert (meta.feature_max_observed_date<=meta.origin).all() and (meta.origin<meta.date).all()
    checks.append("Feature observations stop at closing origin; every target is later.")
    forbidden={"latent_demand","unbooked_demand","booked_demand","expected_mean","future_actual_temperature"}
    assert forbidden.isdisjoint(FEATURES); checks.append("No latent demand, simulator mean or future actual weather in feature schema.")
    assert forecast.date.min()>TODAY and "actual" not in " ".join(forecast.columns)
    checks.append("Published next-cycle forecast has no future actual observations.")
    assert world.dates.max()==TODAY; checks.append("Latent demand generation stops at 2026-10-14, including private audit truth.")
    for rec in forecast.to_dict("records"):
        expected=world.known_bookings(TODAY,rec["date"])[rec["key"]]
        assert expected==rec["known_bookings"]
    checks.append("Only contracts already registered at 2026-10-14 are direct future inputs.")
    # Adversarial suffix mutation catches future leakage independently of the
    # self-reported max-observation provenance column.
    oi=(pd.Timestamp("2025-08-01")-START).days; target=world.dates[oi]+pd.Timedelta(days=5)
    original=world.features(oi,target,sales,oos)
    changed_sales=sales.copy(); changed_sales[oi+1:]=999999.
    changed_oos=oos.copy(); changed_oos[oi+1:]=~changed_oos[oi+1:]
    old_weather=world.weather.temperature.copy()
    try:
        world.weather.loc[oi+1:,"temperature"]=9999.
        changed=world.features(oi,target,changed_sales,changed_oos)
    finally:
        world.weather["temperature"]=old_weather
    assert np.array_equal(original,changed)
    checks.append("Adversarial future sales/stockout/weather mutation leaves origin features unchanged.")
    transitions=trace.sort_values(["policy","key","date"]).groupby(["policy","key"],sort=False)
    previous=transitions.on_order.shift(1)
    known=previous.notna()
    assert (trace.sort_values(["policy","key","date"]).loc[known,"on_order"].to_numpy()==
            (previous[known]-trace.sort_values(["policy","key","date"]).loc[known,"arrivals"]+
             trace.sort_values(["policy","key","date"]).loc[known,"order_qty"]).to_numpy()).all()
    checks.append("Open-order ledger: prior pipeline - arrivals + new orders == closing pipeline.")
    matched=orders.pivot_table(index=["origin","key"],columns="policy",values="supplier_delay_business_days",aggfunc="first").dropna()
    if matched.shape[1]==2: assert np.array_equal(matched.iloc[:,0],matched.iloc[:,1])
    checks.append("Matched policy orders receive identical supplier shocks.")
    for bound,end in ((TRAIN_END,VAL_END),(VAL_END,CAL_END),(CAL_END,TEST_END)):
        split=(meta.date>bound)&(meta.date<=end)&(meta.origin>=bound)
        assert (meta.loc[split,"origin"]>=bound).all()
    checks.append("All scored chronological origins occur after their fit/calibration information cutoff.")
    # Construct an adversarial positive residual on every sampled day, plus a
    # huge Sunday point. Known closure must still contribute EXACTLY zero.
    test_origin=(pd.Timestamp("2026-10-14")-START).days
    test_dates=pd.date_range(pd.Timestamp("2026-10-15"),periods=7)
    mask=test_dates.weekday!=6
    test_pool=ErrorPool(np.full((35,16),4.))
    test_points=np.full((7,16),2.); test_points[~mask]=99999.
    quantity=test_pool.protection_quantile(test_points,test_origin,open_mask=mask)
    assert np.all(quantity==6.*int(mask.sum()))
    # Zero expected sales on open days is different from known closed days.
    zero_points=np.zeros((7,16))
    assert np.all(test_pool.protection_quantile(zero_points,test_origin,open_mask=mask)==4.*int(mask.sum()))
    checks.append("Protection paths: known closed Sunday contributes exactly zero despite positive residuals/points; open zero-point days retain uncertainty.")
    # An arrival before opening protects its own day. The current review must
    # not also purchase that day's nominal demand as an unnamed extra buffer.
    example_origin=pd.Timestamp("2026-10-14")
    assert next_review(example_origin)==pd.Timestamp("2026-10-17")
    assert add_business_days(next_review(example_origin),2)==pd.Timestamp("2026-10-20")
    assert list(protection_calendar(example_origin))==list(pd.date_range("2026-10-15","2026-10-19"))
    assert pd.Timestamp("2026-10-20") not in protection_calendar(example_origin)
    assert protection_calendar(pd.Timestamp("2026-10-17"))[-1]==pd.Timestamp("2026-10-22")
    checks.append("Protection end exclusive: 14 October review covers sales through 19 October, excluding next pre-opening arrival on 20 October; 17 October review ends 22 October.")
    # Observable depletion includes exact sell-through: lack of hidden lost
    # demand cannot be used to label an apparently identical empty shelf.
    observable=(trace.closing_stock==0)&(trace.date.dt.weekday!=6)
    assert np.array_equal(trace.stockout.to_numpy(),observable.to_numpy())
    public_observable=(public_observations.closing_stock==0)&(public_observations.date.dt.weekday!=6)
    assert np.array_equal(public_observations.stockout.to_numpy(),public_observable.to_numpy())
    assert np.array_equal(observed_depletion([0,1],pd.Timestamp("2026-10-14")),[True,False])
    assert not observed_depletion([0,0],pd.Timestamp("2026-10-18")).any()
    assert {"unbooked_demand","booked_demand","lost_units","expected_mean"}.isdisjoint(public_observations.columns)
    lookup=public_observations.set_index(["date","key"])
    observable_labels=np.array([lookup.loc[(d,k),"unbooked_sales"] for d,k in zip(meta.date,meta.key)])
    assert np.array_equal(meta.actual_observed_unbooked_sales.to_numpy(),observable_labels)
    checks.append("Public stockout equals observable open-day zero closing stock (including exact sell-through); labels are POS sales and public export has no hidden demand/loss/mean.")
    return dict(passed=True,count=len(checks),checks=checks)


def run(output_dir: Path, fast=False)->dict:
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    project=output.parents[1]
    os.environ.setdefault("MPLCONFIGDIR",str(project/".cache"/"matplotlib"))
    with threadpool_limits(limits=2):
        return _run(output,fast)


def _run(output,fast):
    world=make_world(); n=len(world.dates)
    print("beverage: create observed reference POS and inventory history",flush=True)
    reference,reference_orders,sales,oos,snapshots=inventory_replay(world,0,n-1)
    public=reference[["date","store_id","sku_id","key","opening_stock","arrivals","unbooked_sales","booked_sales","sales","stockout","closing_stock"]].copy()
    public.to_csv(output/"public_observations.csv",index=False)
    world.weather.to_csv(output/"public_weather_observations.csv",index=False)
    world.bookings[world.bookings.announced_at<=TODAY].to_csv(output/"public_registered_orders.csv",index=False)
    world.promotions[world.promotions.announced_at<=TODAY].to_csv(output/"public_planned_promotions.csv",index=False)
    world.skus.to_csv(output/"sku_dictionary.csv",index=False)
    audit=output/"audit_private"; audit.mkdir(exist_ok=True)
    truth=reference[["date","key","store_id","sku_id","unbooked_demand","booked_demand"]].copy()
    truth.to_csv(audit/"latent_demand.csv",index=False)
    reference_orders.to_csv(audit/"reference_orders.csv",index=False)
    print("beverage: fit chronological seasonal / ridge / regularized boosting comparison",flush=True)
    x,meta=dataset(world,sales,oos)
    models,pools,selection,calib,test,valid=train_models(x,meta,fast)
    selected=selection["selected_model"]
    predictions=[]; scores=[]
    for name,estimator in models.items():
        point=predict(estimator,name,x[test])
        keys=meta.loc[test,"key"].to_numpy()
        lo=np.clip(point+pools[name].q05[keys],0,None); hi=np.clip(point+pools[name].q95[keys],0,None)
        q90=np.clip(point+pools[name].q90[keys],0,None)
        frame=meta.loc[test].copy(); frame["model"]=name
        frame["forecast_unbooked"]=point; frame["lower_90"]=lo; frame["upper_90"]=hi
        frame["pred_q90"]=q90; frame["forecast_total"]=point+frame.known_bookings.to_numpy()
        use=(~frame.stockout)&frame.open
        scores.append(dict(model=name,**metric(frame.loc[use,"actual_observed_unbooked_sales"].to_numpy(),point[use],lo[use],hi[use],q90[use])))
        predictions.append(frame)
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_csv(output/"test_predictions.csv",index=False)
    meta[["date","origin","key","stockout","feature_max_observed_date"]].to_csv(output/"feature_availability_audit.csv",index=False)
    metrics=dict(case="beverage",target="observed_unbooked_sales on open days without observable end-of-day depletion",test_period=selection["test_period"],
                 forecast_metrics=scores,censored_open_test_fraction=float(meta.loc[test&meta.open,"stockout"].mean()),
                 stockout_semantics="Observable open-day zero closing stock; conservative potential-censoring marker, not latent lost-demand truth",
                 uncertainty="Signed empirical calibration residual quantiles; temporal coverage is measured, not guaranteed.",
                 selection=selected,synthetic=True)
    dump(output/"metrics.json",metrics); dump(output/"model_selection.json",selection)
    for name,estimator in models.items():
        joblib.dump(dict(model=estimator,kind=name,feature_names=FEATURES,fit_through=str(VAL_END.date()),pool=pools[name]),output/f"model_test_{name}.joblib")
    start_idx=(CAL_END+pd.Timedelta(days=1)-START).days; end_idx=(TEST_END-START).days
    init_stock,init_pipe=snapshots[str(CAL_END.date())]
    print(f"beverage: locked test, selected={selected}; paired inventory replay",flush=True)
    kwargs=dict(world=world,start_idx=start_idx,end_idx=end_idx,reference_sales=sales,reference_oos=oos,initial_stock=init_stock,initial_pipeline=init_pipe)
    base,base_orders,*_=inventory_replay(**kwargs,kind="seasonal_minmax",pool=pools["seasonal_minmax"],policy="seasonal_minmax")
    chosen,chosen_orders,*_=inventory_replay(**kwargs,kind=selected,estimator=models[selected],pool=pools[selected],policy="selected_model")
    trace=pd.concat([base,chosen],ignore_index=True); orders=pd.concat([base_orders,chosen_orders],ignore_index=True)
    trace.to_csv(output/"inventory_daily.csv",index=False); orders.to_csv(output/"inventory_orders.csv",index=False)
    paired=economic_comparison(base,chosen,draws=500 if fast else 1500)
    simple=selection["best_simple_model"]
    if simple==selected:
        simple_trace=chosen.copy(); simple_trace["policy"]="best_simple_model"
    elif simple=="seasonal_minmax":
        simple_trace=base.copy(); simple_trace["policy"]="best_simple_model"
    else:
        simple_trace,*_=inventory_replay(**kwargs,kind=simple,estimator=models[simple],pool=pools[simple],policy="best_simple_model")
    simple_trace.to_csv(output/"inventory_daily_simple_comparator.csv",index=False)
    simple_comparison=economic_comparison(simple_trace,chosen,draws=500 if fast else 1500)
    base_end=base[base.date==base.date.max()].inventory_value_eur.sum()
    model_end=chosen[chosen.date==chosen.date.max()].inventory_value_eur.sum()
    paired["incremental_cash_proxy_before_adoption_eur"]=paired["annual_gain_eur"]-float(model_end-base_end)
    paired["cash_proxy_assumption"]="Immediate customer collection and supplier payment on arrival; unpaid closing pipeline is a separate commitment. Not a full real first-year cashflow."
    stress=[]
    for label,args in (("25–30 % schlechtere Modellprognose",dict(bad_forecast=True)),
                       ("Nachfragesprung ab März",dict(demand_shift=True)),
                       ("Häufigere Lieferverzögerungen",dict(supplier_stress=True))):
        b,*_=inventory_replay(**kwargs,kind="seasonal_minmax",pool=pools["seasonal_minmax"],policy="seasonal_minmax",**{k:v for k,v in args.items() if k!="bad_forecast"})
        m,*_=inventory_replay(**kwargs,kind=selected,estimator=models[selected],pool=pools[selected],policy="selected_model",**args)
        comparison=economic_comparison(b,m,draws=300 if fast else 700)
        stress.append(dict(name=label,**comparison))
    # Pure costing stress uses the identical decisions/sales: does not pretend
    # that prices or quantities respond when unit contribution is reduced.
    low_base=base.copy(); low_chosen=chosen.copy()
    for frame in (low_base,low_chosen):
        frame["contribution_eur"]*=.35; frame["lost_contribution_eur"]*=.35
        frame["net_contribution_eur"]=frame.contribution_eur-frame.holding_eur-frame.order_cost_eur
    stress.append(dict(name="Nur 35 % Stückdeckungsbeitrag (Kosten-Sensitivität, kein neues Nachfrageverhalten)",**economic_comparison(low_base,low_chosen,draws=300 if fast else 700)))
    sensitivity=[]
    for setup in (0.,1800.,3600.):
        for provider in (75.,150.,300.):
            for internal in (25.,50.,100.):
                sensitivity.append(dict(setup_eur=setup,monthly_provider_eur=provider,monthly_internal_eur=internal,
                                        annual_gain_before_adoption_eur=paired["annual_gain_eur"],
                                        first_year_net_gain_eur=paired["annual_gain_eur"]-setup-12*(provider+internal)))
    economics=dict(case="beverage",test_period=selection["test_period"],
                   policies=[summarize_inventory(base,"seasonal_minmax"),summarize_inventory(chosen,"selected_model")],
                   paired_difference=paired,stresses=stress,cost_sensitivity=sensitivity,
                   simple_comparator=dict(model=simple,policy=summarize_inventory(simple_trace,"best_simple_model"),selected_minus_simple=simple_comparison),
                   interpretation={"lost_contribution_is_separate_counterfactual":True,"lost_contribution_not_subtracted_twice":True,
                                   "inventory_released_is_capital_not_profit":True,"vat_and_deposits_excluded":True,
                                   "expediting":"Not modeled; explicit zero, no invented saving.",
                                   "same_demand_supplier_and_start_state":True,"lag_feedback":"Each policy updates lags from its own fulfilled unbooked sales and stockout flags.",
                                   "unvalidated_feedback":"Customer substitution/churn, price responses and supplier reactions are not simulated.",
                                   "baseline_safety":"Both use their own disjoint calibration residuals, joint block paths, same service target and constraints."})
    dump(output/"economic_results.json",economics)
    # Future refit is separate. August–October calibration is not in fitting;
    # the previous full-year test is now past information but never tuned on.
    selected_model=models[selected]
    if selected!="seasonal_minmax":
        selected_model=joblib.loads(joblib.dumps(selected_model)) if hasattr(joblib,"loads") else __import__("copy").deepcopy(selected_model)
        future_fit=valid&(meta.date<=FUTURE_FIT_END)
        selected_model.fit(x[future_fit],meta.loc[future_fit,"actual_observed_unbooked_sales"].to_numpy())
    future_cal=(meta.date>FUTURE_FIT_END)&(meta.date<=TODAY)&(meta.origin>=FUTURE_FIT_END)
    future_pool=error_pool(meta,future_cal,predict(selected_model,selected,x[future_cal]))
    next_rows=[]; weather_rows=[]
    for target in pd.date_range(TODAY+pd.Timedelta(days=1),TODAY+pd.Timedelta(days=7)):
        xx=world.features(n-1,target,sales,oos); pt=predict(selected_model,selected,xx)
        lo,hi,q90=future_pool.intervals(pt); known=world.known_bookings(TODAY,target)
        temp,rain=world.weather_forecast(n-1,target)
        weather_rows.append(dict(origin=TODAY,date=target,temperature_forecast=temp,rain_probability=rain))
        for key in range(16):
            next_rows.append(dict(origin=TODAY,date=target,key=key,store_id=key//8,sku_id=key%8,
                                  forecast_unbooked=float(pt[key]),lower_90=float(lo[key]) if target.weekday()!=6 else 0.,
                                  upper_90=float(hi[key]) if target.weekday()!=6 else 0.,known_bookings=int(known[key]),
                                  forecast_total=float(pt[key]+known[key]),model=selected))
    future=pd.DataFrame(next_rows); future.to_csv(output/"forecast_next_cycle.csv",index=False)
    pd.DataFrame(weather_rows).to_csv(output/"future_weather_forecasts.csv",index=False)
    joblib.dump(dict(model=selected_model,kind=selected,feature_names=FEATURES,pool=future_pool,fit_through=str(FUTURE_FIT_END.date()),forecast_origin=str(TODAY.date())),output/"model_future.joblib")
    final_stock,final_pipe=snapshots[str(TODAY.date())]
    # The actionable proposal replaces today's legacy draft, not an additional
    # order on top of the old Wednesday reorder. Earlier commitments remain.
    final_pipe=[o for o in final_pipe if o["origin"]<TODAY]
    current_onorder=np.zeros(16)
    for order in final_pipe: current_onorder[order["key"]]+=order["qty"]
    protected=protection_calendar(TODAY)
    point=np.array([predict(selected_model,selected,world.features(n-1,d,sales,oos)) for d in protected])
    target=future_pool.protection_quantile(point,n-1,open_mask=protected.weekday!=6)+np.array([world.known_bookings(TODAY,d) for d in protected]).sum(axis=0)
    position=final_stock+current_onorder; packs=world.skus.casepack.to_numpy(int)
    wanted=np.ceil(np.maximum(target-position,0)/packs).astype(int)*packs
    suggested=allocate_packs(wanted,1650-int(position.sum()),packs,position,target)
    suggested_rows=world.skus[["key","store_id","sku_id","sku","casepack"]].copy()
    suggested_rows["origin"]=TODAY; suggested_rows["on_hand"]=final_stock; suggested_rows["on_order"]=current_onorder.astype(int)
    suggested_rows["protection_end"]=protected[-1]; suggested_rows["protection_target_q90"]=target
    suggested_rows["next_review_nominal_arrival"]=add_business_days(next_review(TODAY),2)
    suggested_rows["suggested_order_units"]=suggested; suggested_rows["nominal_arrival"]=add_business_days(TODAY,2)
    suggested_rows.to_csv(output/"next_order_suggestion.csv",index=False)
    config=dict(case="beverage",fictional_name="Ilmtal Getränke – fiktiver Zweifilialbetrieb",synthetic=True,seed=SEED,run_fast=bool(fast),
                observation_start=str(START.date()),observation_end=str(TODAY.date()),forecast_origin=str(TODAY.date()),
                forecast_period=["2026-10-15","2026-10-21"],stores=2,skus_per_store=8,sale_unit="Pack/Kiste; keine Einzel-Flaschen",
                review_days="Wednesday and Saturday after sales/closing",supplier_lead="2 Monday–Friday business days plus latent 0–2 business-day delay",
                arrival_timing="Before opening/sales; contracts served first; unbooked retail demand second",capacity_units=1650,
                capacity_rule="On-hand + reserved pipeline across both stores <= capacity; complete supplier casepacks only",
                supplier_delay_probability=.09,capital_rate_annual=.08,storage_eur_per_unit_day=.0035,order_fixed_eur=10.,order_line_eur=.4,
                protection_end_semantics="Last covered sales day; next review's pre-opening nominal arrival date is exclusive",
                supplier_delay_risk="Realized delays are simulated separately; no unnamed extra nominal-demand day and no separately optimized supplier-delay buffer",
                service_quantile=.90,protection_uncertainty="Quantile of sums of joint contiguous calibration-error paths, not sum of daily quantiles",
                known_orders="Only contracts announced by each origin; later contracts are never an oracle input",
                features=FEATURES,training_label="Observed fulfilled unbooked sales; stockout/open-day training exclusion",
                next_order_state="After 2026-10-14 sales, before replacing that day's legacy draft; earlier on-order commitments retained",
                final_fit_through=str(FUTURE_FIT_END.date()),final_calibration_start="2026-08-01",adoption_assumptions=paired)
    dump(output/"config.json",config)
    checks=assertions(world,x,meta,trace,future,sales,oos,orders,selection,public); dump(output/"checks.json",checks)
    main_score=next(r for r in scores if r["model"]==selected)
    base_score=next(r for r in scores if r["model"]=="seasonal_minmax")
    summary=dict(case="beverage",selected_model=selected,best_simple_model=selection["best_simple_model"],synthetic=True,
                 test_days=len(base.date.unique()),observed_rows=len(public),test_mae=main_score["mae"],test_wape=main_score["wape"],
                 interval_coverage=main_score["coverage_90"],baseline_mae=base_score["mae"],
                 gross_simulated_year_saving_eur=paired["annual_gain_eur"],first_year_net_gain_eur=paired["first_year_net_gain_eur"],
                 gross_saving_ci_95_eur=[paired["bootstrap_ci_low_eur"],paired["bootstrap_ci_high_eur"]],
                 inventory_released_eur=paired["inventory_released_eur"],checks=checks,run_fast=bool(fast),
                 target=metrics["target"],horizon="1–8 days; errors use reference observed sales, inventory replay updates own policy sales",
                 test_period=selection["test_period"],test_tuned=False,
                 terminal_assets="Ending on-hand and unpaid on-order commitment reported separately; inventory release not profit",
                 source_files=["metrics.json","economic_results.json","model_selection.json","checks.json"])
    dump(output/"summary.json",summary)
    charts(output,meta,predictions,trace,selected)
    model_card=f"""# Getränkefall: Modellkarte und Grenzen

Fiktiver Betrieb, vollständig synthetische Daten. Keine Kundenreferenz und kein Beleg für erzielbare Einsparungen.

**Gewählt:** {selected}, ausschließlich nach Validierungs-MAE. Vergleich umfasst eine verfügbarkeitsbewusste saisonale Bestellregel, drei Ridge-Regularisierungen und zwei regulierte HistGradientBoosting-Konfigurationen. Boosting verwendet keine zufällige Early-Stopping-Teilung.

Trainieren: bis 30.06.2024. Parameterauswahl: 01.07.–31.12.2024. Danach Refit bis 31.12.2024, getrennte Fehlerkalibrierung Januar–Juni 2025. Unangetasteter Test: 01.07.2025–30.06.2026. Zieltage an Fensteranfängen werden ausgeschlossen, wenn ihr Prognoseursprung noch vor dem jeweiligen Fit-/Kalibrierungsabschluss liegt. Es gibt keine zufällige Zeilen-CV. Der ökonomische Replay umfasst das ganze Testjahr; sein Modell und Kalibrierung sind vor dem ersten neuen Auftrag verfügbar. Für den unbekannten Zukunftszyklus wird dasselbe gewählte Modell separat bis 31.07.2026 neu trainiert und mit August–14.10.2026 kalibriert, ebenfalls mit Ursprungssperre.

Die Zielgröße ist freie, noch nicht vertraglich gebuchte Nachfrage. Lernen ist ausschließlich mit beobachteten POS-Verkäufen und historischen Verfügbarkeitskennzeichen möglich. Das öffentliche stockout-Kennzeichen bedeutet beobachtbare Bestandserschöpfung: geöffneter Verkaufstag und Schlussbestand null. Es schließt exaktes Ausverkaufen ohne weitere verlorene Nachfrage ein; der Betrieb kann diese beiden Situationen aus POS/Bestand allein nicht unterscheiden. Die latente verlorene Menge wird ausschließlich für privaten Audit und ökonomische Simulation verwendet, nie für die öffentliche Maske, Lags oder Fit-/Fehlerauswahl. Solche möglicherweise zensierten Tage werden konservativ beim Fit und primären Fehlervergleich ausgeschlossen; dadurch entsteht Selektionsunsicherheit. Das Modell kann reale verdeckte Nachfrage nicht direkt lernen. Registrierte Fest-/Firmenaufträge werden nur nach ihrem tatsächlichen Bekanntwerdezeitpunkt direkt hinzugefügt. Später registrierte Aufträge bleiben eine Unsicherheit, keine oracle-Feature.

Wetterprognosen sind synthetische, origin-sichere Vorhersagen aus bis dahin beobachteter Wetterhistorie. Zukünftige Isttemperatur, latente Nachfrage und Generatormittelwerte sind nicht im Trainingsdatensatz. Die unbekannte Prognose 15.–21.10.2026 enthält keine simulierten zukünftigen Istwerte; auch die private Nachfrageerzeugung endet am 14.10.2026.

Tägliche 90%-Bänder sind empirische signierte Kalibrierungsquantile. Ihre Abdeckung im zeitlich späteren Test wird gemessen; eine verteilungsfreie oder exakte Zeitreihenabdeckung wird nicht behauptet. Für die Bestellperiode werden zusammenhängende, nach Startwochentag ausgerichtete mehrtägige Fehlervektoren über alle Artikel/Filialen gezogen, Nachfragepfade summiert und erst anschließend deren 90%-Quantil gewählt. Geschlossene Sonntage bleiben null. Das ist nicht die Summe täglicher Quantile. Füllen zensierter/fehlender Kalibrierungsfehler mit null unterschätzt möglicherweise Extremrisiko. Die Matrix mischt Vorhersagehorizonte 1–8; sie ist eine gemeinsame Pfadheuristik, keine nominelle Schutzperiodengarantie.

Ökonomischer Replay: Beide Policies beginnen mit identischem Bestand und bestehender Pipeline, erleben identische latente Nachfragemengen und dieselben potenziellen Lieferstörungen je Artikel/Bestelldatum. Sie sehen Störungen erst durch ausbleibende Ankunft. Jede Policy aktualisiert Nachfrage-Lags aus ihren eigenen erfüllten Verkäufen; es wird keine unveränderte Referenz-Verkaufshistorie durch den gesamten Test fortgeschrieben. Kundensubstitution, langfristige Abwanderung und Lieferantenreaktionen sind trotzdem unvalidierte Feedbackeffekte.

Neue Bestellungen entstehen nach Ladenschluss mittwochs/samstags. Lieferungen kommen vor Öffnung; Vorbestellungen werden zuerst bedient. Die nächste nominale Voröffnungsankunft ist die exklusive Grenze der heutigen Schutzperiode: Review Mittwoch, 14.10., nächster Review Samstag, 17.10., dessen Ankunft Dienstag, 20.10. – heutiger nominaler Bedarf reicht nur bis Montag, 19.10. Das Feld protection_end bezeichnet diesen letzten Verkaufstag; next_review_nominal_arrival weist die separate Ankunftsgrenze aus. Lieferverzögerungen werden tatsächlich simuliert, aber nicht durch einen unbenannten zusätzlichen Nominalbedarfstag kaschiert; ein gesondert optimierter Lieferverzögerungspuffer ist nicht enthalten. Bestandsposition enthält Lager und offene Bestellungen. Gemeinsame Kapazität und Lieferpacks werden tatsächlich angewendet. Lieferzeit-/Kapazitäts-/Kostenannahmen sind Beispiele und müssen im echten Betrieb ersetzt werden.

Nettonutzen = realisierter Stückdeckungsbeitrag minus Kapital-, Lager- und Bestellkosten. Aktionen senken auch den realisierten freien Verkaufspreis; registrierte Vertragsmengen verwenden den vereinbarten normalen Preis. Entgangener Deckungsbeitrag wird separat gezeigt und nicht nochmals vom realisierten Beitrag abgezogen. Freigesetzter durchschnittlicher Lagerwert ist ein geringerer Kapitalbedarf, kein Gewinn und keine nachgewiesene liquide Auszahlung. Endbestand und offene Bestellverpflichtung werden je Policy gesondert ausgewiesen. Die zusätzliche Cash-Näherung nimmt sofortige Kunden-/Lieferantenzahlung bei Erfüllung/Ankunft an und ersetzt keinen echten Erstjahres-Cashflow. Umsatzsteuer und Pfand sind ausgeschlossen. Expressbeschaffung ist nicht modelliert, daher wird keine Expressersparnis erfunden.

Das 14-Tage-Blockbootstrap hält Policy-Ergebnisse gepaart. Sein Intervall beschreibt nur die gewählte synthetische Welt, nicht unsichere Kundendaten, Modellannahmen, Saisonverschiebungen oder zukünftigen Kundenerfolg. Schlechte Prognosen, geringe Stückbeiträge, Nachfragebruch und mehr Lieferstörungen werden als feste Stressszenarien ohne Test-Nachoptimierung gezeigt.

Die Bestellvorschläge sind eine Demonstration zur Prüfung durch den Betrieb, keine Bestellautomatik. Abbruch: Exporte/Historie fehlen, Zensierung lässt sich nicht erkennen, Baseline/Standardsoftware genügt, Vorhersage verbessert keine echte Entscheidung oder Vorteil deckt Einführung und Betreuung nicht.
"""
    (output/"MODEL_CARD.md").write_text(model_card,encoding="utf-8")
    dump(output/"run_manifest.json",dict(case="beverage",selected_model=selected,outputs=[str(p.relative_to(output)) for p in sorted(output.rglob("*")) if p.is_file()],checks_passed=checks["passed"]))
    result=dict(case="beverage",selected_model=selected,metrics=metrics,economics=economics,output_dir=str(output),checks=checks)
    print(f"beverage: complete; annual operational delta={paired['annual_gain_eur']:.2f} EUR, after adoption={paired['first_year_net_gain_eur']:.2f} EUR",flush=True)
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"cases"/"beverage")
    parser.add_argument("--fast",action="store_true"); args=parser.parse_args()
    # Canonical import keeps persisted ErrorPool objects importable from notebooks.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from src.beverage import run as canonical_run
    canonical_run(args.output,args.fast)
