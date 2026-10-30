"""Separate, predeclared fresh-seed censored-demand experiment.

No hidden-demand values enter optimization, model choice or distribution calibration.
The original observed-sales demo and its negative economic result stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import bakery as base
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FRESH_SEED=2026101402
PENALTIES=[.001,.01,.1]
SIGMA_MULTIPLIERS=[.85,1.0,1.15,1.3]
HGB_PARAMS={"learning_rate":.05,"max_leaf_nodes":23,"max_iter":230,"l2_regularization":25.,"min_samples_leaf":90,"early_stopping":False,"random_state":FRESH_SEED,"categorical_features":base.CAT}


def original_result_fingerprint(path:Path)->str|None:
    if not path.exists():return None
    result=json.loads(path.read_text(encoding="utf-8"))
    keys=["selected_model","test_days","test_rows","test_stockout_fraction","test_mae","test_wape","interval_coverage","baseline_name","baseline_mae","gross_simulated_year_saving_eur","gross_saving_ci_95_eur","negative_case_net_eur"]
    return hashlib.sha256(json.dumps({k:result.get(k) for k in keys},sort_keys=True,allow_nan=False).encode("utf-8")).hexdigest()


def design_matrix(frame:pd.DataFrame)->tuple[np.ndarray,list[str]]:
    """Explicit log-demand additive/interacting design from permissible inputs only."""
    store=frame.store_code.to_numpy(int);product=frame.product_code.to_numpy(int);wd=frame.weekday.to_numpy(int)
    vals=[np.ones(len(frame))];names=["intercept"]
    for pair in range(1,24):
        vals.append((store*8+product==pair).astype(float));names.append(f"store_product_{pair}")
    for pair in range(1,21):
        vals.append((store*7+wd==pair).astype(float));names.append(f"store_weekday_{pair}")
    for field in ["doy_sin","doy_cos","trend","holiday","promo","weather_forecast"]:
        v=frame[field].to_numpy(float)
        if field=="weather_forecast":v=(v-10)/10
        for p in range(8):
            vals.append(v*(product==p));names.append(f"{field}_product_{p}")
    for s in range(3):
        vals.append(frame.vacation.to_numpy(float)*(store==s));names.append(f"vacation_store_{s}")
    for field in ["lag7","lag14","lag21","lag28","origin_mean7","origin_mean28","sameweekday_mean8"]:
        vals.append(np.log(np.maximum(frame[field].to_numpy(float),.5)));names.append("log_"+field)
    vals.append(frame.recent_stockout_share.to_numpy(float));names.append("recent_stockout_share")
    # Predeclared plausible commuter/weather interaction; no realized future weather.
    cold=np.maximum(4-frame.weather_forecast.to_numpy(float),0)/10
    for p in [2,7]:
        vals.append(cold*(store==1)*(wd<5)*(product==p));names.append(f"cold_commuter_product_{p}")
    matrix=np.column_stack(vals)
    assert np.isfinite(matrix).all()
    assert not any("latent" in n or "actual_weather" in n for n in names)
    return matrix,names


def nll_components(mu:np.ndarray,sigma:np.ndarray,sales:np.ndarray,censored:np.ndarray)->np.ndarray:
    """Continuous shifted-lognormal approximation to integer daily demand.

    Uncensored observed units get density/Jacobian; censored observations contribute
    probability D >= observed production. They are never imputed from private truth.
    """
    z=np.log(np.maximum(sales+.5,.5))
    a=(z-mu)/sigma
    density=np.log(sigma)+.5*a*a+.5*np.log(2*np.pi)+z
    survival=-log_ndtr(-a)
    return np.where(censored,survival,density)


class CensoredLognormal:
    def __init__(self,penalty:float):
        self.penalty=penalty
        self.scaler=StandardScaler()

    def transform(self,frame:pd.DataFrame,fit:bool=False)->np.ndarray:
        raw,names=design_matrix(frame)
        scaled=self.scaler.fit_transform(raw[:,1:]) if fit else self.scaler.transform(raw[:,1:])
        self.feature_names=names
        return np.column_stack([np.ones(len(frame)),scaled])

    def fit(self,frame:pd.DataFrame,maxiter:int=500):
        x=self.transform(frame,True)
        y=frame.sales_units.to_numpy(float)
        stock=frame.stockout_flag.to_numpy(bool)
        product=frame.product_code.to_numpy(int)
        z=np.log(y+.5)
        init=Ridge(alpha=10,fit_intercept=False).fit(x,z,sample_weight=np.where(stock,.2,1.))
        beta0=init.coef_.copy()
        init_res=z-x@beta0
        sigma0=np.array([max(float(np.std(init_res[(product==p)&(~stock)])),.12) for p in range(8)])
        theta0=np.r_[beta0,np.log(sigma0)]
        d=x.shape[1]
        def objective(theta):
            beta=theta[:d];logsig=theta[d:]
            sigma=np.exp(logsig[product]);mu=x@beta
            a=(z-mu)/sigma
            value=nll_components(mu,sigma,y,stock).mean()+.5*self.penalty*np.sum(beta[1:]**2)
            log_hazard=-.5*a*a-.5*np.log(2*np.pi)-log_ndtr(-a)
            hazard=np.exp(np.minimum(log_hazard,30))
            gmu=np.where(stock,-hazard/sigma,-a/sigma)
            glog=np.where(stock,-hazard*a,1-a*a)
            gbeta=x.T@gmu/len(x)
            gbeta[1:]+=self.penalty*beta[1:]
            gsigma=np.bincount(product,weights=glog,minlength=8)/len(x)
            return float(value),np.r_[gbeta,gsigma]
        result=minimize(objective,theta0,jac=True,method="L-BFGS-B",bounds=[(None,None)]*d+[(np.log(.055),np.log(.9))]*8,options={"maxiter":maxiter,"ftol":1e-10,"gtol":1e-6,"maxls":40})
        self.beta=result.x[:d]
        self.sigma=np.exp(result.x[d:])
        self.optimization={"success":bool(result.success),"message":str(result.message),"iterations":int(result.nit),"objective":float(result.fun),"max_abs_gradient":float(np.max(np.abs(result.jac))),"penalty":self.penalty,"n":len(frame)}
        if not np.isfinite(result.fun):raise RuntimeError("Nonfinite censored likelihood")
        return self

    def distribution(self,frame:pd.DataFrame,sigma_multiplier:float=1.)->tuple[np.ndarray,np.ndarray]:
        mu=self.transform(frame)@self.beta
        sigma=self.sigma[frame.product_code.to_numpy(int)]*sigma_multiplier
        return mu,sigma

    def mean(self,frame:pd.DataFrame,sigma_multiplier:float=1.)->np.ndarray:
        mu,sigma=self.distribution(frame,sigma_multiplier)
        return np.maximum(np.exp(mu+.5*sigma*sigma)-.5,0.)

    def quantile(self,frame:pd.DataFrame,q:float|np.ndarray,sigma_multiplier:float=1.)->np.ndarray:
        mu,sigma=self.distribution(frame,sigma_multiplier)
        return np.maximum(np.exp(mu+sigma*norm.ppf(q))-.5,0.)

    def likelihood(self,frame:pd.DataFrame,sigma_multiplier:float=1.)->float:
        mu,sigma=self.distribution(frame,sigma_multiplier)
        return float(nll_components(mu,sigma,frame.sales_units.to_numpy(float),frame.stockout_flag.to_numpy(bool)).mean())


def gradient_check()->dict:
    """Independent finite-difference check of the censored likelihood derivatives."""
    mu=np.array([3.,3.5,4.0,3.7]);sigma=np.array([.2,.3,.25,.18]);y=np.array([20.,35.,60.,42.]);c=np.array([False,True,False,True])
    z=np.log(y+.5);a=(z-mu)/sigma
    hazard=np.exp(-.5*a*a-.5*np.log(2*np.pi)-log_ndtr(-a))
    grad_mu=np.where(c,-hazard/sigma,-a/sigma)
    grad_lsig=np.where(c,-hazard*a,1-a*a)
    epsilon=1e-6
    num_mu=(nll_components(mu+epsilon,sigma,y,c)-nll_components(mu-epsilon,sigma,y,c))/(2*epsilon)
    num_lsig=(nll_components(mu,sigma*np.exp(epsilon),y,c)-nll_components(mu,sigma*np.exp(-epsilon),y,c))/(2*epsilon)
    assert np.max(np.abs(grad_mu-num_mu))<1e-5
    assert np.max(np.abs(grad_lsig-num_lsig))<1e-5
    # Larger mean must make a fixed censored lowerbound no less likely.
    assert (nll_components(mu+.1,sigma,y,np.ones(4,bool))<=nll_components(mu,sigma,y,np.ones(4,bool))).all()
    return {"analytic_gradient_matches_finite_difference":True,"censored_lowerbound_monotonicity":True}


def _batch_quantities(frame:pd.DataFrame,raw:np.ndarray)->np.ndarray:
    q=np.zeros(len(frame),int)
    for (_date,store),g in frame.groupby(["date","store_code"],sort=False):
        ix=g.index.to_numpy(int)
        q[ix]=base.allocate_batches(raw[ix],g.product_code.to_numpy(int),g.price.to_numpy(float),base.CAPACITY[int(store)])
    return q


def run(output_dir:Path,fast:bool=False)->dict:
    started=time.perf_counter()
    output_dir=Path(output_dir)
    for folder in ["data","audit_private","model","charts"]:(output_dir/folder).mkdir(parents=True,exist_ok=True)
    original=output_dir.parent/"summary.json"
    original_hash=hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None
    original_core_hash=original_result_fingerprint(original)
    protocol={"experiment":"Separate fresh-seed replication after detecting original fulfilled-sales vs unmet-demand mismatch","protocol_written_before_generate_and_evaluate":True,"protocol_scope":"internal design record, not an external preregistration","utc_timestamp":datetime.now(timezone.utc).isoformat(),"fresh_seed":FRESH_SEED,"original_summary_sha256":original_hash,"original_result_retained":True,"hypothesis":"Treating exhausted inventory as a lowerbound via censored likelihood may better estimate demand and cost-aware quantities; benefit is not guaranteed","target":"conditional latent demand inferred ONLY from observed uncensored/censored POS data, never hidden demand labels","distribution":"D+.5 approximately lognormal; additive/interacting log-mean and per-product residual log-sigma","sigma_bounds":[.055,.9],"independent_censoring_assumption":"production limit conditionally independent of the current unexpected demand given available calendar/weather/history covariates; approximately true in this synthetic process, unverified in a real bakery","known_design_features":"product-store, store-weekday, product calendar/trend/weather/promo, store vacation, permitted lagged POS sales and stockout fraction, cold commuter interaction","regularization_candidates":PENALTIES,"selection":"mean censored negative loglikelihood on 2024H2; no test-demand/error/economics tuning","sigma_calibration_candidates":SIGMA_MULTIPLIERS,"calibration":"choose global sigma factor solely on censored likelihood 2025H1 after fitting 2023–2024","raw_comparators":"weekday experience, Ridge alpha 10/100/500 selected on uncensored observed-sales validation MAE, fixed regularized HGB_2 from original design","raw_hgb_parameters":HGB_PARAMS,"train_end":base.TRAIN_END,"validation_end":base.VALID_END,"calibration_end":base.CALIB_END,"test":[base.TEST_START,base.TEST_END],"economic_rule":"known under/overproduction costs choose Tobit distribution critical quantile directly; same batches/capacity across all policies","evaluation":"once on a complete new synthetic year; hidden demand used only after all models/distributions/policies frozen","risk":"lognormal/specification/independent-censoring assumptions may be wrong; integer likelihood is continuous approximation; no deployment ROI proof","fast":fast}
    base._dump(output_dir/"PREDECLARED_PROTOCOL.json",protocol)
    protocol["original_numerical_test_result_fingerprint"]=original_core_hash
    base._dump(output_dir/"PREDECLARED_PROTOCOL.json",protocol)
    observed,audit,weather=base.generate_data(seed=FRESH_SEED)
    observed.to_csv(output_dir/"data/observed_sales.csv",index=False,date_format="%Y-%m-%d")
    audit.to_csv(output_dir/"audit_private/latent_demand.csv",index=False,date_format="%Y-%m-%d")
    weather.loc[weather.origin_date<=pd.Timestamp(base.LAST_OBS)].to_csv(output_dir/"data/weather_forecast_vintages.csv",index=False,date_format="%Y-%m-%d")
    x,meta=base.feature_table(observed,weather,pd.date_range("2023-03-01",base.LAST_OBS))
    checks=gradient_check()
    assert "latent_demand_units" not in meta.columns
    assert (meta.latest_history_date<=meta.origin_date).all()
    assert (meta.latest_weekday_lag_date<=meta.origin_date).all()
    train=(meta.date<=base.TRAIN_END).to_numpy();valid=((meta.date>base.TRAIN_END)&(meta.date<=base.VALID_END)).to_numpy();calib=((meta.date>base.VALID_END)&(meta.date<=base.CALIB_END)).to_numpy();test=((meta.date>=base.TEST_START)&(meta.date<=base.TEST_END)).to_numpy()
    train_meta=meta.loc[train].reset_index(drop=True);valid_meta=meta.loc[valid].reset_index(drop=True);cal_meta=meta.loc[calib].reset_index(drop=True);test_meta=meta.loc[test].reset_index(drop=True)
    choices=[]
    for penalty in PENALTIES:
        model=CensoredLognormal(penalty).fit(train_meta,250 if fast else 500)
        choices.append({"penalty":penalty,"validation_censored_nll":model.likelihood(valid_meta),**{f"fit_{k}":v for k,v in model.optimization.items()}})
    choice_df=pd.DataFrame(choices).sort_values("validation_censored_nll")
    selected_penalty=float(choice_df.iloc[0].penalty)
    choice_df["selected"]=choice_df.penalty.eq(selected_penalty)
    choice_df.to_csv(output_dir/"model/tobit_selection.csv",index=False)
    refit=meta.loc[train|valid].reset_index(drop=True)
    tobit=CensoredLognormal(selected_penalty).fit(refit,350 if fast else 600)
    factor_scores=[{"sigma_multiplier":factor,"calibration_censored_nll":tobit.likelihood(cal_meta,factor)} for factor in SIGMA_MULTIPLIERS]
    factor_df=pd.DataFrame(factor_scores).sort_values("calibration_censored_nll")
    selected_factor=float(factor_df.iloc[0].sigma_multiplier)
    factor_df["selected"]=factor_df.sigma_multiplier.eq(selected_factor)
    factor_df.to_csv(output_dir/"model/distribution_calibration.csv",index=False)
    joblib.dump(tobit,output_dir/"model/censored_lognormal_test_model.joblib")
    base._dump(output_dir/"model/censored_model_parameters.json",{"feature_names":tobit.feature_names,"coefficients":tobit.beta.tolist(),"residual_log_sigma_per_product":tobit.sigma.tolist(),"sigma_multiplier":selected_factor,"optimization":tobit.optimization})
    # Raw-sales comparators use their original public target and censored weights.
    y=meta.sales_units.to_numpy(float);weights=np.where(meta.stockout_flag,.2,1.)
    ridge_scores=[]
    for alpha in [10.,100.,500.]:
        ridge=base.make_ridge(alpha).fit(x.loc[train],y[train],ridge__sample_weight=weights[train])
        vp=np.maximum(ridge.predict(x.loc[valid]),0.)
        mask=~valid_meta.stockout_flag.to_numpy(bool)
        ridge_scores.append({"alpha":alpha,"validation_uncensored_mae":float(np.mean(np.abs(valid_meta.sales_units.to_numpy()[mask]-vp[mask])))})
    ridge_df=pd.DataFrame(ridge_scores).sort_values("validation_uncensored_mae")
    ridge_alpha=float(ridge_df.iloc[0].alpha)
    ridge_df.to_csv(output_dir/"model/ridge_selection.csv",index=False)
    ridge=base.make_ridge(ridge_alpha).fit(x.loc[train|valid],y[train|valid],ridge__sample_weight=weights[train|valid])
    hgb=HistGradientBoostingRegressor(loss="squared_error",**HGB_PARAMS).fit(x.loc[train|valid],y[train|valid],sample_weight=weights[train|valid])
    joblib.dump(ridge,output_dir/"model/raw_sales_ridge.joblib");joblib.dump(hgb,output_dir/"model/raw_sales_hgb.joblib")
    ratio=base.learn_promo_ratio(train_meta)
    predictions={};lows={};highs={};raw_calibrations={}
    for name in ["weekday_experience","raw_ridge","raw_hgb"]:
        if name=="weekday_experience":cp=base.experience_predict(cal_meta,ratio);tp=base.experience_predict(test_meta,ratio)
        else:
            model=ridge if name=="raw_ridge" else hgb
            cp=np.maximum(model.predict(x.loc[calib]),0.);tp=np.maximum(model.predict(x.loc[test]),0.)
        raw_calibrations[name]=base.calibrate(cal_meta,cp)
        predictions[name]=tp
        lows[name]=base.quantile_predict(test_meta,tp,raw_calibrations[name],.1)
        highs[name]=base.quantile_predict(test_meta,tp,raw_calibrations[name],.9)
    predictions["censored_tobit"]=tobit.mean(test_meta,selected_factor)
    lows["censored_tobit"]=tobit.quantile(test_meta,.1,selected_factor)
    highs["censored_tobit"]=tobit.quantile(test_meta,.9,selected_factor)
    # No hidden demand has been used above. This is the first audit merge.
    latent=test_meta[["date","store_code","product_code"]].merge(audit,on=["date","store_code","product_code"],validate="one_to_one").latent_demand_units.to_numpy(float)
    metrics={"fresh_seed":FRESH_SEED,"selected_model":"censored_lognormal_tobit","selected_penalty":selected_penalty,"selected_sigma_multiplier":selected_factor,"test":[base.TEST_START,base.TEST_END],"test_stockout_fraction":float(test_meta.stockout_flag.mean()),"models":{},"audit_only_warning":"latent demand errors are evaluated only after selection/calibration; they are not target labels or optimization criteria"}
    pred_file=test_meta[["date","origin_date","store","product","store_code","product_code","sales_units","stockout_flag","price"]].copy()
    for name,pred in predictions.items():
        metrics["models"][name]={"uncensored_observed_sales":base.metric_pack(test_meta.sales_units,pred,lows[name],highs[name],test_meta.stockout_flag),"synthetic_latent_demand_oracle_only":base.metric_pack(latent,pred,lows[name],highs[name])}
        pred_file[f"{name}_point"]=pred;pred_file[f"{name}_p10"]=lows[name];pred_file[f"{name}_p90"]=highs[name]
    base._dump(output_dir/"metrics.json",metrics)
    pred_file.to_csv(output_dir/"data/test_predictions.csv",index=False,date_format="%Y-%m-%d")
    pd.DataFrame({"date":test_meta.date,"latent_demand_units":latent,"censored_tobit_point":predictions["censored_tobit"]}).to_csv(output_dir/"audit_private/demand_predictions.csv",index=False,date_format="%Y-%m-%d")
    quantities={name:base.policy_quantities(test_meta,pred,raw_calibrations[name]) for name,pred in predictions.items() if name!="censored_tobit"}
    p=test_meta.product_code.to_numpy(int);price=test_meta.price.to_numpy(float)
    critical=np.maximum(price-base.COSTS[p],.01)/np.maximum(price-base.SALVAGE[p],.01)
    quantities["censored_tobit"]=_batch_quantities(test_meta,tobit.quantile(test_meta,critical,selected_factor))
    for name,q in quantities.items():
        assert (q%base.BATCH[p]==0).all()
        work=test_meta[["date","store_code"]].copy();work["work_units"]=q*base.WORK[p]
        assert all(g.work_units.sum()<=base.CAPACITY[int(s)]+1e-7 for (_d,s),g in work.groupby(["date","store_code"]))
    rows,policies=base.economics(test_meta,quantities,audit)
    rows.to_csv(output_dir/"audit_private/test_economic_replay.csv",index=False,date_format="%Y-%m-%d")
    daily=rows.groupby("date")[[c for c in rows if c.endswith("avoidable_loss_eur") or c.endswith("waste_cost_eur") or c.endswith("lost_contribution_eur")]].sum()
    daily.to_csv(output_dir/"data/day_economic_results.csv",date_format="%Y-%m-%d")
    savings={name:base.paired_week_bootstrap(rows,name,"censored_tobit",500 if fast else 2500) for name in ["weekday_experience","raw_ridge","raw_hgb"]}
    # Predeclared strong raw-sales comparator, not selected from whichever is weakest on test.
    comparator="raw_ridge"
    gross=savings[comparator]["paired_test_saving_eur"]
    sensitivities=[]
    for adoption in [.25,.5,.75,1.]:
        for monthly_fee in [0,99,199,349]:
            sensitivities.append({"adoption_fraction":adoption,"monthly_fee_eur":monthly_fee,"annual_review_eur":52*.5*22,"annual_implementation_eur":0,"simulated_gross_eur":gross*adoption,"simulated_net_eur":gross*adoption-monthly_fee*12-52*.5*22,"note":"linear implementation approximation at frozen quantities; real changes in feedback not simulated"})
    pd.DataFrame(sensitivities).to_csv(output_dir/"data/cost_adoption_sensitivity.csv",index=False)
    stress=[]
    for waste_mult in [.5,1,1.5]:
        for margin_mult in [.5,1,1.5]:
            base_loss=rows[f"{comparator}_waste_cost_eur"].sum()*waste_mult+rows[f"{comparator}_lost_contribution_eur"].sum()*margin_mult
            model_loss=rows["censored_tobit_waste_cost_eur"].sum()*waste_mult+rows["censored_tobit_lost_contribution_eur"].sum()*margin_mult
            stress.append({"waste_cost_multiplier":waste_mult,"lost_margin_multiplier":margin_mult,"saving_eur":float(base_loss-model_loss),"policy_retuned":False})
    negative_case={"scenario":"10% implementation, 349 EUR/month, 2h weekly operator review at22EUR/h, 4000EUR first-year setup; arithmetic sensitivity, not measured cost","simulated_net_eur":gross*.1-349*12-52*2*22-4000}
    econ={"fresh_seed":FRESH_SEED,"evaluation_type":"fixed-information offline replay on a fresh synthetic path; no endogenous closed-loop sales feedback","policies":policies,"saving_vs_baselines":savings,"strong_comparator_frozen_in_protocol":comparator,"gross_saving_vs_comparator_eur":gross,"frozen_quantity_cost_stress":stress,"negative_case":negative_case,"economic_identity":"same as original: surplus*(cost-salvage)+shortage*(price-cost); no revenue double-counting","distribution_assumptions":"shifted continuous lognormal, product-specific sigma and conditionally independent production censoring; these assumptions are not established for real clients","not_market_roi":True,"original_result_not_replaced":True}
    base._dump(output_dir/"economic_results.json",econ)
    # Separate current proposal, no test outcomes used to alter selected penalty/model design.
    future_model=CensoredLognormal(selected_penalty).fit(meta.reset_index(drop=True),350 if fast else 600)
    future_x,future_meta=base.feature_table(observed,weather,pd.date_range("2026-10-15",periods=7),fixed_origin=base.LAST_OBS)
    future_meta=future_meta.reset_index(drop=True)
    future_output=future_meta[["date","origin_date","store","product","store_code","product_code","lead_days","price","weather_forecast","latest_history_date","latest_weekday_lag_date"]].copy()
    future_output["forecast_units"]=future_model.mean(future_meta,selected_factor)
    future_output["p10_units"]=future_model.quantile(future_meta,.1,selected_factor)
    future_output["p90_units"]=future_model.quantile(future_meta,.9,selected_factor)
    fp=future_meta.product_code.to_numpy(int);fprice=future_meta.price.to_numpy(float)
    fq=np.maximum(fprice-base.COSTS[fp],.01)/np.maximum(fprice-base.SALVAGE[fp],.01)
    future_output["recommended_production_units"]=_batch_quantities(future_meta,future_model.quantile(future_meta,fq,selected_factor))
    future_output["status"]=np.where(future_meta.lead_days==1,"synthetic censored-demand quantity draft; human/operator validation required","fixed-Oct14 origin scenario; h>1 unvalidated")
    future_output.to_csv(output_dir/"data/future_forecast_2026-10-15_to_21.csv",index=False,date_format="%Y-%m-%d")
    future_output.loc[future_output.lead_days==1].to_csv(output_dir/"data/next_day_recommendations_2026-10-15.csv",index=False,date_format="%Y-%m-%d")
    joblib.dump(future_model,output_dir/"model/future_censored_model.joblib")
    checks.update({"hidden_truth_not_in_feature_frame":True,"all_lags_available_at_origin":True,"all_policies_equal_capacity_and_batches":True,"likelihood_only_uses_observed_sales_and_stockout_lowerbound":True,"original_summary_hash_unchanged":original_hash==hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else True,"original_numerical_test_results_unchanged":original_core_hash==original_result_fingerprint(original),"fresh_seed_differs_from_original":FRESH_SEED!=base.SEED})
    # Fixed-origin lookahead perturbation: future POS changes cannot affect predictor design.
    probe=pd.DatetimeIndex(["2025-10-02"])
    xa,ma=base.feature_table(observed,weather,probe)
    poisoned=observed.copy();poisoned.loc[poisoned.date>=probe[0],"sales_units"]=999999
    xb,mb=base.feature_table(poisoned,weather,probe)
    np.testing.assert_allclose(design_matrix(ma)[0],design_matrix(mb)[0],rtol=0,atol=0)
    checks["future_sales_poisoning_does_not_change_tobit_features"]=True
    base._dump(output_dir/"verification.json",checks)
    # Charts compare demand-aware target to audit truth explicitly after all fitting.
    fig,ax=plt.subplots(figsize=(9,4.2))
    names=list(predictions)
    vals=[metrics["models"][n]["synthetic_latent_demand_oracle_only"]["mae"] for n in names]
    ax.bar(names,vals,color=["#94A3B8","#F59E0B","#3B82F6","#0F766E"])
    ax.set_title("Getrennte neue Replikation: Nachfragefehler gegen reine Audit-Wahrheit")
    ax.set_ylabel("MAE der latenten synthetischen Nachfrage [Stück]")
    base.save_chart(fig,output_dir/"charts/demand_audit_error")
    fig,ax=plt.subplots(figsize=(9,4.2))
    waste=np.array([policies[n]["waste_cost_eur"] for n in names]);short=np.array([policies[n]["lost_contribution_eur"] for n in names])
    ax.bar(names,waste/1000,color="#F59E0B",label="Überschusskosten")
    ax.bar(names,short/1000,bottom=waste/1000,color="#0F766E",label="Fehlmengenbeitrag")
    ax.set_title("Neue Trajektorie: zensierte Likelihood vs Verkaufsschätzung")
    ax.set_ylabel("Simulierter vermeidbarer Jahresverlust [Tsd. EUR]")
    ax.legend(frameon=False)
    base.save_chart(fig,output_dir/"charts/economic_comparison")
    audit_pred=test_meta[["date","store_code","product_code","sales_units","stockout_flag"]].copy()
    audit_pred["latent_demand"]=latent;audit_pred["tobit"]=predictions["censored_tobit"];audit_pred["hgb"]=predictions["raw_hgb"];audit_pred["low"]=lows["censored_tobit"];audit_pred["high"]=highs["censored_tobit"]
    ex=audit_pred.loc[(audit_pred.store_code==1)&(audit_pred.product_code==0)&(audit_pred.date>="2026-05-01")&(audit_pred.date<="2026-06-11")]
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.fill_between(ex.date,ex.low,ex.high,color="#99F6E4",alpha=.55,label="Tobit 80-%-Intervall")
    ax.plot(ex.date,ex.latent_demand,color="#334155",lw=1.6,label="Latente Nachfrage: nur synthetisches Audit")
    ax.plot(ex.date,ex.tobit,color="#0F766E",lw=1.5,label="Zensierungsbewusste Nachfrageprognose")
    ax.plot(ex.date,ex.hgb,color="#3B82F6",lw=1.0,label="Raw-HGB-Verkaufsprognose")
    ax.scatter(ex.date,ex.sales_units,s=13,color="#F59E0B",label="Beobachtete Verkäufe")
    ax.set_title("Nachfrage ist bei Ausverkauf größer als Kassendaten")
    ax.set_ylabel("Stück, Bahnhof/Kaisersemmel")
    ax.legend(fontsize=8,frameon=False)
    base.save_chart(fig,output_dir/"charts/latent_vs_observed_forecasts")
    s={"case":"bakery_censored_replication","fresh_seed":FRESH_SEED,"original_preserved":checks["original_numerical_test_results_unchanged"],"original_summary_byte_hash_unchanged":checks["original_summary_hash_unchanged"],"selected_model":"censored_lognormal_tobit","selected_penalty":selected_penalty,"selected_sigma_multiplier":selected_factor,"test_days":int(test_meta.date.nunique()),"test_rows":len(test_meta),"stockout_fraction":metrics["test_stockout_fraction"],"audit_demand_mae":metrics["models"]["censored_tobit"]["synthetic_latent_demand_oracle_only"]["mae"],"raw_hgb_audit_demand_mae":metrics["models"]["raw_hgb"]["synthetic_latent_demand_oracle_only"]["mae"],"audit_demand_wape":metrics["models"]["censored_tobit"]["synthetic_latent_demand_oracle_only"]["wape"],"audit_interval_80_coverage":metrics["models"]["censored_tobit"]["synthetic_latent_demand_oracle_only"]["interval_80_coverage"],"gross_simulated_saving_vs_ridge_eur":gross,"gross_saving_ci_95_eur":savings[comparator]["ci_95_eur"],"gross_simulated_saving_vs_raw_hgb_eur":savings["raw_hgb"]["paired_test_saving_eur"],"negative_case_net_eur":negative_case["simulated_net_eur"],"next_day_production_units":int(future_output.loc[future_output.lead_days==1,"recommended_production_units"].sum()),"runtime_seconds":round(time.perf_counter()-started,2),"optimization":tobit.optimization,"charts":["charts/demand_audit_error.png","charts/economic_comparison.png","charts/latent_vs_observed_forecasts.png"],"simulation_only":True,"research_replication_not_original_test":True}
    base._dump(output_dir/"summary.json",s)
    base._dump(output_dir/"data_dictionary.json",{"observed_sales.csv":"Same column meanings as ../data_dictionary.json, but separate fresh-seed path. Available sales and production/stockout flags only; no latent labels.","weather_forecast_vintages.csv":"Archived synthetic weather by origin/target date; only origin information.","test_predictions.csv":{"sales_units":"observed POS units, censored lowerbound on stockout days","stockout_flag":"production exhausted, amount of lost demand unknown","censored_tobit_point":"conditional estimated demand mean from shifted lognormal censored likelihood","censored_tobit_p10/p90":"conditional demand distribution quantiles, nominal central80% interval","raw_ridge/raw_hgb_point":"observed-sales target predictions, not uncensored demand labels"},"audit_private/latent_demand.csv":"Pure hidden simulation outcome; never model label/feature/tuning/calibration input.","audit_private/test_economic_replay.csv":"All constrained quantities and same-trajectory surplus/shortage loss per row/policy.","day_economic_results.csv":"Daily loss components by policy.","cost_adoption_sensitivity.csv":"Assumed adoption/fees/review time at frozen economics, no closed-loop implementation mixture.","future_forecast_2026-10-15_to_21.csv":"Origin held fixed after Oct14 close; h>1 are scenarios with unvalidated operational horizon; no future realized outcomes.","likelihood":"For uncensored POS use shifted-lognormal density with Jacobian; for stockout use survival probability beyond exhausted production, not an imputed private demand.","model/censored_model_parameters.json":"Interpretable standardized-feature coefficients, product log-sigma, scale-calibration factor and optimization status."})
    _write_methods(output_dir,protocol,s,metrics,econ,checks)
    return s


def _write_methods(output:Path,protocol:dict,s:dict,metrics:dict,econ:dict,checks:dict):
    text=f'''# Getrennte Replikation: Nachfrage bei Ausverkauf modellieren

Dies ist eine **neue, separat gekennzeichnete synthetische Replikation** mit Seed {FRESH_SEED}. Sie ersetzt den originalen negativen Bäckereitest nicht. Der originale MAE-Auswahltest zeigte: eine bessere Prognose erfüllter Verkäufe kann wirtschaftlich schlechter sein. Rund 40 % Ausverkaufstage ließen die Kassendaten gerade dort nur eine Nachfrage-Untergrenze beobachten. Die gezielte nächste Forschungsfrage ist daher, ob ein zensierungsbewusstes Nachfragemodell besser zur Entscheidung passt.

## Vorab festgelegtes Design

`PREDECLARED_PROTOCOL.json` wird vor Datenerzeugung und Testauswertung geschrieben. Es ist ein internes Designprotokoll, keine externe Präregistrierung. Neuer Seed, Split, Likelihood, Features, Regularisierung, Verteilungskalibrierung und Kostenregel sind darin festgelegt. Die numerischen ursprünglichen Testergebnisse werden vor/nach der Replikation per SHA256-Fingerprint verglichen: unverändert={checks['original_numerical_test_results_unchanged']}. Ein zusätzlicher Bytehashvergleich der gesamten Summary ist {checks['original_summary_hash_unchanged']}; die Datei enthält auch ausführungsabhängige Laufzeitangaben. Nach dem neuen Testergebnis werden weder Modell noch Mengenzuschlag für ein schönes Resultat geändert. Die neue Modellhypothese wurde nach Diagnose des ursprünglichen Versuchs entwickelt; deshalb ist sie ausdrücklich kein nachträglich reparierter Ergebniswert desselben Heldout-Tests.

## Was tatsächlich trainiert wird

Wir nehmen an, dass `D + 0,5` bedingt auf bekannte Informationen ungefähr lognormal ist. Auf der Log-Skala enthält der Mittelwert Filial-/Produktniveau, filialabhängigen Wochentag, produktabhängige Saison/Trend/Aktion/Wettereffekte, filialabhängige Ferien, stichtagsgerechte Verkaufs-Lags und eine vorab benannte kalte Pendlerinteraktion. Pro Produkt wird eine residuale Log-Standardabweichung gelernt.

Bei nicht ausverkauftem Produkt beobachten wir die vollständige Verkäufsmenge dieses simulierten Tages und verwenden die Lognormaldichte einschließlich Jacobian. Bei Ausverkauf gilt ausschließlich `D >= verfügbare Produktionsmenge`; die Likelihood erhält dann die Überlebenswahrscheinlichkeit oberhalb dieser Grenze. Der unbekannte Betrag der Fehlmenge wird weder mit privater Wahrheit gefüllt noch als scheinbar echte Beobachtung ausgegeben. Die Produktionsmenge ist eine Zensierungsgrenze, kein zukünftiger Nachfrageinput.

Das ist eine verschobene kontinuierliche Annäherung an ganzzahlige Mengen. Die Verteilungsform ist eine Modellannahme; das Verfahren kann bei seltenen Produkten, Substitution, falschen Ausverkaufsflags oder nicht unabhängiger Zensierung falsch liegen. Voraussetzung ist näherungsweise unabhängige Produktionszensierung bedingt auf bekannte Kalender-/Wetter-/Historieninformationen. Eine Bäckerei könnte morgens aufgrund nicht dokumentierter Sonderinformationen Mengen wählen; dann wäre diese Annahme gerade verletzt.

## Auswahl ohne versteckte Nachfrage

Training endet 30.06.2024. Regularisierung wird nur über zensierte Validierungs-Negativloglikelihood im zweiten Halbjahr 2024 aus den festen Werten {PENALTIES} ausgewählt. Gewählt wurde {s['selected_penalty']}. Danach wird bis Ende 2024 refit. Ein globaler Sigma-Faktor aus {SIGMA_MULTIPLIERS} wird auf der zensierten Likelihood des ersten Halbjahrs 2025 kalibriert; gewählt wurde {s['selected_sigma_multiplier']}. Der vollständige Test vom 01.07.2025–30.06.2026 bleibt bis dahin unangetastet. Kein Auswahl-/Kalibrierungsschritt liest `audit_private/latent_demand.csv`.

Ridge und Raw-HGB trainieren weiterhin auf beobachteten Verkäufen mit Ausverkaufsgewicht 0,2; die transparenten achtwöchigen Erfahrungswerte sind ebenfalls enthalten. Ridge-Alpha wird chronologisch gewählt; HGB verwendet dieselben vorab festgelegten regularisierten Parameter wie der ursprüngliche stärkere HGB-Kandidat. Alle bekommen dieselben zulässigen Rohinformationen; Tobit konstruiert daraus eine explizite Log-Skalen-Interaktionsdarstellung. Die Unterschiede in Ziel/Likelihood/Funktionsklasse sind Teil dieser Forschung, kein isolierter Beweis, dass jede Verbesserung nur durch Zensierungskorrektur verursacht wurde.

## Neue Auswertung

{s['test_rows']} Artikel-/Filialtage in {s['test_days']} Testtagen, {s['stockout_fraction']:.1%} Ausverkaufsflags. Nach allen eingefrorenen Fits wird die private synthetische Nachfrage ausschließlich zur Prüfung der Prognosen und wirtschaftlichen Politiken aufgedeckt.

- Tobit-Nachfrage-MAE im Audit: {s['audit_demand_mae']:.3f} Stück; Raw-HGB: {s['raw_hgb_audit_demand_mae']:.3f} Stück.
- Tobit-Nachfrage-WAPE im Audit: {s['audit_demand_wape']:.2%}.
- Empirische Audit-Abdeckung des nominellen 80-%-Nachfrageintervalls: {s['audit_interval_80_coverage']:.2%}. Keine garantierte Deckung auf anderen Daten.
- Brutto-Unterschied gegenüber vorab benannter Raw-Ridge auf derselben neuen Trajektorie: {s['gross_simulated_saving_vs_ridge_eur']:,.2f} EUR; gepaartes Wochenblockbootstrap: {s['gross_saving_ci_95_eur'][0]:,.2f} bis {s['gross_saving_ci_95_eur'][1]:,.2f} EUR.
- Gegen Raw-HGB: {s['gross_simulated_saving_vs_raw_hgb_eur']:,.2f} EUR.

Diese Zahlen können positiv oder negativ sein und stehen vollständig in `metrics.json`/`economic_results.json`. Sie sind keine Beweise für reale regionale Firmen und keine Schätzung des Markt-ROI.

## Von Nachfrageverteilung zur Entscheidung

Der Mengenentwurf ist das bedingte Nachfragequantil mit kritischem Verhältnis `(Preis − vermeidbare Stückkosten) / (Preis − Restwert)`. Direktes Lognormalquantil, kein nachträglich optimierter 15-%-Zuschlag. Anschließend identische Chargen und Tagesarbeitskapazität wie alle Baselines. Überschuss kostet tatsächlich vermeidbare Herstellungskosten; Fehlmenge verliert Stückbeitrag, nicht den gesamten Umsatz plus zusätzlichen Gewinn. Restwert ist hier null. Vorhandene feste Miete/Gehälter werden nicht als neue Ersparnis gezählt.

Auch diese Auswertung ist ein **fixed-information offline replay**: alle Politiken verwenden dieselben vergangenen POS-Daten der historischen Referenzpolitik. Neue Mengen hätten ihre eigenen späteren Verkaufs-Lags erzeugt; dieser geschlossene Feedbackkreislauf wird nicht behauptet. Kapazitätsallocator und fixe Kostenparameter sind vereinfachte Geschäftsannahmen, keine Produktionsintegration. Die Sensitivität skaliert Umsetzung und Betreuungsaufwand; sie ist keine tatsächlich simulierte Mischung der Feedbacksysteme.

## Negativer Geschäftsfall und Übertragbarkeit

Im explizit angenommenen Fall mit nur 10 % Umsetzung, 349 EUR monatlicher Betreuung, zwei Stunden Prüfaufwand pro Woche zu 22 EUR und 4000 EUR Erstaufwand ist der synthetische Nettowert {s['negative_case_net_eur']:,.2f} EUR. Selbst ein brauchbarer Nachfragefit finanziert nicht automatisch Individualberatung. Kosten-/Margenstress werden bei eingefrorenen Mengen gezeigt.

Ein echter Kunde müsste belastbare Zeitstempel, Filiale/Artikel, Verkauf, Produktions-/Liefermengen, Rückläufe, Bestands-/Ausverkaufszeiten, Vorbestellungen, bekannte Aktionen und vermeidbare Kosten liefern. Stockout-Zensierung ist nicht dasselbe wie alle Formen verlorener Nachfrage. Kunden, die ein ausverkauftes Produkt ersetzen, wechseln oder den Laden verlassen, können die hier verwendete Einzelartikelverteilung verändern. Zuerst Datengüte und heutige Spezialsoftware prüfen; dann parallele Prognosen, dann begrenzter realer Vergleich unter menschlicher Mengenfreigabe.

Die künftigen Mengenentwürfe sind separat bis Ladenschluss 14.10.2026 refit. Der historische Sigma-Faktor wird nur als Näherung übertragen. H>1 im festen Sieben-Tage-Ausblick ist nicht als operative Politik validiert. Ohne Zukunftsdaten ist gerade die tatsächliche zukünftige Fehler-/Kostenleistung unbekannt.

## Prüfbarkeit

Analytische Likelihoodgradienten wurden gegen unabhängige finite Differenzen geprüft. Die zensierte Wahrscheinlichkeit steigt, wenn der angenommene Mittelwert steigt. Tests prüfen zulässige Lagstichtage, Batch-/Kapazitätsgrenzen, unveränderte Originalergebnisse und unveränderte Features nach Vergiftung zukünftiger POS-Verkäufe. Joblib-Dateien enthalten echte gelernte Modelle; `censored_model_parameters.json` legt Koeffizienten, Produkt-Sigmas und Optimierungsdiagnostik offen.

Dateien: `PREDECLARED_PROTOCOL.json`, `model/tobit_selection.csv`, `model/distribution_calibration.csv`, `metrics.json`, `economic_results.json`, `summary.json`, `verification.json`, beobachtete CSVs in `data`, reine Simulationswahrheit in `audit_private`, exportierbare PNG/SVG in `charts`. Laufzeit {s['runtime_seconds']:.1f} Sekunden.
'''
    (output/"METHODS.md").write_text(text,encoding="utf-8")
    (output/"MODEL_CARD.md").write_text(f"# Model Card: Censored Lognormal Tobit\n\nFresh-seed research replication, not replacement of original test.\nTarget: conditional demand inferred from observed POS quantities and right-censor flags.\nChosen penalty: {s['selected_penalty']}; sigma factor: {s['selected_sigma_multiplier']}.\nHidden demand never labels/features/selection/calibration.\nLimits: approximate continuous lognormal, independent censoring, feedback unvalidated, synthetic-only economics.\nDo not use for autonomous recipe, hygiene or personnel decisions.\nSee METHODS.md and PREDECLARED_PROTOCOL.json for complete assumptions.\n",encoding="utf-8")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"cases/bakery/censored_replication");parser.add_argument("--fast",action="store_true")
    args=parser.parse_args()
    # Use a stable importable class path in joblib, also when launched as a script.
    from src.bakery_censored import run as imported_run
    print(json.dumps(imported_run(args.output,args.fast),ensure_ascii=False,indent=2))
