"""M06/E08/v001: prospective context-budget test on new synthetic DGPs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as source  # noqa: E402

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E08/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E08/v001"
OUT = ROOT / "artifacts/local/m06_e08_results"
EXPERIMENT = "M06/E08/v001"
N_TRAIN, N_EVAL, FEATURES = 672, 512, 12
K_SMALL, K_LARGE = 64, 128
FOLDS = 5
BOOTSTRAP, BOOTSTRAP_SEED = 5000, 2808
FAMILIES = ("smooth_gate", "threshold_gate", "constant_control")
SEEDS = {"smooth_gate": tuple(range(3101,3109)),
         "threshold_gate": tuple(range(3101,3109)),
         "constant_control": (3191,3192)}
MODEL = dict(learning_rate=.05,max_iter=200,max_leaf_nodes=31,
             l2_regularization=1.0,random_state=42)
METHODS = ("fm_feature","adjusted_feature","local_feature",
           "global_cv","train_ate")


def require(ok: bool,message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return source.sha256(path)


def save_json(path: Path,value: dict) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def git_state() -> dict:
    paths=(Path(__file__),PROTOCOL,Path(source.__file__))
    relative=[path.relative_to(ROOT).as_posix() for path in paths]
    dirty=subprocess.run(["git","status","--porcelain","--",*relative],
                         cwd=ROOT,check=True,capture_output=True,
                         text=True).stdout.strip()
    require(not dirty,"Commit E08 runner, protocol and source before inference")
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                          capture_output=True,text=True).stdout.strip()
    return dict(code_git_commit=commit,code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(PROTOCOL),
                dependency_sha256={Path(source.__file__).relative_to(ROOT).as_posix():
                                   sha(Path(source.__file__))})


def cases():
    for family in FAMILIES:
        for seed in SEEDS[family]:
            yield family,seed


def covariates(rng: np.random.Generator,n: int) -> np.ndarray:
    latent=rng.normal(size=(n,FEATURES))
    x=latent.copy()
    x[:,1]=.6*latent[:,0]+.8*latent[:,1]
    x[:,3]=.5*latent[:,2]+np.sqrt(.75)*latent[:,3]
    x[:,8:]=(latent[:,8:]>0).astype(np.float64)
    return x.astype(np.float32)


def make_base(seed: int) -> dict[str,np.ndarray]:
    rng=np.random.default_rng(seed)
    x_train=covariates(rng,N_TRAIN)
    x_eval=covariates(rng,N_EVAL)
    def propensity(x):
        logit=.4*x[:,0]-.3*x[:,2]+.2*(x[:,8]-.5)+.2*x[:,3]*x[:,4]
        return np.clip(1/(1+np.exp(-logit)),.2,.8)
    p_train,p_eval=propensity(x_train),propensity(x_eval)
    t_train=rng.binomial(1,p_train).astype(np.int8)
    t_eval=rng.binomial(1,p_eval).astype(np.int8)
    eps0_train=rng.normal(0,.5,N_TRAIN)
    eps1_train=rng.normal(0,.5,N_TRAIN)
    eps0_eval=rng.normal(0,.5,N_EVAL)
    eps1_eval=rng.normal(0,.5,N_EVAL)
    base=dict(x_train=x_train,x_eval=x_eval,p_train=p_train,p_eval=p_eval,
        t_train=t_train,t_eval=t_eval,eps0_train=eps0_train,
        eps1_train=eps1_train,eps0_eval=eps0_eval,eps1_eval=eps1_eval)
    require(np.bincount(t_train,minlength=2).min()>=K_LARGE//2,
            f"Seed {seed}: fewer than {K_LARGE//2} train rows in an arm")
    require(all(np.isfinite(value).all() for value in base.values()),
            f"Seed {seed}: nonfinite base data")
    train_hash=pd.util.hash_pandas_object(pd.DataFrame(x_train),index=False).to_numpy()
    eval_hash=pd.util.hash_pandas_object(pd.DataFrame(x_eval),index=False).to_numpy()
    require(not np.intersect1d(train_hash,eval_hash).size,
            f"Seed {seed}: duplicate train/eval covariates")
    return base


def potential_means(x: np.ndarray,family: str) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if family=="smooth_gate":
        mu0=.7*np.tanh(x[:,0]+x[:,1])+.4*x[:,2]+.2*x[:,8]
        tau=(.2+.7*np.exp(-(.8*x[:,0]-x[:,1])**2)
             +.35*np.tanh(x[:,3]+x[:,4])-.35*x[:,9])
    elif family=="threshold_gate":
        mu0=.5*(x[:,0]>0)+.35*(x[:,2]+x[:,3]>0)+.15*x[:,4]
        tau=(.6*((x[:,0]>0)&(x[:,1]<0))-.45*(x[:,2]>0)
             +.25*x[:,9]+.15)
    elif family=="constant_control":
        mu0=.5*np.tanh(x[:,0])+.3*x[:,2]
        tau=np.full(len(x),.25,dtype=np.float64)
    else:
        raise ValueError(f"Unknown family: {family}")
    mu0=np.asarray(mu0,dtype=np.float64)
    tau=np.asarray(tau,dtype=np.float64)
    return mu0,mu0+tau,tau


def case_data(family: str,seed: int) -> dict[str,np.ndarray]:
    base=make_base(seed)
    mu0_train,mu1_train,tau_train=potential_means(base["x_train"],family)
    mu0_eval,mu1_eval,tau_eval=potential_means(base["x_eval"],family)
    t=base["t_train"]
    factual_train=np.where(t==1,mu1_train+base["eps1_train"],
                           mu0_train+base["eps0_train"])
    base.update(mu0_train=mu0_train,mu1_train=mu1_train,tau_train=tau_train,
        mu0_eval=mu0_eval,mu1_eval=mu1_eval,tau_eval=tau_eval,
        factual_train=factual_train)
    require(all(np.isfinite(value).all() for value in base.values()),
            f"{family}/{seed}: nonfinite case")
    sd=float(np.std(tau_eval))
    require((sd>1e-8)==(family!="constant_control"),
            f"{family}/{seed}: CATE type changed")
    return base


def contexts(x_train: np.ndarray,x_eval: np.ndarray,
             t_train: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    scaler=StandardScaler().fit(x_train)
    a=scaler.transform(x_train).astype(np.float32)
    q=scaler.transform(x_eval).astype(np.float32)
    arms=[np.flatnonzero(t_train==arm) for arm in (0,1)]
    selected=[arm[NearestNeighbors(n_neighbors=K_LARGE//2,n_jobs=1)
                  .fit(a[arm]).kneighbors(q,return_distance=False)]
              for arm in arms]
    large=np.concatenate(selected,axis=1).astype(np.int32)
    small=np.concatenate((large[:,:K_SMALL//2],
                          large[:,K_LARGE//2:K_LARGE//2+K_SMALL//2]),axis=1)
    fresh=[arm[NearestNeighbors(n_neighbors=K_SMALL//2,n_jobs=1)
                .fit(a[arm]).kneighbors(q,return_distance=False)]
           for arm in arms]
    require(np.array_equal(small,np.concatenate(fresh,axis=1)),
            "Nested K64 context differs from separate kNN")
    for k,index in ((K_SMALL,small),(K_LARGE,large)):
        require(index.shape==(N_EVAL,k) and
                np.all(t_train[index[:,:k//2]]==0) and
                np.all(t_train[index[:,k//2:]]==1) and
                np.all(np.diff(np.sort(index,axis=1),axis=1)>0),
                f"Invalid K{k} context")
    return a,q,small,large


def preflight() -> dict:
    require(PROTOCOL.is_file(),"E08 protocol missing")
    try:
        weights=Path(hf_hub_download("vdblm/causalpfn","causalpfn_v0.pt",
            revision=source.REVISION,local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint absent from local cache") from exc
    counts={}
    for family,seed in cases():
        data=case_data(family,seed)
        contexts(data["x_train"],data["x_eval"],data["t_train"])
        counts[f"{family}/{seed}"]=np.bincount(data["t_train"],minlength=2).tolist()
    for seed in SEEDS["smooth_gate"]:
        a,b=case_data("smooth_gate",seed),case_data("threshold_gate",seed)
        for key in ("x_train","x_eval","t_train","t_eval",
                    "eps0_train","eps1_train","eps0_eval","eps1_eval"):
            require(np.array_equal(a[key],b[key]),
                    f"Paired families differ in base {key}, seed {seed}")
    return dict(checkpoint_revision=source.REVISION,
        checkpoint_sha256=sha(weights),checkpoint_path=str(weights),
        train_rows=N_TRAIN,eval_rows=N_EVAL,features=FEATURES,
        context_sizes=[K_SMALL,K_LARGE],families={k:list(v) for k,v in SEEDS.items()},
        train_treatment_counts=counts,model=MODEL,crossfit_folds=FOLDS,
        bootstrap_draws=BOOTSTRAP,bootstrap_seed=BOOTSTRAP_SEED)


class Progress:
    def __init__(self,run: Path):
        self.run=run
        self.started=time.monotonic()
        self.last_log=0.0
        self.queries_done=0
        self.queries_total=sum(len(v) for v in SEEDS.values())*2*N_EVAL
        self.fits_done=0
        self.fits_total=sum(len(v) for v in SEEDS.values())*2*FOLDS
        self.bar=tqdm(total=self.queries_total,desc="M06/E08 inference",
            unit="query",mininterval=2,maxinterval=10,dynamic_ncols=True,
            leave=False,disable=not sys.stdout.isatty(),file=sys.stdout)

    def report(self,stage: str,family: str | None=None,seed: int | None=None,
               k: int | None=None,case_query: int | None=None,force: bool=False) -> None:
        elapsed=time.monotonic()-self.started
        eta=(elapsed/self.queries_done*(self.queries_total-self.queries_done)
             if self.queries_done else None)
        save_json(self.run/"status.json",dict(status="incomplete",
            experiment=EXPERIMENT,run_id=self.run.name,stage=stage,
            family=family,seed=seed,context_size=k,
            case_queries_done=case_query,case_queries_total=N_EVAL,
            queries_done=self.queries_done,queries_total=self.queries_total,
            fits_done=self.fits_done,fits_total=self.fits_total,
            elapsed_seconds=round(elapsed,1),
            eta_seconds=round(eta,1) if eta is not None else None,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        if sys.stdout.isatty():
            self.bar.set_description(f"M06/E08 {stage} {family or ''} s{seed or 0} K{k or 0}",
                                     refresh=False)
        elif force or time.monotonic()-self.last_log>=20:
            eta_text=f"{eta/60:.1f}m" if eta is not None else "unknown"
            print(f"M06/E08 {stage} {family or ''} s{seed or 0} K{k or 0}: "
                  f"{self.queries_done}/{self.queries_total} queries, "
                  f"{self.fits_done}/{self.fits_total} fits, "
                  f"elapsed {elapsed/60:.1f}m, ETA {eta_text}",flush=True)
            self.last_log=time.monotonic()

    def advance_fit(self,family: str,seed: int) -> None:
        self.fits_done+=1
        self.report("fitting",family,seed)

    def advance_query(self,count: int,family: str,seed: int,k: int,hi: int) -> None:
        self.queries_done+=count
        self.bar.update(count)
        if hi%64==0 or hi==N_EVAL:
            self.report("inference",family,seed,k,hi)

    def close(self) -> None:
        self.bar.close()


def fit_nuisance(a: np.ndarray,q: np.ndarray,t: np.ndarray,y: np.ndarray,
                 family: str,seed: int,progress: Progress) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    donor_fold=np.full(N_TRAIN,-1,dtype=np.int8)
    donor_pred=np.full(N_TRAIN,np.nan)
    query_pred=np.full((2,FOLDS,N_EVAL),np.nan)
    for arm in (0,1):
        rows=np.flatnonzero(t==arm)
        splitter=KFold(n_splits=FOLDS,shuffle=True,random_state=42+arm)
        for fold,(fit_part,test_part) in enumerate(splitter.split(rows)):
            progress.report("fitting",family,seed)
            model=HistGradientBoostingRegressor(**MODEL)
            with threadpool_limits(limits=1):
                model.fit(a[rows[fit_part]],y[rows[fit_part]])
                donor_pred[rows[test_part]]=model.predict(a[rows[test_part]])
                query_pred[arm,fold]=model.predict(q)
            donor_fold[rows[test_part]]=fold
            progress.advance_fit(family,seed)
    require(np.all(donor_fold>=0) and np.isfinite(donor_pred).all() and
            np.isfinite(query_pred).all(),f"{family}/{seed}: invalid nuisance")
    return donor_fold,donor_pred,query_pred


def local_adjusted(index: np.ndarray,y: np.ndarray,
                   donor_fold: np.ndarray,donor_pred: np.ndarray,
                   query_pred: np.ndarray,k: int) -> tuple[np.ndarray,np.ndarray]:
    rows=np.arange(N_EVAL)[:,None]
    means,direct=[],[]
    for arm in (0,1):
        neighbors=index[:,arm*(k//2):(arm+1)*(k//2)]
        matching_query=query_pred[arm,donor_fold[neighbors],rows]
        means.append((y[neighbors]+matching_query-donor_pred[neighbors]).mean(axis=1))
        direct.append(y[neighbors].mean(axis=1))
    return means[1]-means[0],direct[1]-direct[0]


def pehe(truth: np.ndarray,score: np.ndarray) -> float:
    require(truth.shape==score.shape and np.isfinite(score).all(),"Invalid score")
    return float(np.sqrt(np.mean((truth-score)**2)))


def regret(truth: np.ndarray,score: np.ndarray) -> float:
    count=math.ceil(.1*N_EVAL)
    oracle=np.argsort(-truth,kind="stable")[:count]
    chosen=np.argsort(-score,kind="stable")[:count]
    return float(truth[oracle].mean()-truth[chosen].mean())


def calculate_case(family: str,seed: int,model,run: Path,
                   progress: Progress) -> tuple[list[dict],dict]:
    folder=run/f"{family}_{seed}"
    folder.mkdir(parents=True,exist_ok=False)
    base=case_data(family,seed)
    np.savez_compressed(folder/"base.npz",**base)
    progress.report("context",family,seed)
    a,q,small,large=contexts(base["x_train"],base["x_eval"],base["t_train"])
    t,y,truth=base["t_train"],base["factual_train"],base["tau_eval"]
    donor_fold,donor_pred,query_pred=fit_nuisance(a,q,t,y,family,seed,progress)
    np.save(folder/"donor_fold.npy",donor_fold)
    global_cv=query_pred[1].mean(axis=0)-query_pred[0].mean(axis=0)
    train_ate=np.full(N_EVAL,y[t==1].mean()-y[t==0].mean())
    np.save(folder/"score_global_cv.npy",global_cv)
    np.save(folder/"score_train_ate.npy",train_ate)
    forward=source.make_forward(model,torch.device("cpu"),a,t,y)
    score_by_k={}
    rows=[]
    scale=float(np.std(truth))
    for k,index in ((K_SMALL,small),(K_LARGE,large)):
        np.save(folder/f"context_K{k}.npy",index)
        adjusted,direct=local_adjusted(index,y,donor_fold,donor_pred,query_pred,k)
        path=folder/f"score_fm_K{k}.npy"
        fm_file=np.lib.format.open_memmap(path,mode="w+",dtype="float64",shape=(N_EVAL,))
        fm_file[:]=np.nan
        progress.report("inference",family,seed,k,0)
        for lo in range(0,N_EVAL,2):
            hi=min(lo+2,N_EVAL)
            fm_file[lo:hi]=forward(index[lo:hi],q[lo:hi])
            if hi%64==0 or hi==N_EVAL:
                fm_file.flush()
            progress.advance_query(hi-lo,family,seed,k,hi)
        fm_file.flush()
        fm=np.asarray(fm_file).copy()
        require(np.isfinite(fm).all(),f"{family}/{seed}/K{k}: nonfinite FM")
        np.save(folder/f"score_adjusted_K{k}.npy",adjusted)
        np.save(folder/f"score_local_K{k}.npy",direct)
        scores=dict(fm_feature=fm,adjusted_feature=adjusted,local_feature=direct,
                    global_cv=global_cv,train_ate=train_ate)
        score_by_k[k]=scores
        for method in METHODS:
            err=pehe(truth,scores[method])
            rows.append(dict(family=family,seed=seed,context_size=k,method=method,
                pehe=err,normalized_pehe=(err/scale if family!="constant_control" else np.nan),
                regret_at_10=regret(truth,scores[method]),scale=scale,
                included_in_primary=family!="constant_control"))
    pd.DataFrame(rows).to_csv(folder/"metrics.csv",index=False)
    pair=dict(family=family,seed=seed,scale=scale,
              included_in_primary=family!="constant_control")
    if family!="constant_control":
        d={k:(pehe(truth,scores["adjusted_feature"])-
              pehe(truth,scores["fm_feature"]))/scale
           for k,scores in score_by_k.items()}
        pair.update(d_k64=d[K_SMALL],d_k128=d[K_LARGE],
                    delta_k=d[K_LARGE]-d[K_SMALL])
    else:
        pair.update(d_k64=np.nan,d_k128=np.nan,delta_k=np.nan)
    pair["regret_fm_minus_adjusted_k64"]=(
        regret(truth,score_by_k[K_SMALL]["fm_feature"])-
        regret(truth,score_by_k[K_SMALL]["adjusted_feature"]))
    pair["regret_fm_minus_adjusted_k128"]=(
        regret(truth,score_by_k[K_LARGE]["fm_feature"])-
        regret(truth,score_by_k[K_LARGE]["adjusted_feature"]))
    pd.DataFrame([pair]).to_csv(folder/"paired.csv",index=False)
    save_json(folder/"status.json",dict(status="complete_exploratory",
              family=family,seed=seed,context_sizes=[K_SMALL,K_LARGE]))
    return rows,pair


def evaluate(run: Path,rows: list[dict],pairs: list[dict],progress: Progress) -> dict:
    progress.report("evaluation",force=True)
    metrics=pd.DataFrame(rows)
    paired=pd.DataFrame(pairs)
    total_cases=sum(len(v) for v in SEEDS.values())
    require(len(metrics)==total_cases*2*len(METHODS) and
            len(paired)==total_cases and progress.queries_done==progress.queries_total and
            progress.fits_done==progress.fits_total,"Incomplete E08 result grid")
    metrics.to_csv(run/"all_metrics.csv",index=False)
    paired.to_csv(run/"paired_deltas.csv",index=False)
    summary=dict(status="complete_exploratory",prospective_generator=True,
        primary_families={},constant_control_cases=len(SEEDS["constant_control"]),
        label_access="Adjusted uses all train labels; FM sees K labels per query")
    for family in FAMILIES[:2]:
        subset=paired[paired.family==family].sort_values("seed")
        values=subset.delta_k.to_numpy(dtype=float)
        require(len(values)==8 and np.isfinite(values).all(),
                f"{family}: incomplete primary cases")
        rng=np.random.default_rng(BOOTSTRAP_SEED+(0 if family=="smooth_gate" else 1))
        draws=values[rng.integers(0,len(values),size=(BOOTSTRAP,len(values)))].mean(axis=1)
        np.save(run/f"bootstrap_seed_mean_{family}.npy",draws)
        low,high=np.quantile(draws,[.025,.975])
        sub_metrics=metrics[metrics.family==family]
        compact=sub_metrics.groupby(["context_size","method"])[
            ["normalized_pehe","regret_at_10"]].mean().reset_index()
        summary["primary_families"][family]=dict(mean_delta_k=float(values.mean()),
            median_delta_k=float(np.median(values)),
            positive_seeds=int((values>0).sum()),
            mean_d_k64=float(subset.d_k64.mean()),
            mean_d_k128=float(subset.d_k128.mean()),
            seed_bootstrap_ci=[float(low),float(high)],
            cell_means=compact.to_dict(orient="records"))
    summary["diagnostic_gate_met"]=bool(all(
        entry["mean_delta_k"]>.05 and entry["positive_seeds"]>=6
        for entry in summary["primary_families"].values()))
    controls=metrics[metrics.family=="constant_control"]
    summary["constant_control_pehe"]={
        f"K{int(k)}/{method}":float(value)
        for (k,method),value in controls.groupby(["context_size","method"]).pehe.mean().items()}
    summary["interpretation"]=("Eight independent base draws paired across two fixed synthetic mechanisms; "
        "generator selected after viewed ACIC/IHDP, no external RCT confirmation")
    save_json(run/"summary.json",summary)
    return summary


def package(run: Path,pre: dict,code: dict,summary: dict) -> Path:
    target=TARGET/run.name
    require(not target.exists(),"Result snapshot already exists")
    staging=ROOT/"artifacts/local/m06_e08_staging"/run.name
    staging.mkdir(parents=True,exist_ok=False)
    for name in ("all_metrics.csv","paired_deltas.csv","summary.json"):
        shutil.copy2(run/name,staging/name)
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,
        sha256=sha(p)) for p in sorted(run.rglob("*")) if p.is_file()
        and p.name!="status.json"]).to_csv(staging/"local_files.csv",index=False)
    save_json(staging/"study_protocol.json",dict(experiment=EXPERIMENT,
        run_id=run.name,local_path=run.relative_to(ROOT).as_posix(),source=pre,
        model=MODEL,context_sizes=[K_SMALL,K_LARGE],
        families={k:list(v) for k,v in SEEDS.items()},
        bootstrap_draws=BOOTSTRAP,bootstrap_seed=BOOTSTRAP_SEED,
        python=sys.version.split()[0],
        package_versions={name:importlib.metadata.version(name)
            for name in ("numpy","pandas","scikit-learn","torch","tqdm")},**code))
    code_folder=staging/"code"
    code_folder.mkdir()
    for path in (Path(__file__),Path(source.__file__)):
        shutil.copy2(path,code_folder/path.name)
    a,b=(summary["primary_families"][name] for name in FAMILIES[:2])
    (staging/"README.md").write_text(
        f"# M06 / E08 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E08/v001.md). "
        "Новые draws двух заранее заданных synthetic-DGP и constant-CATE контроль.\n\n"
        f"Primary Δ_K=D128−D64: smooth {a['mean_delta_k']:+.4f} "
        f"({a['positive_seeds']}/8), threshold {b['mean_delta_k']:+.4f} "
        f"({b['positive_seeds']}/8). Совместный ориентир: "
        f"{'выполнен' if summary['diagnostic_gate_met'] else 'не выполнен'}.\n\n"
        "[Метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
        "[сводка](summary.json), [протокол/SHA](study_protocol.json), "
        "[локальные файлы](local_files.csv). Не является внешним RCT.\n")
    files=[dict(path=p.relative_to(staging).as_posix(),size=p.stat().st_size,
                sha256=sha(p)) for p in sorted(staging.rglob("*")) if p.is_file()
           and p.name not in ("README.md","manifest.json")]
    save_json(staging/"manifest.json",dict(schema_version=1,
        project="model_comparison",hypothesis="M06",experiment="E08",
        experiment_version="v001",run_id=run.name,result_revision=1,
        status="complete_exploratory",summary="Prospective K-budget test on new DGPs",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],code_sha256=code["code_sha256"],
        code_snapshot=(target/"code"/Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    TARGET.mkdir(parents=True,exist_ok=True)
    staging.replace(target)
    print("SNAPSHOT COMPLETE:",target.resolve(),flush=True)
    return target


def status() -> None:
    if not OUT.is_dir():
        print("M06/E08: no runs")
        return
    for run in sorted(OUT.iterdir()):
        path=run/"status.json"
        if run.is_dir() and path.is_file():
            value=json.loads(path.read_text())
            print(run.name,value.get("status"),value.get("stage"),
                  value.get("family"),value.get("seed"),
                  f"{value.get('queries_done',0)}/{value.get('queries_total',0)} queries",
                  f"{value.get('fits_done',0)}/{value.get('fits_total',0)} fits",
                  "ETA",value.get("eta_seconds"))


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("all",),default="all")
    parser.add_argument("--check",action="store_true",
                        help="Verify DGPs, arm counts, nested contexts and checkpoint without scoring")
    parser.add_argument("--status",action="store_true",
                        help="Read local progress without inference")
    args=parser.parse_args()
    if args.status:
        status()
        return 0
    pre=preflight()
    print("PREFLIGHT PASS: 8 smooth, 8 threshold, 2 constant cases; "
          "N=672, K=64/128, 512 eval rows",flush=True)
    if args.check:
        return 0
    code=git_state()
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run=OUT/run_id
    run.mkdir(parents=True,exist_ok=False)
    save_json(run/"protocol.json",dict(experiment=EXPERIMENT,run_id=run_id,
                                       source=pre,**code))
    progress=Progress(run)
    progress.report("model_load",force=True)
    try:
        torch.set_num_threads(4)
        model=CATEEstimator(device="cpu",model_path=pre["checkpoint_path"],
                            calibrate=False,verbose=False)
        model.load_model()
        model.icl_model.eval()
        rows,pairs=[],[]
        for family,seed in cases():
            new_rows,pair=calculate_case(family,seed,model,run,progress)
            rows.extend(new_rows)
            pairs.append(pair)
        progress.close()
        summary=evaluate(run,rows,pairs,progress)
        progress.report("packaging",force=True)
        package(run,pre,code,summary)
        save_json(run/"status.json",dict(status="complete_exploratory",
            experiment=EXPERIMENT,run_id=run_id,stage="done",
            queries_done=progress.queries_done,queries_total=progress.queries_total,
            fits_done=progress.fits_done,fits_total=progress.fits_total,
            elapsed_seconds=round(time.monotonic()-progress.started,1),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print("RUN COMPLETE:",run.resolve(),flush=True)
    except Exception:
        save_json(run/"status.json",dict(status="incomplete",experiment=EXPERIMENT,
            run_id=run_id,stage="failed",queries_done=progress.queries_done,
            queries_total=progress.queries_total,fits_done=progress.fits_done,
            fits_total=progress.fits_total,error=traceback.format_exc(),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        progress.close()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
