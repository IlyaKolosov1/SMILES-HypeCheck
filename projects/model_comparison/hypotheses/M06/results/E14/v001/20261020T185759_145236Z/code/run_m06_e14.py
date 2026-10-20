"""M06/E14/v001: external binary generator test of comparative K response."""

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as source  # noqa: E402
import download_zhao_uplift as fetch  # noqa: E402

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E14/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E14/v001"
OUT = ROOT / "artifacts/local/m06_e14_results"
DATA = fetch.DEST
EXPERIMENT = "M06/E14/v001"
N_TRAIN, N_EVAL, FEATURES = 672, 512, 36
K_SMALL, K_LARGE = 64, 128
FOLDS = 5
BOOTSTRAP, BOOTSTRAP_SEED = 5000, 2814
TRIALS = tuple(range(10))
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
    paths=(Path(__file__),PROTOCOL,Path(source.__file__),Path(fetch.__file__))
    relative=[path.relative_to(ROOT).as_posix() for path in paths]
    dirty=subprocess.run(["git","status","--porcelain","--",*relative],
                         cwd=ROOT,check=True,capture_output=True,
                         text=True).stdout.strip()
    require(not dirty,"Commit E14 runner, protocol and dependencies before inference")
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                          capture_output=True,text=True).stdout.strip()
    return dict(code_git_commit=commit,code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(PROTOCOL),
                dependency_sha256={path.relative_to(ROOT).as_posix():sha(path)
                    for path in paths[2:]})


FEATURE_COLUMNS = tuple(
    f"x{i}_{'informative' if i <= 10 else 'irrelevant' if i <= 30 else 'uplift_increase'}"
    for i in range(1, FEATURES + 1)
)
REQUIRED_COLUMNS = ("trial_id", "treatment_group_key", "conversion",
    "control_conversion_prob", "treatment1_conversion_prob", "treatment1_true_effect",
    *FEATURE_COLUMNS)


def load_cases() -> tuple[dict[int,dict[str,np.ndarray]],dict]:
    require(DATA.is_file() and DATA.stat().st_size==fetch.SIZE,
            "Published CSV absent or size mismatch; run tools/download_zhao_uplift.py")
    md5,source_sha256=fetch.digest(DATA)
    require(md5==fetch.MD5,"Published CSV MD5 mismatch")
    print("M06/E14 data audit: source checksum verified; scanning trials",flush=True)
    counts={trial:[0,0] for trial in range(50)}
    for chunk in pd.read_csv(DATA,usecols=["trial_id","treatment_group_key"],chunksize=100_000):
        require(chunk.trial_id.notna().all(),"Missing trial ID")
        for (trial,arm),size in chunk.groupby(["trial_id","treatment_group_key"]).size().items():
            require(int(trial) in counts and arm in ("control","treatment1"),
                    f"Unexpected trial/arm: {trial}/{arm}")
            counts[int(trial)][int(arm=="treatment1")]+=int(size)
    require(all(value==[10_000,10_000] for value in counts.values()),
            "Published trial/arm grid differs from 50 x 10,000 per arm")
    selected={trial:[] for trial in TRIALS}
    source_row=0
    for chunk in pd.read_csv(DATA,chunksize=50_000):
        require(set(REQUIRED_COLUMNS).issubset(chunk.columns),"Missing published fields")
        require(chunk[list(REQUIRED_COLUMNS)].notna().all().all(),"Missing source value")
        require(chunk.conversion.isin((0,1)).all(),"Nonbinary conversion")
        prob=chunk[["control_conversion_prob","treatment1_conversion_prob"]].to_numpy()
        truth=chunk.treatment1_true_effect.to_numpy()
        require(np.isfinite(prob).all() and np.isfinite(truth).all()
                and ((prob>=0)&(prob<=1)).all(),"Invalid potential-outcome probabilities")
        require(np.allclose(prob[:,1]-prob[:,0],truth,atol=1e-10,rtol=1e-10),
                "Published CATE differs from potential-outcome probabilities")
        require(np.isfinite(chunk[list(FEATURE_COLUMNS)].to_numpy()).all(),
                "Nonfinite feature")
        chunk.insert(0,"source_row",np.arange(source_row,source_row+len(chunk)))
        source_row+=len(chunk)
        for trial in TRIALS:
            section=chunk.loc[chunk.trial_id==trial]
            if len(section):
                selected[trial].append(section)
    require(source_row==1_000_000,"Unexpected source row count")
    cases={}
    split_counts={}
    for trial in TRIALS:
        frame=pd.concat(selected[trial],ignore_index=True)
        require(len(frame)==20_000,f"Trial {trial}: incomplete source")
        order=np.random.default_rng(27014+trial).permutation(len(frame))
        train=frame.iloc[order[:N_TRAIN]]
        evaluation=frame.iloc[order[N_TRAIN:N_TRAIN+N_EVAL]]
        def arm(df):
            return (df.treatment_group_key.to_numpy()=="treatment1").astype(np.int8)
        t_train=arm(train)
        y_train=train.conversion.to_numpy(dtype=np.float64)
        split_counts[str(trial)]=np.bincount(t_train,minlength=2).tolist()
        require(min(split_counts[str(trial)])>=K_LARGE//2,
                f"Trial {trial}: too few train rows in an arm")
        for treatment in (0,1):
            arm_rows=np.flatnonzero(t_train==treatment)
            for fit,_ in KFold(n_splits=FOLDS,shuffle=True,
                               random_state=42+treatment).split(arm_rows):
                require(np.unique(y_train[arm_rows[fit]]).size==2,
                        f"Trial {trial}: single-class nuisance fit")
        base=dict(x_train=train[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32),
            x_eval=evaluation[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32),
            t_train=t_train,y_train=y_train,
            tau_eval=evaluation.treatment1_true_effect.to_numpy(dtype=np.float64),
            p0_eval=evaluation.control_conversion_prob.to_numpy(dtype=np.float64),
            p1_eval=evaluation.treatment1_conversion_prob.to_numpy(dtype=np.float64),
            source_row_train=train.source_row.to_numpy(dtype=np.int64),
            source_row_eval=evaluation.source_row.to_numpy(dtype=np.int64))
        require(not np.intersect1d(base["source_row_train"],base["source_row_eval"]).size,
                f"Trial {trial}: train/eval overlap")
        require(np.std(base["tau_eval"])>0,f"Trial {trial}: constant eval CATE")
        cases[trial]=base
    return cases,dict(source_url=fetch.URL,source_sha256=source_sha256,
        source_md5=md5,source_size=fetch.SIZE,all_trial_arm_counts=counts,
        selected_train_arm_counts=split_counts,trial_ids=list(TRIALS),
        train_rows=N_TRAIN,eval_rows=N_EVAL,features=FEATURES,
        split_seed_rule="27014 + trial_id",feature_columns=list(FEATURE_COLUMNS))


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


def preflight() -> tuple[dict[int,dict[str,np.ndarray]],dict]:
    require(PROTOCOL.is_file(),"E14 protocol missing")
    try:
        weights=Path(hf_hub_download("vdblm/causalpfn","causalpfn_v0.pt",
            revision=source.REVISION,local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint absent from local cache") from exc
    cases,metadata=load_cases()
    for trial,data in cases.items():
        contexts(data["x_train"],data["x_eval"],data["t_train"])
        print(f"M06/E14 preflight trial {trial}: split and contexts verified",flush=True)
    metadata.update(checkpoint_revision=source.REVISION,
        checkpoint_sha256=sha(weights),checkpoint_path=str(weights),
        context_sizes=[K_SMALL,K_LARGE],model=MODEL,crossfit_folds=FOLDS,
        bootstrap_draws=BOOTSTRAP,bootstrap_seed=BOOTSTRAP_SEED)
    return cases,metadata


class Progress:
    def __init__(self,run: Path):
        self.run=run
        self.started=time.monotonic()
        self.last_log=0.0
        self.queries_done=0
        self.queries_total=len(TRIALS)*2*N_EVAL
        self.fits_done=0
        self.fits_total=len(TRIALS)*2*FOLDS
        self.bar=tqdm(total=self.queries_total,desc="M06/E14 inference",
            unit="query",mininterval=2,maxinterval=10,dynamic_ncols=True,
            leave=False,disable=not sys.stdout.isatty(),file=sys.stdout)

    def report(self,stage: str,trial: int | None=None,
               k: int | None=None,case_query: int | None=None,force: bool=False) -> None:
        elapsed=time.monotonic()-self.started
        eta=(elapsed/self.queries_done*(self.queries_total-self.queries_done)
             if self.queries_done else None)
        save_json(self.run/"status.json",dict(status="incomplete",
            experiment=EXPERIMENT,run_id=self.run.name,stage=stage,
            trial=trial,context_size=k,
            case_queries_done=case_query,case_queries_total=N_EVAL,
            queries_done=self.queries_done,queries_total=self.queries_total,
            fits_done=self.fits_done,fits_total=self.fits_total,
            elapsed_seconds=round(elapsed,1),
            eta_seconds=round(eta,1) if eta is not None else None,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        if sys.stdout.isatty():
            self.bar.set_description(f"M06/E14 {stage} trial {trial} K{k or 0}",
                                     refresh=False)
        elif force or time.monotonic()-self.last_log>=20:
            eta_text=f"{eta/60:.1f}m" if eta is not None else "unknown"
            print(f"M06/E14 {stage} trial {trial} K{k or 0}: "
                  f"{self.queries_done}/{self.queries_total} queries, "
                  f"{self.fits_done}/{self.fits_total} fits, "
                  f"elapsed {elapsed/60:.1f}m, ETA {eta_text}",flush=True)
            self.last_log=time.monotonic()

    def advance_fit(self,trial: int) -> None:
        self.fits_done+=1
        self.report("fitting",trial)

    def advance_query(self,count: int,trial: int,k: int,hi: int) -> None:
        self.queries_done+=count
        self.bar.update(count)
        if hi%64==0 or hi==N_EVAL:
            self.report("inference",trial,k,hi)

    def close(self) -> None:
        self.bar.close()


def fit_nuisance(a: np.ndarray,q: np.ndarray,t: np.ndarray,y: np.ndarray,
                 trial: int,progress: Progress) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    donor_fold=np.full(N_TRAIN,-1,dtype=np.int8)
    donor_pred=np.full(N_TRAIN,np.nan)
    query_pred=np.full((2,FOLDS,N_EVAL),np.nan)
    for arm in (0,1):
        rows=np.flatnonzero(t==arm)
        splitter=KFold(n_splits=FOLDS,shuffle=True,random_state=42+arm)
        for fold,(fit_part,test_part) in enumerate(splitter.split(rows)):
            progress.report("fitting",trial)
            model=HistGradientBoostingClassifier(**MODEL)
            with threadpool_limits(limits=1):
                model.fit(a[rows[fit_part]],y[rows[fit_part]])
                donor_pred[rows[test_part]]=model.predict_proba(a[rows[test_part]])[:,1]
                query_pred[arm,fold]=model.predict_proba(q)[:,1]
            donor_fold[rows[test_part]]=fold
            progress.advance_fit(trial)
    require(np.all(donor_fold>=0) and np.isfinite(donor_pred).all() and
            np.isfinite(query_pred).all(),f"Trial {trial}: invalid nuisance")
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


def calculate_case(trial: int,base: dict[str,np.ndarray],model,run: Path,
                   progress: Progress) -> tuple[list[dict],dict]:
    folder=run/f"trial_{trial:02d}"
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/"base.npz",**base)
    progress.report("context",trial)
    a,q,small,large=contexts(base["x_train"],base["x_eval"],base["t_train"])
    t,y,truth=base["t_train"],base["y_train"],base["tau_eval"]
    donor_fold,donor_pred,query_pred=fit_nuisance(a,q,t,y,trial,progress)
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
        progress.report("inference",trial,k,0)
        for lo in range(0,N_EVAL,2):
            hi=min(lo+2,N_EVAL)
            fm_file[lo:hi]=forward(index[lo:hi],q[lo:hi])
            if hi%64==0 or hi==N_EVAL:
                fm_file.flush()
            progress.advance_query(hi-lo,trial,k,hi)
        fm_file.flush()
        fm=np.asarray(fm_file).copy()
        require(np.isfinite(fm).all(),f"Trial {trial}/K{k}: nonfinite FM")
        np.save(folder/f"score_adjusted_K{k}.npy",adjusted)
        np.save(folder/f"score_local_K{k}.npy",direct)
        scores=dict(fm_feature=fm,adjusted_feature=adjusted,local_feature=direct,
                    global_cv=global_cv,train_ate=train_ate)
        score_by_k[k]=scores
        for method in METHODS:
            err=pehe(truth,scores[method])
            rows.append(dict(trial=trial,context_size=k,method=method,
                pehe=err,normalized_pehe=err/scale,
                regret_at_10=regret(truth,scores[method]),scale=scale))
    pd.DataFrame(rows).to_csv(folder/"metrics.csv",index=False)
    d={k:(pehe(truth,scores["adjusted_feature"])-
          pehe(truth,scores["fm_feature"]))/scale
       for k,scores in score_by_k.items()}
    pair=dict(trial=trial,scale=scale,d_k64=d[K_SMALL],d_k128=d[K_LARGE],
              delta_k=d[K_LARGE]-d[K_SMALL])
    pair["regret_fm_minus_adjusted_k64"]=(
        regret(truth,score_by_k[K_SMALL]["fm_feature"])-
        regret(truth,score_by_k[K_SMALL]["adjusted_feature"]))
    pair["regret_fm_minus_adjusted_k128"]=(
        regret(truth,score_by_k[K_LARGE]["fm_feature"])-
        regret(truth,score_by_k[K_LARGE]["adjusted_feature"]))
    pair["regret_fm_minus_global_k128"]=(
        regret(truth,score_by_k[K_LARGE]["fm_feature"])-
        regret(truth,score_by_k[K_LARGE]["global_cv"]))
    pd.DataFrame([pair]).to_csv(folder/"paired.csv",index=False)
    save_json(folder/"status.json",dict(status="complete_exploratory",
              trial=trial,context_sizes=[K_SMALL,K_LARGE]))
    return rows,pair


def evaluate(run: Path,rows: list[dict],pairs: list[dict],progress: Progress) -> dict:
    progress.report("evaluation",force=True)
    metrics=pd.DataFrame(rows)
    paired=pd.DataFrame(pairs)
    require(len(metrics)==len(TRIALS)*2*len(METHODS) and
            len(paired)==len(TRIALS) and progress.queries_done==progress.queries_total and
            progress.fits_done==progress.fits_total,"Incomplete E14 result grid")
    metrics.to_csv(run/"all_metrics.csv",index=False)
    paired.to_csv(run/"paired_deltas.csv",index=False)
    values=paired.sort_values("trial").delta_k.to_numpy(dtype=float)
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    draws=values[rng.integers(0,len(values),size=(BOOTSTRAP,len(values)))].mean(axis=1)
    np.save(run/"bootstrap_trial_mean.npy",draws)
    low,high=np.quantile(draws,[.025,.975])
    means=metrics.groupby(["context_size","method"])[
        ["normalized_pehe","regret_at_10","pehe"]].mean().reset_index()
    positive=int((values>0).sum())
    pehe_gate=float(values.mean())>.05 and positive>=8
    regret_rows=paired[["regret_fm_minus_adjusted_k128",
                        "regret_fm_minus_global_k128"]]
    regret_positive=int((regret_rows["regret_fm_minus_adjusted_k128"]<0).sum())
    global_positive=int((regret_rows["regret_fm_minus_global_k128"]<0).sum())
    policy_gate=regret_positive>=7 and global_positive>=7
    summary=dict(status="complete_exploratory",external_published_generator=True,
        trials=len(TRIALS),primary=dict(mean_delta_k=float(values.mean()),
            median_delta_k=float(np.median(values)),positive_trials=positive,
            mean_d_k64=float(paired.d_k64.mean()),mean_d_k128=float(paired.d_k128.mean()),
            trial_bootstrap_ci=[float(low),float(high)],
            descriptive_gate_met=bool(pehe_gate)),
        policy=dict(regret_fm_below_adjusted_trials=regret_positive,
            regret_fm_below_global_trials=global_positive,
            descriptive_gate_met=bool(policy_gate)),
        cell_means=means.to_dict(orient="records"),
        label_access="Adjusted and global models use all 672 train labels; FM sees K labels per query",
        interpretation="Ten trials from one published synthetic generator; no external field RCT")
    save_json(run/"summary.json",summary)
    return summary


def package(run: Path,pre: dict,code: dict,summary: dict) -> Path:
    target=TARGET/run.name
    require(not target.exists(),"Result snapshot already exists")
    staging=ROOT/"artifacts/local/m06_e14_staging"/run.name
    staging.mkdir(parents=True,exist_ok=False)
    for name in ("all_metrics.csv","paired_deltas.csv","summary.json"):
        shutil.copy2(run/name,staging/name)
    shutil.copy2(PROTOCOL,staging/"protocol.md")
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,
        sha256=sha(p)) for p in sorted(run.rglob("*")) if p.is_file()
        and p.name!="status.json"]).to_csv(staging/"local_files.csv",index=False)
    save_json(staging/"study_protocol.json",dict(experiment=EXPERIMENT,
        run_id=run.name,local_path=run.relative_to(ROOT).as_posix(),source=pre,
        model=MODEL,context_sizes=[K_SMALL,K_LARGE],trial_ids=list(TRIALS),
        bootstrap_draws=BOOTSTRAP,bootstrap_seed=BOOTSTRAP_SEED,
        python=sys.version.split()[0],
        package_versions={name:importlib.metadata.version(name)
            for name in ("numpy","pandas","scikit-learn","torch","tqdm")},**code))
    code_folder=staging/"code"
    code_folder.mkdir()
    for path in (Path(__file__),Path(source.__file__),Path(fetch.__file__)):
        shutil.copy2(path,code_folder/path.name)
    primary=summary["primary"]
    policy=summary["policy"]
    (staging/"README.md").write_text(
        f"# M06 / E14 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E14/v001.md). "
        "Десять trial опубликованного бинарного synthetic uplift-бенчмарка.\n\n"
        f"Primary ΔK: {primary['mean_delta_k']:+.4f} "
        f"({primary['positive_trials']}/10; условный bootstrap CI "
        f"[{primary['trial_bootstrap_ci'][0]:+.4f}; {primary['trial_bootstrap_ci'][1]:+.4f}]). "
        f"PEHE gate: {'пройден' if primary['descriptive_gate_met'] else 'не пройден'}.\n\n"
        f"FM K128 regret лучше adjusted/global в "
        f"{policy['regret_fm_below_adjusted_trials']}/10 и "
        f"{policy['regret_fm_below_global_trials']}/10 trial; policy gate: "
        f"{'пройден' if policy['descriptive_gate_met'] else 'не пройден'}.\n\n"
        "[Метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
        "[сводка](summary.json), [протокол/SHA](study_protocol.json), "
        "[локальные файлы](local_files.csv). Не является полевым RCT.\n")
    files=[dict(path=p.relative_to(staging).as_posix(),size=p.stat().st_size,
                sha256=sha(p)) for p in sorted(staging.rglob("*")) if p.is_file()
           and p.name not in ("README.md","manifest.json")]
    save_json(staging/"manifest.json",dict(schema_version=1,
        project="model_comparison",hypothesis="M06",experiment="E14",
        experiment_version="v001",run_id=run.name,result_revision=1,
        status="complete_exploratory",summary="Published binary uplift generator K transfer test",
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
        print("M06/E14: no runs")
        return
    for run in sorted(OUT.iterdir()):
        path=run/"status.json"
        if run.is_dir() and path.is_file():
            value=json.loads(path.read_text())
            print(run.name,value.get("status"),value.get("stage"),
                  f"trial {value.get('trial')}",
                  f"{value.get('queries_done',0)}/{value.get('queries_total',0)} queries",
                  f"{value.get('fits_done',0)}/{value.get('fits_total',0)} fits",
                  "ETA",value.get("eta_seconds"))


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("all",),default="all")
    parser.add_argument("--check",action="store_true",
                        help="Verify source CSV, fixed splits, nested contexts and checkpoint without scoring")
    parser.add_argument("--status",action="store_true",
                        help="Read local progress without inference")
    args=parser.parse_args()
    if args.status:
        status()
        return 0
    cases,pre=preflight()
    print("PREFLIGHT PASS: 10 published trials; N=672, K=64/128, 512 eval rows",flush=True)
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
        for trial,data in cases.items():
            new_rows,pair=calculate_case(trial,data,model,run,progress)
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
