"""M06/E07/v001: fixed-train IHDP context-budget transfer diagnostic."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_m06_e05 as old  # noqa: E402

ihdp = old.old.ihdp
acic = old.old.acic
context = old.old.context
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E07/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E07/v001"
OUT = ROOT / "artifacts/local/m06_e07_results"
SOURCE_E05 = (ROOT / "projects/model_comparison/hypotheses/M06/results/E05/v001"
              / "20261020T094727_834998Z")
LOCAL_E05 = ROOT / "artifacts/local/m06_e05_results/20261020T094727_834998Z"
LOCAL_E13 = ROOT / "artifacts/local/m01_e13_results/20261019T223041_152756Z"
EXPERIMENT = "M06/E07/v001"
K_NEW, K_OLD = 64, 128
N_TRAIN, N_EVAL, REPS, FOLDS = 672, 75, 100, 5
BOOTSTRAP, BOOTSTRAP_SEED = 500, 2707
MODEL = dict(learning_rate=.05, max_iter=200, max_leaf_nodes=31,
             l2_regularization=1.0, random_state=42)
METHODS = ("fm_feature", "adjusted_feature", "local_feature",
           "global_cv", "train_ate")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return acic.sha(path)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git_state() -> dict:
    paths = [Path(__file__), PROTOCOL, Path(old.__file__),
             Path(old.old.__file__), Path(ihdp.__file__),
             Path(acic.__file__), Path(acic.source.__file__),
             Path(context.__file__)]
    relative = [p.relative_to(ROOT).as_posix() for p in paths]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *relative],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    require(not dirty, "Commit E07 runner, protocol and dependencies before inference")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(PROTOCOL),
                dependency_sha256={p.relative_to(ROOT).as_posix(): sha(p)
                                   for p in paths[2:]})


def verify_snapshot(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    require(manifest["status"] == "complete_exploratory",
            f"Source snapshot incomplete: {folder}")
    for item in manifest["files"]:
        path = folder / item["path"]
        require(path.is_file() and path.stat().st_size == item["size"] and
                sha(path) == item["sha256"],
                f"Snapshot SHA mismatch: {path}")
    local = pd.read_csv(folder / "local_files.csv")
    for row in local.itertuples(index=False):
        path = ROOT / row.path
        require(path.is_file() and path.stat().st_size == row.size and
                sha(path) == row.sha256,
                f"Local source SHA mismatch: {path}")
    return dict(path=folder.relative_to(ROOT).as_posix(),
                manifest_sha256=sha(folder / "manifest.json"),
                local_index_sha256=sha(folder / "local_files.csv"),
                code_git_commit=manifest["code_git_commit"],
                local_files=len(local))


def inputs(rep: int, train: dict, test: dict, expected: pd.DataFrame,
           verify_k64: bool = False) -> tuple:
    a, q, z, y, truth, index128, archive, scale, applicable = old.old.case_inputs(
        "IHDP", rep, (train, test))
    require(applicable and len(a) == N_TRAIN and len(q) == N_EVAL and
            index128.shape == (N_EVAL,K_OLD), f"IHDP {rep}: input shape changed")
    local_folder = LOCAL_E05 / "ihdp" / f"case_{rep:03d}"
    archive["adjusted_feature"] = np.load(local_folder / "score_adjusted_feature.npy")
    archive["global_cv"] = np.load(local_folder / "score_global_cv.npy")
    archive["train_ate"] = np.load(local_folder / "score_train_ate.npy")
    source = expected[expected.case == rep].set_index("method")
    require(set(METHODS).issubset(set(source.index)),
            f"IHDP {rep}: E05 metrics incomplete")
    for method in METHODS:
        value = archive[method]
        require(value.shape == truth.shape and np.isfinite(value).all() and
                abs(acic.pehe(truth,value)-source.loc[method,"pehe"]) < 1e-10,
                f"IHDP {rep}: E05 score/metric mismatch for {method}")
    index64 = np.concatenate((index128[:,:K_NEW//2],
                              index128[:,K_OLD//2:K_OLD//2+K_NEW//2]), axis=1)
    require(index64.shape == (N_EVAL,K_NEW) and
            np.all(z[index64[:,:K_NEW//2]] == 0) and
            np.all(z[index64[:,K_NEW//2:]] == 1),
            f"IHDP {rep}: invalid nested K64 context")
    if verify_k64:
        fresh = context.contexts_for("feature_knn",None,a,q,z,None,None,K_NEW)
        require(np.array_equal(index64,fresh),
                f"IHDP {rep}: K64 does not match fixed selector")
    direct128 = y[index128[:,K_OLD//2:]].mean(axis=1)-y[index128[:,:K_OLD//2]].mean(axis=1)
    require(np.array_equal(direct128,archive["local_feature"]),
            f"IHDP {rep}: K128 direct local differs")
    return a,q,z,y,truth,index64,archive,float(scale)


def preflight() -> tuple[dict,dict,dict,Path]:
    require(PROTOCOL.is_file(), "E07 protocol missing")
    train,test,info,weights = ihdp.preflight()
    source_e13 = verify_snapshot(old.old.SOURCES["IHDP"])
    source_e05 = verify_snapshot(SOURCE_E05)
    manifest_e05 = json.loads((SOURCE_E05 / "manifest.json").read_text())
    require(manifest_e05["experiment"] == "E05" and
            manifest_e05["experiment_version"] == "v001",
            "Wrong E05 source version")
    expected = pd.read_csv(SOURCE_E05 / "all_metrics.csv")
    require(len(expected[expected.family == "IHDP"]) == REPS*6,
            "E05 IHDP metric grid changed")
    for rep in range(REPS):
        inputs(rep,train,test,expected[expected.family == "IHDP"],verify_k64=True)
    info.update(dict(source_e13=source_e13,source_e05=source_e05,
                     new_context_size=K_NEW, archived_context_size=K_OLD,
                     bootstrap_draws=BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED,
                     model=MODEL, crossfit_folds=FOLDS))
    return train,test,info,weights


class Progress:
    def __init__(self, run: Path):
        self.run = run
        self.started = time.monotonic()
        self.last_log = 0.0
        self.queries_done = 0
        self.queries_total = REPS*N_EVAL
        self.fits_done = 0
        self.fits_total = REPS*2*FOLDS
        self.bar = tqdm(total=self.queries_total, desc="M06/E07 inference",
                        unit="query", mininterval=2, maxinterval=10,
                        dynamic_ncols=True, leave=False,
                        disable=not sys.stdout.isatty(), file=sys.stdout)

    def report(self, stage: str, rep: int | None = None,
               case_query: int | None = None, force: bool = False) -> None:
        elapsed = time.monotonic()-self.started
        eta = (elapsed/self.queries_done*(self.queries_total-self.queries_done)
               if self.queries_done else None)
        save_json(self.run / "status.json", dict(status="incomplete",
            experiment=EXPERIMENT,run_id=self.run.name,stage=stage,
            replication=rep,context_size=K_NEW,
            case_queries_done=case_query,case_queries_total=N_EVAL,
            queries_done=self.queries_done,queries_total=self.queries_total,
            fits_done=self.fits_done,fits_total=self.fits_total,
            elapsed_seconds=round(elapsed,1),
            eta_seconds=round(eta,1) if eta is not None else None,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        if sys.stdout.isatty():
            self.bar.set_description(f"M06/E07 {stage} rep{rep if rep is not None else 0:03d}",
                                     refresh=False)
        elif force or time.monotonic()-self.last_log >= 20:
            eta_text=f"{eta/60:.1f}m" if eta is not None else "unknown"
            print(f"M06/E07 {stage} rep{rep if rep is not None else 0:03d}: "
                  f"{self.queries_done}/{self.queries_total} queries, "
                  f"{self.fits_done}/{self.fits_total} fits, "
                  f"elapsed {elapsed/60:.1f}m, total ETA {eta_text}",flush=True)
            self.last_log=time.monotonic()

    def advance_fit(self, rep: int) -> None:
        self.fits_done += 1
        self.report("fitting",rep)

    def advance_query(self,count: int, rep: int, case_query: int) -> None:
        self.queries_done += count
        self.bar.update(count)
        if case_query % 20 == 0 or case_query == N_EVAL:
            self.report("inference",rep,case_query)

    def close(self) -> None:
        self.bar.close()


def adjusted_score(a: np.ndarray,q: np.ndarray,z: np.ndarray,y: np.ndarray,
                   index: np.ndarray,rep: int,progress: Progress) -> np.ndarray:
    donor_fold=np.full(N_TRAIN,-1,dtype=np.int8)
    donor_pred=np.full(N_TRAIN,np.nan)
    query_pred=np.full((2,FOLDS,N_EVAL),np.nan)
    for arm in (0,1):
        rows=np.flatnonzero(z == arm)
        splitter=KFold(n_splits=FOLDS,shuffle=True,random_state=42+arm)
        for fold,(fit_part,test_part) in enumerate(splitter.split(rows)):
            progress.report("fitting",rep)
            model=HistGradientBoostingRegressor(**MODEL)
            with threadpool_limits(limits=1):
                model.fit(a[rows[fit_part]],y[rows[fit_part]])
                donor_pred[rows[test_part]]=model.predict(a[rows[test_part]])
                query_pred[arm,fold]=model.predict(q)
            donor_fold[rows[test_part]]=fold
            progress.advance_fit(rep)
    require(np.all(donor_fold>=0) and np.isfinite(donor_pred).all() and
            np.isfinite(query_pred).all(),f"IHDP {rep}: crossfit invalid")
    means=[]
    query_rows=np.arange(N_EVAL)[:,None]
    for arm in (0,1):
        neighbors=index[:,arm*(K_NEW//2):(arm+1)*(K_NEW//2)]
        matching_query=query_pred[arm,donor_fold[neighbors],query_rows]
        means.append((y[neighbors]+matching_query-donor_pred[neighbors]).mean(axis=1))
    return means[1]-means[0]


def pehe(truth: np.ndarray,score: np.ndarray) -> float:
    require(truth.shape == score.shape and np.isfinite(score).all(),"Invalid score")
    return float(np.sqrt(np.mean((truth-score)**2)))


def regret(truth: np.ndarray,score: np.ndarray) -> float:
    count=8
    oracle=np.argsort(-truth,kind="stable")[:count]
    chosen=np.argsort(-score,kind="stable")[:count]
    return float(truth[oracle].mean()-truth[chosen].mean())


def calculate_rep(rep: int, train: dict, test: dict, expected: pd.DataFrame,
                  model, run: Path, progress: Progress) -> tuple[list[dict],dict,np.ndarray]:
    folder=run/f"rep_{rep:03d}"
    folder.mkdir(parents=True,exist_ok=False)
    a,q,z,y,truth,index64,archive,scale=inputs(rep,train,test,expected)
    adjusted64=adjusted_score(a,q,z,y,index64,rep,progress)
    local64=y[index64[:,K_NEW//2:]].mean(axis=1)-y[index64[:,:K_NEW//2]].mean(axis=1)
    forward=acic.source.make_forward(model,torch.device("cpu"),a,z,y)
    fm_path=folder/"score_fm_K64.npy"
    fm_file=np.lib.format.open_memmap(fm_path,mode="w+",dtype="float64",shape=(N_EVAL,))
    fm_file[:]=np.nan
    progress.report("inference",rep,0)
    for lo in range(0,N_EVAL,2):
        hi=min(lo+2,N_EVAL)
        fm_file[lo:hi]=forward(index64[lo:hi],q[lo:hi])
        if hi % 20 == 0 or hi == N_EVAL:
            fm_file.flush()
        progress.advance_query(hi-lo,rep,hi)
    fm_file.flush()
    fm64=np.asarray(fm_file).copy()
    require(np.isfinite(fm64).all(),f"IHDP {rep}: nonfinite FM score")
    np.save(folder/"context_K64.npy",index64)
    np.save(folder/"score_adjusted_K64.npy",adjusted64)
    np.save(folder/"score_local_K64.npy",local64)
    score_by_k={K_NEW:dict(fm_feature=fm64,adjusted_feature=adjusted64,
                           local_feature=local64,global_cv=archive["global_cv"],
                           train_ate=archive["train_ate"]),
                K_OLD:{method:archive[method] for method in METHODS}}
    rows=[]
    for k,scores in score_by_k.items():
        for method in METHODS:
            value=scores[method]
            rows.append(dict(replication=rep,context_size=k,method=method,
                pehe=pehe(truth,value),normalized_pehe=pehe(truth,value)/scale,
                regret_at_10=regret(truth,value),scale=scale))
    pd.DataFrame(rows).to_csv(folder/"metrics.csv",index=False)
    d={k:(pehe(truth,scores["adjusted_feature"])-
           pehe(truth,scores["fm_feature"]))/scale
       for k,scores in score_by_k.items()}
    pair=dict(replication=rep,scale=scale,d_k64=d[K_NEW],d_k128=d[K_OLD],
        delta_k=d[K_OLD]-d[K_NEW],
        regret_fm_minus_adjusted_k64=(regret(truth,fm64)-regret(truth,adjusted64)),
        regret_fm_minus_adjusted_k128=(regret(truth,archive["fm_feature"])-
                                       regret(truth,archive["adjusted_feature"])))
    rng=np.random.default_rng(BOOTSTRAP_SEED+rep)
    ix=rng.integers(0,N_EVAL,size=(BOOTSTRAP,N_EVAL))
    db={}
    for k,scores in score_by_k.items():
        adj=truth[ix]-scores["adjusted_feature"][ix]
        fm=truth[ix]-scores["fm_feature"][ix]
        db[k]=(np.sqrt(np.mean(adj**2,axis=1))-
               np.sqrt(np.mean(fm**2,axis=1)))/scale
    draw=db[K_OLD]-db[K_NEW]
    np.save(folder/"bootstrap_delta_k.npy",draw)
    pair["conditional_low"],pair["conditional_high"]=map(
        float,np.quantile(draw,[.025,.975]))
    pd.DataFrame([pair]).to_csv(folder/"paired.csv",index=False)
    save_json(folder/"status.json",dict(status="complete_exploratory",
              replication=rep,context_sizes=[K_NEW,K_OLD]))
    return rows,pair,draw


def evaluate(run: Path,rows: list[dict],pairs: list[dict],
             draws: dict[int,np.ndarray],progress: Progress) -> dict:
    progress.report("evaluation",force=True)
    metrics=pd.DataFrame(rows)
    paired=pd.DataFrame(pairs).sort_values("replication")
    require(len(metrics)==REPS*2*len(METHODS) and len(paired)==REPS and
            len(draws)==REPS and progress.queries_done==progress.queries_total and
            progress.fits_done==progress.fits_total,"Incomplete E07 result grid")
    metrics.to_csv(run/"all_metrics.csv",index=False)
    paired.to_csv(run/"paired_deltas.csv",index=False)
    values=paired.delta_k.to_numpy(dtype=float)
    pooled=np.mean(np.stack([draws[int(rep)] for rep in paired.replication]),axis=0)
    low,high=np.quantile(pooled,[.025,.975])
    mean_d128=float(paired.d_k128.mean())
    criterion=bool(values.mean()>.05 and (values>0).sum()>=70 and mean_d128>.05)
    compact=metrics.groupby(["context_size","method"])[
        ["normalized_pehe","regret_at_10"]].mean().reset_index()
    summary=dict(status="complete_exploratory",
        mean_delta_k=float(values.mean()),median_delta_k=float(np.median(values)),
        positive_replications=int((values>0).sum()),
        mean_d_k64=float(paired.d_k64.mean()),mean_d_k128=mean_d128,
        conditional_bootstrap_ci=[float(low),float(high)],
        diagnostic_gate_met=criterion,cell_means=compact.to_dict(orient="records"),
        interpretation="Viewed IHDP family; 100 replications share children; paired within-rep bootstrap only",
        label_access="Adjusted uses all train labels; FM sees K labels per query")
    source_summary=json.loads((SOURCE_E05/"summary.json").read_text())
    require(np.isclose(summary["mean_d_k128"],
            source_summary["families"]["IHDP"]["mean_delta"],rtol=0,atol=1e-12),
            "K128 mean D did not reproduce E05")
    save_json(run/"summary.json",summary)
    return summary


def package(run: Path,info: dict,code: dict,summary: dict) -> Path:
    target=TARGET/run.name
    require(not target.exists(),"Result snapshot already exists")
    staging=ROOT/"artifacts/local/m06_e07_staging"/run.name
    staging.mkdir(parents=True,exist_ok=False)
    for name in ("all_metrics.csv","paired_deltas.csv","summary.json"):
        shutil.copy2(run/name,staging/name)
    for name,folder in (("e13",old.old.SOURCES["IHDP"]),("e05",SOURCE_E05)):
        shutil.copy2(folder/"local_files.csv",staging/f"source_{name}_files.csv")
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,
        sha256=sha(p)) for p in sorted(run.rglob("*")) if p.is_file()
        and p.name!="status.json"]).to_csv(staging/"local_files.csv",index=False)
    save_json(staging/"study_protocol.json",dict(experiment=EXPERIMENT,
        run_id=run.name,local_path=run.relative_to(ROOT).as_posix(),source=info,
        model=MODEL,new_context_size=K_NEW,archived_context_size=K_OLD,
        bootstrap_draws=BOOTSTRAP,bootstrap_seed=BOOTSTRAP_SEED,
        python=sys.version.split()[0],
        package_versions={name:importlib.metadata.version(name)
            for name in ("numpy","pandas","scikit-learn","torch","tqdm")},**code))
    code_folder=staging/"code"
    code_folder.mkdir()
    for source in (Path(__file__),Path(old.__file__),Path(old.old.__file__),
                   Path(ihdp.__file__),Path(acic.__file__),
                   Path(acic.source.__file__),Path(context.__file__)):
        shutil.copy2(source,code_folder/source.name)
    (staging/"README.md").write_text(
        f"# M06 / E07 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E07/v001.md). "
        "Новый K64 FM-score на всех 100 IHDP-репликациях; K128 из архива.\n\n"
        f"Primary Δ_K=D128−D64: {summary['mean_delta_k']:+.4f}, "
        f"{summary['positive_replications']}/100 положительных; "
        f"conditional 95% CI [{summary['conditional_bootstrap_ci'][0]:+.4f}; "
        f"{summary['conditional_bootstrap_ci'][1]:+.4f}]. "
        f"Ориентир: {'выполнен' if summary['diagnostic_gate_met'] else 'не выполнен'}.\n\n"
        "[Метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
        "[сводка](summary.json), [полный протокол/SHA](study_protocol.json), "
        "[локальные файлы](local_files.csv). Репликации разделяют детей и "
        "их outcome уже просмотрены.\n")
    files=[dict(path=p.relative_to(staging).as_posix(),size=p.stat().st_size,
                sha256=sha(p)) for p in sorted(staging.rglob("*")) if p.is_file()
           and p.name not in ("README.md","manifest.json")]
    save_json(staging/"manifest.json",dict(schema_version=1,
        project="model_comparison",hypothesis="M06",experiment="E07",
        experiment_version="v001",run_id=run.name,result_revision=1,
        status="complete_exploratory",summary="IHDP K64 versus K128 FM gain diagnostic",
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
        print("M06/E07: no runs")
        return
    for run in sorted(OUT.iterdir()):
        path=run/"status.json"
        if run.is_dir() and path.is_file():
            state=json.loads(path.read_text())
            print(run.name,state.get("status"),state.get("stage"),
                  f"rep={state.get('replication')}",
                  f"{state.get('queries_done',0)}/{state.get('queries_total',0)} queries",
                  f"{state.get('fits_done',0)}/{state.get('fits_total',0)} fits",
                  "ETA",state.get("eta_seconds"))


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("all",),default="all")
    parser.add_argument("--check",action="store_true",
                        help="Verify IHDP source SHA and all K64/K128 pairings without fitting")
    parser.add_argument("--status",action="store_true",
                        help="Read local progress without inference")
    args=parser.parse_args()
    if args.status:
        status()
        return 0
    train,test,info,weights=preflight()
    print("PREFLIGHT PASS: 100 IHDP replications, archived K128 and nested K64",
          flush=True)
    if args.check:
        return 0
    code=git_state()
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run=OUT/run_id
    run.mkdir(parents=True,exist_ok=False)
    save_json(run/"protocol.json",dict(experiment=EXPERIMENT,run_id=run_id,
                                       source=info,**code))
    progress=Progress(run)
    progress.report("model_load",force=True)
    try:
        model=acic.load_model(weights)
        expected=pd.read_csv(SOURCE_E05/"all_metrics.csv")
        expected=expected[expected.family=="IHDP"]
        metrics,pairs,draws=[],[],{}
        for rep in range(REPS):
            rows,pair,draw=calculate_rep(rep,train,test,expected,model,run,progress)
            metrics.extend(rows)
            pairs.append(pair)
            draws[rep]=draw
        progress.close()
        summary=evaluate(run,metrics,pairs,draws,progress)
        progress.report("packaging",force=True)
        package(run,info,code,summary)
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
