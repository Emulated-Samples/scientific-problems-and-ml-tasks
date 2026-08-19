#!/usr/bin/env python3
"""Export every sc-pcabench rollout (solution + reasoning + per-fold subrewards)
from the S3 agent/tests logs into a local folder.

Workers are torn down after a run, so run-file/run-diff cannot serve submission bytes; the
solution is reconstructed verbatim by replaying the agent log's Write/Edit tool-call payloads.

Usage:  python scripts/export_rollouts.py            # -> ~/Downloads/pcabench_rollouts
        PCABENCH_EXPORT_DIR=/some/dir python scripts/export_rollouts.py
"""
import json, os, subprocess, sys, concurrent.futures as cf

DEST = os.path.expanduser(os.environ.get("PCABENCH_EXPORT_DIR", "~/Downloads/pcabench_rollouts"))
PROBLEM = "from_scratch_pca"

def sh(args, timeout=120):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)

def run_list():
    r = sh(["hfdev","runs","--env","sc-pcabench","--all","--json"])
    d = json.loads(r.stdout)
    runs = d if isinstance(d,list) else d.get("runs",d.get("items",[]))
    out=[]
    for x in runs:
        out.append({
            "id": x.get("id") or x.get("runId"),
            "n": int(x.get("totalRollouts") or x.get("rollouts") or 1),
            "status": x.get("status","?"),
            "model": x.get("model") or x.get("requestedModel","?"),
            "score": x.get("score"),
        })
    return out

def log_url(run, idx, cat):
    r = sh(["hfdev","run-log-url",run,str(idx),"--problem",PROBLEM,"--category",cat])
    url = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return url if url.startswith("http") else ""

def curl(url, path):
    if not url: return False
    r = sh(["curl","-sf",url,"-o",path], timeout=180)
    return r.returncode==0 and os.path.exists(path) and os.path.getsize(path)>0

def load_jsonl(path):
    recs=[]
    try:
        with open(path,encoding="utf-8",errors="replace") as f:
            for line in f:
                line=line.strip()
                if line:
                    try: recs.append(json.loads(line))
                    except: pass
    except FileNotFoundError: pass
    return recs

def reconstruct(recs):
    files={}; reasoning=[]
    for o in recs:
        t=o.get("type")
        if t=="agent_reasoning": reasoning.append("[REASONING]\n"+str(o.get("message","")))
        elif t=="agent_text": reasoning.append("[TEXT]\n"+str(o.get("message","")))
        elif t=="tool_call":
            d=o.get("data",{}) or {}; tool=d.get("tool"); inp=d.get("input",{}) or {}
            fp=inp.get("file_path") or inp.get("path")
            if tool=="Write" and fp and "content" in inp: files[fp]=inp["content"]
            elif tool=="Edit" and fp and "old_string" in inp:
                cur=files.get(fp,""); old=inp.get("old_string",""); new=inp.get("new_string","")
                files[fp]= cur.replace(old,new) if inp.get("replace_all") else cur.replace(old,new,1)
    return files, reasoning

def parse_subrewards(recs):
    folds=[]
    for o in recs:
        d=o.get("data",{}) or {}
        out=d.get("output")
        if isinstance(out,str):
            try: out=json.loads(out)
            except: out=None
        if isinstance(out,dict):
            for ds in out.get("datasets",[]) or []:
                folds.append(ds)
    return folds

def do_rollout(run, idx, rdir):
    os.makedirs(rdir, exist_ok=True)
    info={"rollout":idx,"solution_files":[],"folds":[],"status":""}
    # agent
    aurl=log_url(run,idx,"agent"); apath=os.path.join(rdir,"transcript.jsonl")
    if curl(aurl,apath):
        recs=load_jsonl(apath)
        files,reasoning=reconstruct(recs)
        soldir=os.path.join(rdir,"solution"); os.makedirs(soldir,exist_ok=True)
        for fp,content in files.items():
            if "submission" in fp or fp.endswith("pca") or fp.endswith(".py"):
                base=os.path.basename(fp) or "file"
                with open(os.path.join(soldir,base),"w",encoding="utf-8") as w: w.write(content)
                info["solution_files"].append(base)
        with open(os.path.join(rdir,"reasoning.txt"),"w",encoding="utf-8") as w:
            w.write("\n\n".join(reasoning))
    # tests
    turl=log_url(run,idx,"tests"); tpath=os.path.join(rdir,"subrewards.jsonl")
    if curl(turl,tpath):
        trecs=load_jsonl(tpath)
        folds=parse_subrewards(trecs)
        info["folds"]=[{"dataset":f.get("dataset"),"reward":f.get("reward"),
                        "accuracy":f.get("accuracy"),"time_quality":f.get("time_quality"),
                        "gate_product":f.get("gate_product")} for f in folds]
        with open(os.path.join(rdir,"subrewards_summary.json"),"w") as w:
            json.dump(info["folds"],w,indent=2)
    # setup + qa (best-effort)
    for cat in ("setup","qa"):
        u=log_url(run,idx,cat); p=os.path.join(rdir,f"{cat}.log")
        curl(u,p)
    # per-rollout summary
    with open(os.path.join(rdir,"summary.txt"),"w") as w:
        w.write(f"run={run} rollout={idx}\n")
        w.write(f"solution_files={info['solution_files']}\n\n")
        if info["folds"]:
            w.write("per-fold subrewards:\n")
            for f in sorted(info["folds"],key=lambda x:(x.get('reward') or 0)):
                w.write(f"  {f['dataset']:<26} reward={f['reward']} acc={f['accuracy']} tq={f['time_quality']} gates={f['gate_product']}\n")
    return run, idx, info

def main():
    os.makedirs(DEST, exist_ok=True)
    runs=run_list()
    tasks=[]
    for r in runs:
        rid=r["id"]; short=rid.split("-")[0]
        sc = f"{r['score']:.3f}" if isinstance(r["score"],(int,float)) else "NA"
        rundir=os.path.join(DEST, f"{short}__{r['model']}__{r['status']}__score{sc}")
        for idx in range(r["n"]):
            tasks.append((rid, idx, os.path.join(rundir, f"rollout-{idx}")))
    print(f"{len(runs)} runs, {len(tasks)} rollouts -> {DEST}")
    results=[]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(do_rollout,rid,idx,rdir):(rid,idx) for rid,idx,rdir in tasks}
        for fut in cf.as_completed(futs):
            rid,idx=futs[fut]
            try:
                run,i,info=fut.result()
                nfolds=len(info["folds"]); nsol=len(info["solution_files"])
                print(f"  {rid.split('-')[0]} r{idx}: sol={nsol} folds={nfolds}")
                results.append((run,idx,info))
            except Exception as e:
                print(f"  {rid.split('-')[0]} r{idx}: ERROR {e}")
    # top-level index
    with open(os.path.join(DEST,"README.md"),"w") as w:
        w.write("# pcabench rollouts export\n\n")
        w.write("Reconstructed from S3 agent/tests logs (workers torn down; solutions replayed from Write/Edit tool calls).\n\n")
        w.write("| run | model | status | score | rollouts |\n|---|---|---|---|---|\n")
        for r in sorted(runs,key=lambda x:-(x['score'] or 0) if isinstance(x['score'],(int,float)) else 0):
            sc=f"{r['score']:.4f}" if isinstance(r['score'],(int,float)) else "—"
            w.write(f"| {r['id'].split('-')[0]} | {r['model']} | {r['status']} | {sc} | {r['n']} |\n")
        w.write("\nEach `rollout-N/` has: `solution/` (reconstructed submission), `reasoning.txt`, `transcript.jsonl` (full agent log), `subrewards_summary.json` + `subrewards.jsonl` (per-fold), `setup.log`, `summary.txt`.\n")
    print("INDEX written:", os.path.join(DEST,"README.md"))

if __name__=="__main__": main()
