#!/usr/bin/env python3
import json, sys, glob, os, statistics as st
d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__) + "/../results"
for f in sorted(glob.glob(d + "/*.json")):
    r = json.load(open(f))
    g = [x["derived"] for x in r["decode"] if x["temperature"] == 0.0]
    s = [x["derived"] for x in r["decode"] if x["temperature"] == 1.0]
    m = lambda rows, k: round(st.mean([x[k] for x in rows if x.get(k)]), 2) if any(x.get(k) for x in rows) else None
    per = {x["domain"]: round(x["derived"]["server_decode_tps"], 1) for x in r["decode"] if x["temperature"] == 0.0}
    pf = {x["target"]: round(x["derived"]["server_prompt_tps"]) for x in r["prefill"]}
    print(f"{r['arm']:<22} greedy {m(g,'server_decode_tps')} (client {m(g,'client_decode_tps')}) sampled {m(s,'server_decode_tps')} "
          f"acc_len {m(g,'accept_len')}/{m(s,'accept_len')} {per} prefill {pf} done={r.get('done')}")
