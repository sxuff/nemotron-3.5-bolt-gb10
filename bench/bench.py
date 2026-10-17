#!/usr/bin/env python3
"""Nemotron 3.5 Bolt single-GB10 bench against llama-server. stdlib only.
Same protocol as the Super recipe:
decode: 4 tasks x 512 forced output tokens (ignore_eos), single stream, greedy + one sampled (T=1.0, top_p=0.95).
prefill: cold (unique nonce at prompt start, cache_prompt=false) real-text prompts, max_tokens=1.
Primary numbers come from llama-server's own per-request `timings` (predicted_per_second, prompt_per_second,
draft_n, draft_n_accepted); client-side SSE timings are kept as a cross-check.
"""
import argparse, json, os, time, urllib.request, uuid, glob

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8090")

DECODE = {
    "code": "Write a complete, well-commented Python module implementing an LRU cache with TTL expiry, thread safety, and a decorator interface. Include unit tests using unittest.",
    "prose": "Write a long, vivid short story about a lighthouse keeper on a remote island who discovers that the light is signalling to something beneath the sea.",
    "math": "Solve step by step: a tank is filled by pipe A in 6 hours and pipe B in 9 hours, and drained by pipe C in 12 hours. Starting empty with all three open, after 2 hours pipe C is closed. How long in total until the tank is full? Then generalize the method.",
    "summary": None,
}

def post(path, body, timeout=7200):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def stream_chat(messages, max_tokens, temperature=0.0, top_p=1.0, ignore_eos=True, thinking=True, cache_prompt=True):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos, "seed": 1234,
            "cache_prompt": cache_prompt, "timings_per_token": False,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if temperature == 0.0:
        body["top_k"] = 1
    t0 = time.perf_counter(); first = None; last = None; events = 0; usage = None; timings = None
    text = []; reasoning = []; finish = None
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            if j.get("timings"): timings = j["timings"]
            for c in j.get("choices", []):
                delta = c.get("delta", {})
                if delta.get("content"): text.append(delta["content"])
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc: reasoning.append(rc)
                if delta.get("content") or rc:
                    now = time.perf_counter(); events += 1
                    if first is None: first = now
                    last = now
                if c.get("finish_reason"): finish = c["finish_reason"]
    t1 = time.perf_counter()
    return {"t_start": t0, "t_first": first, "t_last": last, "t_end": t1, "events": events, "usage": usage,
            "timings": timings, "finish": finish, "content": "".join(text), "reasoning": "".join(reasoning)}

def source_text():
    root = os.environ.get("CORPUS_DIR", os.path.expanduser("~/llama.cpp-b9222"))
    files = sorted(glob.glob(root + "/src/*.cpp")) + sorted(glob.glob(root + "/docs/**/*.md", recursive=True))
    buf = []
    for f in files:
        try: buf.append(open(f, errors="ignore").read())
        except Exception: pass
    return "\n\n".join(buf)

def tokenize(text):
    with post("/tokenize", {"content": text}) as r:
        return len(json.loads(r.read())["tokens"])

def build_prompt(src, target_tokens, nonce):
    lo, hi = 1000, min(len(src), target_tokens * 8)
    best = lo
    for _ in range(16):
        mid = (lo + hi) // 2
        n = tokenize(nonce + src[:mid])
        if n <= target_tokens: best = mid; lo = mid + 1
        else: hi = mid - 1
    return nonce + src[:best]

def summarize(r):
    t = r.get("timings") or {}
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    pn = t.get("predicted_n")
    out = {"server_decode_tps": t.get("predicted_per_second"), "server_prompt_tps": t.get("prompt_per_second"),
           "prompt_n": t.get("prompt_n"), "predicted_n": pn, "draft_n": dn, "draft_accepted": da}
    if dn: out["draft_accept_rate"] = round(da / dn, 4)
    # mean tokens per target verification step: accepted drafts + 1 bonus token per step; steps = predicted - accepted
    if pn and da is not None and pn - da > 0: out["accept_len"] = round(pn / (pn - da), 3)
    if r["events"] > 1 and r["t_last"] > r["t_first"]:
        out["client_decode_tps"] = round((r["events"] - 1) / (r["t_last"] - r["t_first"]), 2)
    if r["t_first"]: out["ttft_s"] = round(r["t_first"] - r["t_start"], 3)
    return out

def run(arm, out, prefill_sizes, reps, decode=True, longdec=()):
    res = {"arm": arm, "time": time.time(), "decode": [], "prefill": []}
    src = source_text()
    DECODE["summary"] = "Summarize the following source code in a detailed technical report, section by section:\n\n" + src[200000:212000]
    stream_chat([{"role": "user", "content": "Say hello."}], 32, ignore_eos=False)  # warmup, excluded
    if decode:
        for rep in range(reps):
            for dom, prompt in DECODE.items():
                for temp, top_p in ((0.0, 1.0),) + (((1.0, 0.95),) if rep == 0 else ()):
                    r = stream_chat([{"role": "user", "content": prompt}], 512, temperature=temp, top_p=top_p)
                    r.update(domain=dom, temperature=temp, rep=rep, derived=summarize(r))
                    res["decode"].append(r)
                    json.dump(res, open(out, "w"), indent=1)
                    print(f"[{arm}] decode {dom} T={temp}: {r['derived']} finish={r['finish']}", flush=True)
    for n in prefill_sizes:
        nonce = f"[session {uuid.uuid4().hex}]\n"
        p = build_prompt(src, n, nonce)
        r = stream_chat([{"role": "user", "content": p + "\n\nIn one word, what language is this?"}], 1,
                        ignore_eos=False, thinking=False, cache_prompt=False)
        r.update(target=n, derived=summarize(r)); r.pop("content"); r.pop("reasoning")
        res["prefill"].append(r)
        json.dump(res, open(out, "w"), indent=1)
        print(f"[{arm}] prefill {n}: {r['derived']}", flush=True)
    res["longdec"] = []
    for n in longdec:
        nonce = f"[session {uuid.uuid4().hex}]\n"
        p = build_prompt(src, n, nonce)
        r = stream_chat([{"role": "user", "content": p + "\n\nWrite a detailed technical report on the code above, section by section."}], 512,
                        cache_prompt=False)
        r.update(target=n, derived=summarize(r)); r.pop("content"); r.pop("reasoning")
        res["longdec"].append(r)
        json.dump(res, open(out, "w"), indent=1)
        print(f"[{arm}] longdec {n}: {r['derived']}", flush=True)
    res["done"] = True
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prefill", default="2048,8192,32768"); ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--longdec", default="")
    a = ap.parse_args()
    run(a.arm, a.out, [int(x) for x in a.prefill.split(",") if x], a.reps, decode=not a.no_decode,
        longdec=[int(x) for x in a.longdec.split(",") if x])
