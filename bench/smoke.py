#!/usr/bin/env python3
"""Functional smokes against the recipe server: exact answer, image, tool call, thinking on/off."""
import base64, json, sys, urllib.request, subprocess, os
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8090")
def chat(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())
out = {}
r = chat({"messages": [{"role": "user", "content": "Compute 17*23. Reply with only the integer."}], "max_tokens": 4096, "temperature": 0, "top_k": 1})
c = r["choices"][0]; out["math"] = {"content": c["message"]["content"], "finish": c["finish_reason"], "has_reasoning": bool(c["message"].get("reasoning_content"))}
r = chat({"messages": [{"role": "user", "content": "Compute 17*23. Reply with only the integer."}], "max_tokens": 256, "temperature": 0, "top_k": 1,
          "chat_template_kwargs": {"enable_thinking": False}})
c = r["choices"][0]; out["no_think"] = {"content": c["message"]["content"], "finish": c["finish_reason"], "has_reasoning": bool(c["message"].get("reasoning_content"))}
# image: render a known number into a PNG with ImageMagick-free pure python (PPM->PNG via zlib)
import zlib, struct
W, H = 240, 80
digits = {"4": ["101", "101", "111", "001", "001"], "7": ["111", "001", "010", "010", "010"], "2": ["111", "001", "111", "100", "111"]}
txt = "472"; px = [[255] * W for _ in range(H)]
for i, ch in enumerate(txt):
    for y, row in enumerate(digits[ch]):
        for x, b in enumerate(row):
            if b == "1":
                for yy in range(10 + y * 12, 10 + y * 12 + 12):
                    for xx in range(20 + i * 70 + x * 16, 20 + i * 70 + x * 16 + 16): px[yy][xx] = 0
raw = b"".join(b"\x00" + bytes(sum(([v, v, v] for v in row), [])) for row in px)
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
b64 = base64.b64encode(png).decode()
r = chat({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
          {"type": "text", "text": "What number is written in this image? Reply with only the digits."}]}], "max_tokens": 4096, "temperature": 0, "top_k": 1})
c = r["choices"][0]; out["image"] = {"expected": txt, "content": c["message"]["content"], "finish": c["finish_reason"], "prompt_tokens": r["usage"]["prompt_tokens"]}
tools = [{"type": "function", "function": {"name": "get_inventory", "description": "Return the inventory count for a product.",
          "parameters": {"type": "object", "properties": {"product": {"type": "string", "enum": ["keyboard", "monitor"]}}, "required": ["product"]}}}]
r = chat({"messages": [{"role": "user", "content": "How many keyboards are in stock?"}], "tools": tools, "max_tokens": 4096, "temperature": 0, "top_k": 1})
c = r["choices"][0]; out["tool"] = {"finish": c["finish_reason"], "tool_calls": [(t["function"]["name"], t["function"]["arguments"]) for t in (c["message"].get("tool_calls") or [])]}
img = os.environ.get("CHART")
if img:
    b = base64.b64encode(open(img, "rb").read()).decode()
    r = chat({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}},
              {"type": "text", "text": "This is a benchmark chart. Which model has the highest GPQA Diamond score and what is the value? One line."}]}], "max_tokens": 8192, "temperature": 0, "top_k": 1})
    c = r["choices"][0]; out["chart"] = {"content": c["message"]["content"], "finish": c["finish_reason"], "prompt_tokens": r["usage"]["prompt_tokens"]}
print(json.dumps(out, indent=1))
