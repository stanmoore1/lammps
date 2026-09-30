#!/usr/bin/env python3
"""Decode uploaded Claude desktop cache pages -> merged event stream.
Usage: recover.py <dir-with-page-files> [out.json]
Follows the recovery doc: URL is plaintext at the START, headers at the END,
zstd frame between; use decompress(), not stream_reader().
"""
import glob, json, os, sys
import zstandard as zstd

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(src), "allev.json")
dctx = zstd.ZstdDecompressor()
events, pages, bad = {}, 0, 0

for path in sorted(glob.glob(os.path.join(src, "**", "*"), recursive=True)):
    if not os.path.isfile(path):
        continue
    try:
        d = open(path, "rb").read()
    except Exception:
        continue
    if b"/events?limit" not in d[:8192]:          # filter by URL first
        continue
    i = d.find(b"\x28\xb5\x2f\xfd")               # zstd magic
    if i < 0:
        continue
    end = d.find(b"HTTP/1.1 200")                 # headers at the END
    body = d[i:end if end > i else len(d)]
    try:
        data = json.loads(dctx.decompress(body, max_output_size=800_000_000))["data"]
    except Exception:
        bad += 1
        continue
    pages += 1
    for e in data:
        s = e.get("sequence_num")
        if s is not None:
            events[int(s)] = e                    # dedupe overlapping pages

if not events:
    print(f"pages ok {pages}, failed {bad}, NO EVENTS FOUND"); sys.exit(1)
ks = sorted(events)
gaps = [(a, b) for a, b in zip(ks, ks[1:]) if b - a > 1]
print(f"pages ok {pages}, failed {bad}, unique events {len(events)}")
print(f"span {ks[0]}..{ks[-1]}")
print(f"gaps {len(gaps)}, missing {sum(b-a-1 for a,b in gaps)}")
json.dump({str(k): events[k] for k in ks}, open(out, "w"))
print(f"wrote {out}")
