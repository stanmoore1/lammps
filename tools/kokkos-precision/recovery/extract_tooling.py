#!/usr/bin/env python3
"""Extract tooling, subagent prompts, reports, and user messages from a
recovered Claude Code session event stream.

Usage:
  extract_tooling.py <events.json> <out-dir> [--names NAME ...] [--ts]

<events.json> is either the merged {"<seq>": event, ...} dictionary written by
recover_cache.py, or the plain list written by the browser-console fallback in
stanmoore1/private:lammps/kokkos-precision/kokkos-precision-session-recovery.md.  <out-dir> receives the same layout as stanmoore1/private:lammps/kokkos-precision/history/:

  tooling/<name>.vN            every distinct full version of each named file,
                               from Write tool calls, Bash heredocs
                               (cat > FILE <<'EOF' ... EOF; "cat >>" appends
                               are applied to the previous version), and
                               full-file Read snapshots
  tooling/<name>.edits.md      every Edit tool call on a named file (old/new)
  tooling/README.md            chronological index with provenance (seq numbers)
  tooling/generator_commands/seqN.sh
                               every Bash command that wrote a named file or
                               ran "ninja ... -t commands"
  agents/all_agent_calls.jsonl every Agent/Task tool_use prompt, verbatim
  agents/templates.md          main-thread prompts clustered by normalized
                               text: first prompt verbatim, the rest as diffs
  agents/nested_agent_calls.md Agent calls made by subagents, verbatim
  agents/notifications.md      every <task-notification> (background agents'
                               final reports, with their judgment calls)
  user_messages.md             main-thread user text messages, verbatim

Default names: RECIPE.md check.sh checkall.sh check_mixed.sh checkall_mixed.sh
chk.sh survey.sh vc.sh vc_mixed.sh full_cmd.txt full_cmd_single.txt
full_cmd_mixed.txt.  Matching is on the basename of the written path.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys

DEFAULT_NAMES = ["RECIPE.md", "check.sh", "checkall.sh", "check_mixed.sh",
                 "checkall_mixed.sh", "chk.sh", "survey.sh", "vc.sh",
                 "vc_mixed.sh", "full_cmd.txt", "full_cmd_single.txt",
                 "full_cmd_mixed.txt"]
AGENT_TOOLS = ("Agent", "Task")
HEREDOC = re.compile(r"cat\s*(>>?)\s*\"?([^\s\"<]+)\"?\s*<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\3\s*(?:\n|$)",
                     re.S)
NUMBERED = re.compile(r"(?m)^\s*(\d+)\t(.*)$")


def load_events(path):
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if isinstance(data, dict):
        evs = list(data.values())
    else:
        evs = list(data)
    out = {}
    for e in evs:
        if isinstance(e, dict) and e.get("sequence_num") is not None:
            out[int(e["sequence_num"])] = e
    return [out[k] for k in sorted(out)]


def blocks(ev):
    """(payload, list of content blocks or a plain string)"""
    p = ev.get("payload")
    if not isinstance(p, dict):
        return {}, []
    m = p.get("message")
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, str):
        return p, c
    return p, c if isinstance(c, list) else []


def text_of(block):
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    if block.get("type") == "text":
        return str(block.get("text", ""))
    if block.get("type") == "tool_result":
        c = block.get("content")
        if isinstance(c, list):
            return "\n".join(text_of(x) for x in c)
        return str(c or "")
    return ""


def sha(s):
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:8]


def normalize(prompt):
    s = re.sub(r"/\S+/", "<PATH>/", prompt)
    s = re.sub(r"\b[\w.-]+_kokkos\.(cpp|h)\b", "<FILE>", s)
    s = re.sub(r"\d+", "N", s)
    return re.sub(r"\s+", " ", s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("events")
    ap.add_argument("outdir")
    ap.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="similarity for clustering prompts into families")
    args = ap.parse_args()
    names = set(args.names)
    evs = load_events(args.events)
    if not evs:
        sys.exit("no events with sequence_num found")
    seqs = [int(e["sequence_num"]) for e in evs]
    gaps = sum(b - a - 1 for a, b in zip(seqs, seqs[1:]) if b - a > 1)
    print(f"{len(evs)} events, seq {seqs[0]}..{seqs[-1]}, {gaps} missing")

    od = args.outdir
    for d in ("tooling/generator_commands", "agents"):
        os.makedirs(os.path.join(od, d), exist_ok=True)

    versions = {n: [] for n in names}   # name -> [(seq, how, text)]
    edits = {n: [] for n in names}
    read_ids = {}                       # tool_use_id -> file path of a Read
    gens, agent_calls, notes, users = [], [], [], []

    def add_version(name, seq, how, text):
        v = versions[name]
        if v and v[-1][2] == text:
            return
        if any(t == text for _, _, t in v):
            return
        v.append((seq, how, text))

    for ev in evs:
        seq = int(ev["sequence_num"])
        ts = str(ev.get("created_at", ""))
        p, content = blocks(ev)
        sub = bool(p.get("parent_tool_use_id"))
        who = "sub" if sub else "main"
        role = ev.get("event_type") or (p.get("message") or {}).get("role")

        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "tool_use":
                name, inp = b.get("name"), b.get("input")
                if not isinstance(inp, dict):
                    continue
                if name == "Write":
                    base = os.path.basename(str(inp.get("file_path", "")))
                    if base in names:
                        add_version(base, seq, f"Write ({who})", str(inp.get("content", "")))
                elif name == "Edit":
                    base = os.path.basename(str(inp.get("file_path", "")))
                    if base in names:
                        edits[base].append((seq, who, inp))
                elif name == "Read":
                    read_ids[b.get("id")] = str(inp.get("file_path", ""))
                elif name == "Bash":
                    cmd = str(inp.get("command", ""))
                    hit = False
                    for m in HEREDOC.finditer(cmd):
                        base = os.path.basename(m.group(2))
                        if base in names:
                            hit = True
                            body = m.group(4) + "\n"
                            if m.group(1) == ">>":
                                prev = versions[base][-1][2] if versions[base] else ""
                                add_version(base, seq, f"heredoc append ({who})", prev + body)
                            else:
                                add_version(base, seq, f"heredoc ({who})", body)
                    if not hit and "-t commands" in cmd and "ninja" in cmd:
                        hit = True
                    if not hit and any(re.search(r">\s*\S*" + re.escape(n) + r"\b", cmd) for n in names):
                        hit = True
                    if hit:
                        gens.append((seq, who, str(inp.get("description", "")), cmd))
                elif name in AGENT_TOOLS:
                    agent_calls.append({
                        "seq": seq, "ts": ts, "id": b.get("id"), "nested": sub,
                        "description": inp.get("description"),
                        "subagent_type": inp.get("subagent_type"),
                        "run_in_background": inp.get("run_in_background"),
                        "prompt": str(inp.get("prompt", ""))})
            elif t == "tool_result":
                path = read_ids.get(b.get("tool_use_id"))
                base = os.path.basename(path) if path else ""
                if base in names:
                    lines = {int(n): s for n, s in NUMBERED.findall(text_of(b))}
                    if lines and min(lines) == 1 and max(lines) == len(lines):
                        add_version(base, seq, f"Read snapshot ({who})",
                                    "\n".join(lines[i] for i in range(1, len(lines) + 1)) + "\n")
            elif t == "text" and role == "user" and not sub:
                txt = str(b.get("text", ""))
                if "<task-notification>" in txt:
                    notes.append((seq, ts, txt))
                elif txt.strip():
                    users.append((seq, ts, txt))

    # ---- tooling
    idx = ["# Tooling versions (chronological)", ""]
    for n in sorted(names):
        for i, (seq, how, text) in enumerate(versions[n], 1):
            fn = f"{n}.v{i}"
            with open(os.path.join(od, "tooling", fn), "w") as fh:
                fh.write(text)
            idx.append(f"- `{fn}` ({text.count(chr(10))} lines): seq {seq}, {how}, sha {sha(text)}")
        if edits[n]:
            fn = f"{n}.edits.md"
            with open(os.path.join(od, "tooling", fn), "w") as fh:
                for seq, who, inp in edits[n]:
                    fh.write(f"## seq {seq} ({who}) replace_all={bool(inp.get('replace_all'))}\n"
                             f"--- old\n{inp.get('old_string', '')}\n+++ new\n{inp.get('new_string', '')}\n\n")
            idx.append(f"- `{fn}`: {len(edits[n])} Edit call(s)")
    for seq, who, desc, cmd in gens:
        with open(os.path.join(od, "tooling", "generator_commands", f"seq{seq}.sh"), "w") as fh:
            fh.write(f"# {desc} ({who})\n{cmd}\n")
        idx.append(f"- `generator_commands/seq{seq}.sh`: {desc}")
    with open(os.path.join(od, "tooling", "README.md"), "w") as fh:
        fh.write("\n".join(idx) + "\n")

    # ---- agents
    with open(os.path.join(od, "agents", "all_agent_calls.jsonl"), "w") as fh:
        for a in agent_calls:
            fh.write(json.dumps(a) + "\n")
    main_calls = [a for a in agent_calls if not a["nested"]]
    fams = []                          # [representative normalized, [calls]]
    for a in main_calls:
        na = normalize(a["prompt"])
        best, br = None, 0.0
        for f in fams:
            r = difflib.SequenceMatcher(None, f[0], na, autojunk=False).quick_ratio()
            if r >= args.threshold:
                r = difflib.SequenceMatcher(None, f[0], na, autojunk=False).ratio()
            if r > br:
                best, br = f, r
        if best is not None and br >= args.threshold:
            best[1].append(a)
        else:
            fams.append([na, [a]])
    with open(os.path.join(od, "agents", "templates.md"), "w") as fh:
        fh.write(f"# {len(main_calls)} Agent calls (main thread), grouped into {len(fams)} prompt families\n\n"
                 "Each family shows its FIRST prompt verbatim, then every later run as a unified diff "
                 "against it.  all_agent_calls.jsonl has every prompt verbatim.\n\n")
        for k, (_, calls) in enumerate(fams, 1):
            t0 = calls[0]
            fh.write(f"## F{k:02d}: {len(calls)} run(s), seq {t0['seq']}..{calls[-1]['seq']}\n\n"
                     f"### template = seq {t0['seq']} '{t0['description']}' (verbatim)\n\n{t0['prompt']}\n\n")
            for c in calls[1:]:
                d = difflib.unified_diff(t0["prompt"].splitlines(), c["prompt"].splitlines(),
                                         f"seq{t0['seq']}", f"seq{c['seq']}", n=0, lineterm="")
                fh.write(f"### run seq {c['seq']} {c['ts'][:16]} '{c['description']}'\n```diff\n"
                         + "\n".join(d) + "\n```\n\n")
    nested = [a for a in agent_calls if a["nested"]]
    with open(os.path.join(od, "agents", "nested_agent_calls.md"), "w") as fh:
        fh.write(f"# {len(nested)} Agent calls made BY subagents, verbatim\n\n")
        for a in nested:
            fh.write(f"## seq {a['seq']} {a['ts']} '{a['description']}' type={a['subagent_type']!r}\n"
                     f"````\n{a['prompt']}\n````\n\n")
    with open(os.path.join(od, "agents", "notifications.md"), "w") as fh:
        for seq, ts, txt in notes:
            fh.write(f"## seq {seq} {ts}\n{txt}\n\n")

    # ---- user messages
    with open(os.path.join(od, "user_messages.md"), "w") as fh:
        fh.write("# User messages (main thread only)\n")
        for seq, ts, txt in users:
            fh.write(f"## seq {seq} {ts}\n{txt}\n")

    nv = sum(len(v) for v in versions.values())
    print(f"{nv} tool versions, {sum(len(v) for v in edits.values())} edits, {len(gens)} generator commands")
    print(f"{len(main_calls)} main-thread agent calls in {len(fams)} families, {len(nested)} nested")
    print(f"{len(notes)} task notifications, {len(users)} user messages -> {od}")


if __name__ == "__main__":
    main()
