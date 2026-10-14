"""Replay the vase..vase24 concept chain (Stock UM vase + ND25FN-4) on Osado from the recovered transcript calls.

Mac paths are rewritten to Osado paths; every python run goes through the memory-capped wrapper
(~/temp/claude-vase-recovery/py, systemd scope MemoryMax=48G, no swap). Boox/adb, sips, open and
sleeps are stubbed. usage: replay_osado.py [first_index] [last_index]
"""
import json, os, re, subprocess, sys, time

B = os.path.expanduser("~/temp/claude-vase-recovery")
MAC_SESS = "/private/tmp/claude-501/-Users-antor-gh-lx-top-baffle-v2/2d2e67bb-b5f6-4669-8c80-db03dec4a417"
OS_SESS = f"{B}/session"
MAC_REPO = "/Users/antor/gh/lx/top_baffle_v2"
OS_REPO = os.path.expanduser("~/temp/claude-obiwan-concept/repo/top_baffle_v2")
PY = f"{B}/py"
calls = json.load(open(f"{B}/all_calls.json")); idx = json.load(open(f"{B}/vase_idx.json"))
lo = int(sys.argv[1]) if len(sys.argv) > 1 else idx[0]; hi = int(sys.argv[2]) if len(sys.argv) > 2 else idx[-1]

def remap(s):
    s = s.replace(MAC_SESS, OS_SESS).replace(MAC_REPO, OS_REPO)
    s = s.replace("/Users/antor/gh/lx/.venv/bin/python", PY).replace("/Users/antor/gh/lx/.venv", os.path.dirname(os.path.dirname(PY)))
    s = s.replace("/Users/antor/.cache/claude-obiwan-port", os.path.expanduser("~/temp/claude-obiwan-concept"))
    return s

STUB = ("cp(){ command cp \"$@\" 2>/dev/null; return 0; }; sips(){ return 0; }; open(){ return 0; }; "
        "adb(){ return 0; }; sleep(){ return 0; }; pbcopy(){ cat >/dev/null; }\n")
log = open(f"{B}/replay.log", "a")
def run(cmd, timeout=2400):
    t0 = time.time()
    try:
        r = subprocess.run(["bash", "-c", cmd], cwd=OS_REPO, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)[-400:], time.time() - t0
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", time.time() - t0

for i in idx:
    if i < lo or i > hi: continue
    ts, nm, inp = calls[i]
    if nm in ("Write", "Edit"):
        fp = remap(inp["file_path"])
        if not fp.startswith(OS_SESS): log.write(f"{i} SKIP non-scratch {nm} {fp}\n"); continue
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        if nm == "Write":
            open(fp, "w").write(remap(inp["content"])); log.write(f"{i} WRITE {fp}\n")
        else:
            s = open(fp).read() if os.path.exists(fp) else ""; a, b = remap(inp["old_string"]), remap(inp["new_string"])
            if a in s: open(fp, "w").write(s.replace(a, b, -1 if inp.get("replace_all") else 1)); log.write(f"{i} EDIT ok {fp}\n")
            else: log.write(f"{i} EDIT MISS {fp}\n")
        log.flush(); continue
    cmd = inp["command"]
    if "claude/projects" in cmd or "Library/Android" in cmd and "python" not in cmd: log.write(f"{i} SKIP meta/boox\n"); log.flush(); continue
    cmd = re.sub(r"until .*?; do sleep \d+; done", "true", cmd, flags=re.S)
    cmd = cmd.replace("~/Library/Android/sdk/platform-tools/adb", "true").replace("$ADB", "true")
    cmd = cmd.replace("sed -i '' ", "sed -i ")                    # macOS sed in-place syntax -> GNU sed
    cmd = STUB + remap(cmd)
    rc, out, dt = run(cmd)
    log.write(f"{i} rc={rc} {dt:.0f}s | {inp['command'][:110]!r}\n")
    if rc != 0: log.write("     out: " + out.replace("\n", " | ")[-380:] + "\n")
    log.flush()
log.write("REPLAY_DONE\n"); log.close()
