#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import sys

RLLA_IMPORT = "from bfcl_eval.model_handler.local_inference.rlla_handler import RLLAHandler"
QWENFC_IMPORT = "from bfcl_eval.model_handler.local_inference.qwen_fc_handler import QwenFCHandler"

LOCAL_HANDLER_SRC = {"rlla": ("rlla_handler.py", RLLA_IMPORT),
                     "qwenfc": ("qwen_fc_handler.py", QWENFC_IMPORT)}

ENTRY = """    "%(name)s": ModelConfig(
        model_name="%(name)s",
        display_name="%(display)s",
        url="%(url)s",
        org="%(org)s",
        license="%(license)s",
        model_handler=%(handler)s,
        input_price=None,
        output_price=None,
        is_fc_model=%(is_fc)s,
        underscore_to_dot=%(dot)s,
    ),
"""

SPECS = {
    "rlla": dict(handler="RLLAHandler", anchor="local_inference_model_map = {",
                 url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct", org="Qwen",
                 license="apache-2.0"),
    "qwenfc": dict(handler="QwenFCHandler", anchor="local_inference_model_map = {",
                   url="https://huggingface.co/Qwen/Qwen3.5-4B", org="Qwen",
                   license="apache-2.0", fc=True, dot=True),
}


def install_local_handler(install, repo_root, handler):
    fname, import_line = LOCAL_HANDLER_SRC[handler]
    src = os.path.join(repo_root, "canonperm", fname)
    if not os.path.isfile(src):
        sys.exit("no %s handler at %s (pass --repo-root)" % (handler, src))
    dst = os.path.join(install, "bfcl_eval", "model_handler", "local_inference", fname)
    if not os.path.isfile(dst) or open(dst).read() != open(src).read():
        shutil.copyfile(src, dst)
        print("copied  %s" % dst)
    cfg = os.path.join(install, "bfcl_eval", "constants", "model_config.py")
    text = open(cfg).read()
    if import_line not in text:
        hits = list(re.finditer(r"^from bfcl_eval\.model_handler\.local_inference\.[^\n]+\n", text, re.M))
        at = hits[-1].end() if hits else 0
        open(cfg, "w").write(text[:at] + import_line + "\n" + text[at:])
        print("imported %s in %s" % (import_line.rsplit(" ", 1)[-1], cfg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", required=True, help="berkeley-function-call-leaderboard dir")
    ap.add_argument("--name", required=True, help="model tag; no underscore")
    ap.add_argument("--display", default="")
    ap.add_argument("--handler", default="rlla", choices=sorted(SPECS))
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--fc", action="store_true",
                    help="register in NATIVE FUNCTION-CALLING mode (is_fc_model=True). The server must then "
                         "run with --enable-auto-tool-choice --tool-call-parser; eval_open.sh --fc does.")
    a = ap.parse_args()
    if "_" in a.name:
        sys.exit("model tag %r contains an underscore; evaluate would look up %r"
                 % (a.name, a.name.replace("_", "/")))
    if a.handler in LOCAL_HANDLER_SRC:
        install_local_handler(a.install, a.repo_root, a.handler)
    fc = bool(SPECS[a.handler].get("fc", False))
    if a.fc and not fc:
        sys.exit("--fc with --handler %s: that handler is prompt-only. Use --handler qwenfc." % a.handler)
    if fc and not a.fc:
        print("note: --handler %s is function-calling; registering with is_fc_model=True" % a.handler)
    cfg = os.path.join(a.install, "bfcl_eval", "constants", "model_config.py")
    src = open(cfg).read()
    spec = dict(SPECS[a.handler], name=a.name, display=a.display or a.name,
                is_fc="True" if fc else "False",
                dot="True" if SPECS[a.handler].get("dot", False) else "False")
    if '"%s": ModelConfig(' % a.name in src:
        block_start = src.index('"%s": ModelConfig(' % a.name)
        block_end = src.index("),", block_start) + 2
        block = src[block_start:block_end]
        fixed, changed = block, []
        for field, want in (("is_fc_model", spec["is_fc"]), ("underscore_to_dot", spec["dot"]),
                            ("model_handler", spec["handler"])):
            m = re.search(r"%s=([A-Za-z]+)," % field, fixed)
            if m and m.group(1) != want:
                changed.append("%s %s -> %s" % (field, m.group(1), want))
                fixed = fixed[:m.start()] + "%s=%s," % (field, want) + fixed[m.end():]
        if changed:
            open(cfg, "w").write(src[:block_start] + fixed + src[block_end:])
            print("updated registration for %s: %s" % (a.name, "; ".join(changed)))
            print("  NOTE: these fields decide SCORING -- re-score the affected cells.")
        else:
            print("already registered: %s" % a.name)
        return
    anchor = spec["anchor"]
    if anchor not in src:
        sys.exit("anchor %r not found in %s -- upstream layout changed" % (anchor, cfg))
    src = src.replace(anchor, anchor + "\n" + (ENTRY % spec).rstrip("\n"), 1)
    open(cfg, "w").write(src)
    print("registered: %s (%s) in %s" % (a.name, spec["handler"], cfg))


if __name__ == "__main__":
    main()
