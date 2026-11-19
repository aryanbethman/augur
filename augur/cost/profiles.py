"""Which GPU profiles exist: ``python -m augur.cost.profiles [--fork DIR]``.

Profiles live in the LLMServingSim fork under
``profiler/perf/<hardware>/<org>/<model>/<dtype>/tp<N>/``; each one prices
steps for that model on that GPU at that TP degree.
"""

from __future__ import annotations

import argparse
import os


def default_fork() -> str:
    return os.environ.get("AUGUR_FORK", os.path.expanduser("~/llmss-vllm-sim-20261114"))


def available(fork_root: str) -> list[tuple[str, str, str, int]]:
    """(hardware, model, dtype, tp) for every profile in the fork."""
    perf = os.path.join(fork_root, "profiler", "perf")
    out = []
    if not os.path.isdir(perf):
        return out
    for dirpath, dirnames, _ in os.walk(perf):
        dirnames[:] = [d for d in dirnames if d != "moe_components"]
        name = os.path.basename(dirpath)
        if not (name.startswith("tp") and name[2:].isdigit()):
            continue
        parts = os.path.relpath(dirpath, perf).split(os.sep)
        if len(parts) < 4:
            continue
        hw, dtype, model = parts[0], parts[-2], "/".join(parts[1:-2])
        out.append((hw, model, dtype, int(name[2:])))
    return sorted(out)


def shipped() -> list[tuple[str, str, str, int]]:
    """(hardware, model, dtype, tp) for every profile shipped inside augur."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles")
    out = []
    for hw in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for model in sorted(os.listdir(os.path.join(root, hw))):
            for name in sorted(os.listdir(os.path.join(root, hw, model))):
                if name.endswith(".json.gz"):
                    dtype, tp = name[:-len(".json.gz")].rsplit("-tp", 1)
                    out.append((hw, model.replace("--", "/"), dtype, int(tp)))
    return out


def check(fork_root: str, hardware: str, model: str, tp: int, variant: str | None = None) -> None:
    """Raise a clear error if no profile covers this hardware/model/TP.

    With ``variant`` (a profile folder such as ``bf16-tp4proj``, what the fork's
    ``--profile-variant`` pins), only that folder counts: the model and TP must
    be present under it, whatever the other folders hold."""
    have = available(fork_root)
    if any(h == hardware and m == model and t == tp and (variant is None or d == variant)
           for h, m, d, t in have):
        return
    if not have:
        raise FileNotFoundError(f"no profiles under {fork_root}/profiler/perf "
                                "(set --fork or AUGUR_FORK to the LLMServingSim fork checkout)")
    where = f" in profile variant {variant!r}" if variant else ""
    pool = [r for r in have if variant is None or r[2] == variant]
    near = [f"  {h} {m} {d} tp{t}" for h, m, d, t in pool if h == hardware or m == model] or \
           [f"  {h} {m} {d} tp{t}" for h, m, d, t in pool or have]
    raise FileNotFoundError(f"no profile for {model} on {hardware} at tp{tp}{where}. Available:\n" + "\n".join(near)
                            + "\n(python -m augur.cost.profiles lists all; the fork's profiler adds new ones)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m augur.cost.profiles", description=__doc__.splitlines()[0])
    p.add_argument("--fork", default=default_fork())
    a = p.parse_args(argv)
    rows = [(*r, "augur (--cost tables)") for r in shipped()]
    rows += [(*r, "fork (--cost llmservingsim)") for r in available(a.fork)]
    if not rows:
        print(f"no profiles shipped with augur or under {a.fork}/profiler/perf")
        return 1
    w = max(len(r[1]) for r in rows)
    print(f"{'hardware':<12} {'model':<{w}} {'dtype':<6} tp  source")
    for h, m, d, t, src in rows:
        print(f"{h:<12} {m:<{w}} {d:<6} {t:<3} {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
