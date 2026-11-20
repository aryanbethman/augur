"""Add a GPU and model to augur: profile it with the LLMServingSim fork's profiler and convert the result.

    python -m augur.profile --model meta-llama/Llama-3.1-70B --hardware H100 --tp 4 [--max-kv 32768 --max-batch-tokens 8192]

Run it on a machine with the GPU (a TP=N profile needs one GPU: the profiler emulates the shard).  Stages, each
resumable (a stage that finished with the same arguments is skipped; the fork's profiler also resumes inside a
stage from its CSVs):

  check    the GPU (nvidia-smi), vLLM, the fork, and the model's config.json (taken from the local Hugging Face
           cache; nothing is downloaded unless --allow-download)
  memory   vLLM's memory accounting for the model on this GPU (weights, non-torch, peak activation, CUDA graphs,
           KV cache) from a short dummy-weight startup; needs N visible GPUs at TP N, else it is skipped.  Cheap,
           so it comes first: it also shows that the GPU, vLLM and the config work
  layers   the fork's profiler, dense / per-sequence / attention tables for TP 1 and N, over a grid that reaches
           --max-kv tokens of KV and --max-batch-tokens tokens per step (skew excluded: next stage)
  skew     the fork's attention skew sweep (heterogeneous decode batches); --no-skew leaves it out
  convert  augur.cost.convert, with its self-check, into augur/profiles/<hardware>/<model>/<variant>-tp<N>.json.gz;
           the measured environment and memory numbers go into the profile's provenance

``--dry-run`` prints the commands and the grid and runs nothing.  ``--stages layers,convert`` runs only those.

Method and data: LLMServingSim (MIT licence, CASYS Lab at KAIST); its profiler measures one layer of the model
with dummy weights, so the model's weights need not be downloaded.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from augur.cost.profiles import default_fork
from augur.cost.tables import PROFILE_DIR

STAGES = ("check", "memory", "layers", "skew", "convert")
SQRT2 = 1.4142135623730951

# Cluster parameters of the self-check at TP>1 (the fork's cluster config): link, GPU memory, GPU memory
# bandwidth.  Other hardware needs them on the command line.
HARDWARE_DEFAULTS = {"H100": {"hbm_gb": 80, "npu_mem_bw": 3350.0, "link_bw": 900.0}}

# Seconds the profiler took on the reference run (Qwen3-1.7B, RTX 4060 Ti, TP1, fork 2a2321d, default grid):
# 14,104 attention shots in 41 min, 24,335 skew cases in 2 h 19 min.
REFERENCE_TIMES = "Qwen3-1.7B on an RTX 4060 Ti, default grid: attention 14,104 shots 41 min, skew 24,335 cases 2 h 19 min"


class ProfileError(Exception):
    """Something the person running the profile can fix."""


# --- grid ---------------------------------------------------------------------

def geometric(maximum: int, factor: float = 2.0, *, start: int = 1, zero: bool = False) -> list[int]:
    """The fork profiler's geometric axis (profiler/core/sampling.py): start, start*factor, ... and the maximum."""
    if factor <= 1 or maximum < 0 or start < 1:
        raise ValueError("bad grid parameters")
    points = {0} if zero else set()
    if maximum:
        points.add(maximum)
    value = float(start)
    while value < maximum:
        points.add(max(1, round(value)))
        value *= factor
    return sorted(points)


def grid_preview(max_kv: int, max_batch_tokens: int, max_num_seqs: int) -> dict:
    """The axes the profiler will sweep, before its feasibility filters and CUDA-graph anchors (which only add
    or remove points): an upper bound on the attention shots, not a count."""
    axes = {
        "dense tokens": geometric(max_batch_tokens),
        "sequences (lm_head, sampler)": geometric(max_num_seqs),
        "attention prefill chunk": geometric(max_batch_tokens, zero=True),
        "attention prefill key (KV)": geometric(max_kv, zero=True),
        "attention decodes": geometric(max_num_seqs, SQRT2),
        "attention decode KV": geometric(max_kv, zero=True),
    }
    bound = 1
    for k in ("attention prefill chunk", "attention prefill key (KV)", "attention decodes", "attention decode KV"):
        bound *= len(axes[k])
    return {"axes": axes, "attention_shots_upper_bound": bound}


# --- inputs -------------------------------------------------------------------

def hf_cache_dirs() -> list[Path]:
    dirs = []
    for var, sub in (("HF_HUB_CACHE", ""), ("HUGGINGFACE_HUB_CACHE", ""), ("HF_HOME", "hub")):
        if os.environ.get(var):
            dirs.append(Path(os.environ[var]) / sub)
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    return dirs


def cached_config(model: str) -> Path | None:
    """config.json of the model in the local Hugging Face cache (any snapshot), or None."""
    folder = "models--" + model.replace("/", "--")
    for d in hf_cache_dirs():
        found = sorted(glob.glob(str(d / folder / "snapshots" / "*" / "config.json")))
        if found:
            return Path(found[-1])
    return None


def fork_config_path(fork: str, model: str) -> Path:
    return Path(fork) / "configs" / "model" / (model + ".json")


def ensure_config(fork: str, model: str, allow_download: bool, dry_run: bool) -> tuple[Path, str]:
    """(path of the model config in the fork, where it came from).  Copies it from the local cache; never downloads
    unless allowed (then the fork's profiler fetches it itself)."""
    target = fork_config_path(fork, model)
    if target.is_file():
        return target, "fork configs/model"
    src = cached_config(model)
    if src is not None:
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
        return target, f"copied from the Hugging Face cache ({src})"
    if allow_download:
        return target, "will be downloaded by the fork's profiler (--allow-download)"
    raise ProfileError(f"{model}: no config.json in {target.parent} or the local Hugging Face cache "
                       f"({', '.join(str(d) for d in hf_cache_dirs())}). Put it at {target}, or pass --allow-download "
                       "(only the config is fetched: profiling uses dummy weights)")


def gpu_info() -> list[dict]:
    """[{name, memory_gib}] from nvidia-smi, honouring CUDA_VISIBLE_DEVICES; raises ProfileError without a GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise ProfileError(f"no GPU: nvidia-smi failed ({e})") from e
    gpus = []
    for line in out.strip().splitlines():
        idx, name, mib, driver = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "memory_gib": round(float(mib) / 1024, 1), "driver": driver})
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip():
        want = {int(x) for x in vis.split(",") if x.strip().isdigit()}
        gpus = [g for g in gpus if g["index"] in want]
    if not gpus:
        raise ProfileError("no visible GPU")
    return gpus


# --- vLLM memory accounting ----------------------------------------------------

_NUM = r"([0-9]+(?:\.[0-9]+)?)"
_PATTERNS = {
    "weights_gib": re.compile(r"Model loading took " + _NUM + r" GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([0-9,]+) tokens"),
    "available_kv_gib": re.compile(r"Available KV cache memory: " + _NUM + r" GiB"),
}
_STARTUP = re.compile(r"Free memory on device \(" + _NUM + "/" + _NUM + r" GiB\) on startup\..*?Actual usage is "
                      + _NUM + r" GiB for consumed memory \(weights \+ non-torch\), " + _NUM
                      + r" GiB for peak activation, and " + _NUM + r" GiB for CUDAGraph memory", re.S)


def parse_vllm_memory(text: str) -> dict:
    """Numbers from the startup log of vLLM 0.33 (first rank's lines): weights, non-torch (consumed minus weights),
    peak activation, CUDA graphs, KV cache.  Missing items are absent, not guessed."""
    out: dict = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(text)
        if m:
            out[key] = int(m.group(1).replace(",", "")) if key == "kv_cache_tokens" else float(m.group(1))
    m = _STARTUP.search(text)
    if m:
        free, total, consumed, peak, graphs = (float(x) for x in m.groups())
        out.update({"free_at_startup_gib": free, "gpu_total_gib": total, "consumed_gib": consumed,
                    "peak_activation_gib": peak, "cuda_graph_gib": graphs})
        if "weights_gib" in out:
            out["non_torch_gib"] = round(consumed - out["weights_gib"], 2)
    return out


def _probe(spec_path: str) -> int:
    """Internal: boot vLLM with dummy weights and no tokenizer, as the profiler does, and exit.  The startup log
    goes to stdout/stderr for parse_vllm_memory."""
    spec = json.loads(Path(spec_path).read_text())
    from vllm import LLM
    kwargs = dict(model=spec["model_dir"], load_format="dummy", skip_tokenizer_init=True,
                  tensor_parallel_size=spec["tp"], gpu_memory_utilization=spec["gpu_memory_utilization"],
                  max_model_len=spec["max_model_len"], max_num_batched_tokens=spec["max_num_batched_tokens"],
                  max_num_seqs=spec["max_num_seqs"], seed=0)
    for k in ("dtype", "block_size"):
        if spec.get(k):
            kwargs[k] = spec[k]
    LLM(**kwargs)
    print("AUGUR_PROBE_DONE", flush=True)
    return 0


# --- commands -------------------------------------------------------------------

def _tp_list(tp: int) -> str:
    return "1" if tp == 1 else f"1,{tp}"


def profiler_command(a, only_skew: bool = False) -> list[str]:
    """The fork profiler invocation for the layers (default) or skew stage."""
    cmd = [a.python, "-m", "profiler", "profile", a.model, "--hardware", a.hardware, "--tp", _tp_list(a.tp),
           "--max-num-batched-tokens", str(a.max_batch_tokens), "--max-num-seqs", str(a.max_num_seqs),
           "--max-model-len", str(a.max_model_len), "--attention-max-kv", str(a.max_kv),
           "--gpu-memory-utilization", str(a.gpu_memory_utilization)]
    if a.block_size:
        cmd += ["--block-size", str(a.block_size)]
    if a.dtype:
        cmd += ["--dtype", a.dtype]
    if a.variant:
        cmd += ["--variant", a.variant]
    cmd += ["--only-skew"] if only_skew else ["--skip-skew"]
    if a.force:
        cmd += ["--force"]
    return cmd + list(a.profiler_arg or [])


def probe_spec(a, model_dir: str) -> dict:
    return {"model_dir": model_dir, "tp": a.tp, "gpu_memory_utilization": a.gpu_memory_utilization,
            "max_model_len": a.max_model_len, "max_num_batched_tokens": a.max_batch_tokens,
            "max_num_seqs": a.max_num_seqs, "dtype": a.dtype, "block_size": a.block_size}


def cluster_config(a) -> dict:
    """The fork cluster config of the self-check: one node, one TP=N instance (what the oracle prices)."""
    d = HARDWARE_DEFAULTS.get(a.hardware, {})
    link_bw = a.link_bw if a.link_bw is not None else d.get("link_bw")
    npu_bw = a.npu_mem_bw if a.npu_mem_bw is not None else d.get("npu_mem_bw")
    hbm = a.hbm_gb if a.hbm_gb is not None else d.get("hbm_gb")
    if None in (link_bw, npu_bw, hbm):
        raise ProfileError(f"{a.hardware}: TP>1 needs --link-bw, --npu-mem-bw (GB/s) and --hbm-gb for the conversion "
                           "self-check (or --cluster-config FILE); augur only knows H100")
    return {"num_nodes": 1, "link_bw": link_bw, "link_latency": a.link_latency_ns,
            "nodes": [{"num_instances": 1, "cpu_mem": {"mem_size": 1024, "mem_bw": 512, "mem_latency": 0},
                       "instances": [{"model_name": a.model, "hardware": a.hardware,
                                      "npu_mem": {"mem_size": hbm, "mem_bw": npu_bw, "mem_latency": 0},
                                      "num_npus": a.tp, "tp_size": a.tp, "pd_type": None}]}]}


def output_path(a, variant: str) -> Path:
    if a.out:
        return Path(a.out)
    return PROFILE_DIR / a.hardware / a.model.replace("/", "--") / f"{variant}-tp{a.tp}.json.gz"


def find_variant(a) -> str:
    """The profile folder the fork wrote for this hardware/model/TP (--variant, else the only one that holds it)."""
    if a.variant:
        return a.variant
    base = Path(a.fork) / "profiler" / "perf" / a.hardware / a.model
    found = sorted(p.parent.name for p in base.glob(f"*/tp{a.tp}") if p.is_dir()) if base.is_dir() else []
    if len(found) == 1:
        return found[0]
    if not found:
        return "bf16"  # the checkpoint-default name for a bf16 model; only used before the profiler has run
    raise ProfileError(f"several profile folders hold tp{a.tp} under {base}: {found}; pass --variant")


def convert_command(a, variant: str, out: Path, cluster: Path | None, prov: Path | None) -> list[str]:
    cmd = [a.python, "-m", "augur.cost.convert", "--fork", a.fork, "--hardware", a.hardware, "--model", a.model,
           "--tp", str(a.tp), "--out", str(out)]
    if a.tp > 1:
        cmd += ["--cluster-config", str(cluster) if cluster else "<generated cluster config>"]
    if prov:
        cmd += ["--provenance-json", str(prov)]
    return cmd


# --- stage bookkeeping ------------------------------------------------------------

class Work:
    """Markers of finished stages, in the work directory: a stage is skipped if it finished with the same command."""

    def __init__(self, root: Path):
        self.root = root

    def _marker(self, stage: str) -> Path:
        return self.root / f"{stage}.done.json"

    def done(self, stage: str, cmd) -> bool:
        m = self._marker(stage)
        try:
            return json.loads(m.read_text())["cmd_sha256"] == self.digest(cmd)
        except (OSError, ValueError, KeyError):
            return False

    def finish(self, stage: str, cmd, **extra) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._marker(stage).write_text(json.dumps({"cmd_sha256": self.digest(cmd), "cmd": list(cmd),
                                                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}, indent=1))

    @staticmethod
    def digest(cmd) -> str:
        return hashlib.sha256(json.dumps(list(cmd)).encode()).hexdigest()


def _env(a) -> dict:
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent.parent)  # so `python -m augur...` finds this checkout
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if not a.allow_download:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def _run(cmd, cwd, env, log: Path | None = None) -> str:
    """Run a command, streaming its output to the terminal (and a log file); returns the output text."""
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    logf = open(log, "w") if log else None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            lines.append(line)
            if logf:
                logf.write(line)
    finally:
        if logf:
            logf.close()
    if proc.wait() != 0:
        raise ProfileError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return "".join(lines)


# --- main -------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m augur.profile", description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="Hugging Face id, e.g. meta-llama/Llama-3.1-70B")
    p.add_argument("--hardware", required=True, help="name for the GPU in augur (H100, RTX4090 ...): a folder name")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-kv", type=int, default=32768, help="longest KV length (tokens) the profile must cover")
    p.add_argument("--max-batch-tokens", type=int, default=8192,
                   help="largest step (tokens); 8192 is vllm serve's default on GPUs of 70 GB and up")
    p.add_argument("--max-num-seqs", type=int, default=1024, help="largest batch (vllm serve default on 70 GB+ GPUs)")
    p.add_argument("--max-model-len", type=int, default=None, help="default: --max-kv + 2")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--dtype", default=None, help="vLLM dtype; default: the checkpoint's")
    p.add_argument("--variant", default=None, help="profile folder name; default: the profiler's (e.g. bf16)")
    p.add_argument("--no-skew", action="store_true", help="leave out the skew sweep (hours)")
    p.add_argument("--profiler-arg", action="append", default=[], metavar="ARG",
                   help="extra argument for `python -m profiler profile` (repeatable)")
    p.add_argument("--fork", default=default_fork(), help="LLMServingSim fork checkout (AUGUR_FORK)")
    p.add_argument("--python", default=sys.executable, help="interpreter that has vLLM and the fork's requirements")
    p.add_argument("--work-dir", default=None, help="markers, logs, memory numbers (default ./augur-profile-work/<hw>-<model>-tp<N>)")
    p.add_argument("--out", default=None, help="profile file (default augur/profiles/<hw>/<model>/<variant>-tp<N>.json.gz)")
    p.add_argument("--stages", default=",".join(STAGES), help=f"comma list of {', '.join(STAGES)}")
    p.add_argument("--force", action="store_true", help="redo finished stages (and pass --force to the profiler)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and the grid, run nothing")
    p.add_argument("--allow-download", action="store_true", help="let the profiler fetch the model's config.json")
    g = p.add_argument_group("TP > 1 conversion self-check (fork cluster config)")
    g.add_argument("--cluster-config", default=None, help="fork cluster config JSON (default: generated)")
    g.add_argument("--link-bw", type=float, default=None, help="GB/s (H100: 900)")
    g.add_argument("--link-latency-ns", type=int, default=0)
    g.add_argument("--npu-mem-bw", type=float, default=None, help="GB/s (H100: 3350)")
    g.add_argument("--hbm-gb", type=float, default=None, help="GPU memory, GB (H100: 80)")
    return p


def estimate_text(a, preview: dict) -> str:
    return (f"Attention grid upper bound {preview['attention_shots_upper_bound']:,} shots per TP "
            f"(the profiler's feasibility filters drop many). Reference: {REFERENCE_TIMES}. "
            "Expect the attention and skew stages to take hours each; run the stages separately (--stages) "
            "and start with --max-kv 8192 to time a pilot.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--_memory-probe"]:
        return _probe(argv[1])
    a = build_parser().parse_args(argv)
    a.fork = os.path.abspath(os.path.expanduser(a.fork))
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"augur.profile: unknown stage {bad}", file=sys.stderr)
        return 2
    if a.max_model_len is None:
        a.max_model_len = a.max_kv + 2
    if a.max_model_len < a.max_kv + 2:
        print("augur.profile: --max-model-len must be at least --max-kv + 2", file=sys.stderr)
        return 2
    if a.tp < 1:
        print("augur.profile: --tp must be positive", file=sys.stderr)
        return 2
    if a.no_skew:
        stages = [s for s in stages if s != "skew"]
    work = Path(a.work_dir or f"augur-profile-work/{a.hardware}-{a.model.replace('/', '--')}-tp{a.tp}").resolve()
    try:
        return run(a, stages, work)
    except ProfileError as e:
        print(f"augur.profile: {e}", file=sys.stderr)
        return 1


def run(a, stages: list[str], work: Path) -> int:
    dry = a.dry_run
    env = _env(a)
    state = Work(work)
    preview = grid_preview(a.max_kv, a.max_batch_tokens, a.max_num_seqs)
    print(f"profile {a.model} on {a.hardware}, TP {a.tp}; fork {a.fork}; work dir {work}")
    print("grid (axes before the profiler's feasibility filters and CUDA-graph anchors):")
    for name, pts in preview["axes"].items():
        print(f"  {name:30s} {len(pts):3d} points: {pts if len(pts) <= 24 else str(pts[:10])[:-1] + ', ..., ' + str(pts[-6:])[1:]}")
    print(" ", estimate_text(a, preview))

    gpus = []
    config_path = fork_config_path(a.fork, a.model)
    if "check" in stages or any(s in stages for s in ("layers", "skew", "memory", "convert")):
        if not Path(a.fork, "profiler", "__main__.py").is_file():
            raise ProfileError(f"{a.fork} is not an LLMServingSim checkout with a profiler (--fork / AUGUR_FORK)")
        config_path, how = ensure_config(a.fork, a.model, a.allow_download, dry)
        print(f"[check] model config: {config_path} ({how})")
        cfg = json.loads(config_path.read_text()) if config_path.is_file() else {}
        limit = cfg.get("max_position_embeddings")
        if limit and a.max_model_len > limit:
            raise ProfileError(f"--max-kv {a.max_kv} needs a context of {a.max_model_len} tokens but {a.model} "
                               f"has max_position_embeddings {limit}")
        if cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts"):
            print("[check] warning: a MoE model: the fork can profile it but augur's table pricer cannot convert it "
                  "(use --cost llmservingsim with the fork's bundle)")
        has_skew = Path(a.fork, "profiler", "core", "skew.py").is_file()
        if not has_skew and "skew" in stages:
            print("[check] the fork has no skew sweep: skipping that stage")
            stages = [s for s in stages if s != "skew"]
        if dry:
            print("[check] dry run: nvidia-smi and vLLM not queried")
        else:
            gpus = gpu_info()
            print(f"[check] GPUs: " + "; ".join(f"{g['index']}: {g['name']} {g['memory_gib']} GiB" for g in gpus))
            ver = subprocess.run([a.python, "-c", "import vllm; print(vllm.__version__)"], capture_output=True,
                                 text=True)
            if ver.returncode:
                raise ProfileError(f"{a.python} cannot import vllm: {ver.stderr.strip().splitlines()[-1:]}")
            print(f"[check] vLLM {ver.stdout.strip()} in {a.python}")

    variant = find_variant(a)
    steps = []  # (stage, cmd)
    spec_path = work / f"memory-probe-tp{a.tp}.json"
    probe_cmd = [a.python, "-m", "augur.profile", "--_memory-probe", str(spec_path)]
    if "memory" in stages:
        steps.append(("memory", probe_cmd))
    if "layers" in stages:
        steps.append(("layers", profiler_command(a)))
    if "skew" in stages:
        steps.append(("skew", profiler_command(a, only_skew=True)))
    out = output_path(a, variant)
    cluster_path = Path(a.cluster_config) if a.cluster_config else work / "cluster.json"
    mem_json = work / f"memory-tp{a.tp}.json"
    prov_path = work / "provenance.json"
    if "convert" in stages:
        steps.append(("convert", convert_command(a, variant, out, cluster_path if a.tp > 1 else None, prov_path)))

    for stage, cmd in steps:
        if not dry and not a.force and state.done(stage, cmd):
            print(f"[{stage}] done before with the same arguments: skipped")
            continue
        if dry:
            print(f"[{stage}] would run (cwd {a.fork if stage in ('layers', 'skew') else '.'}):\n    " + " ".join(cmd))
            if stage == "memory":
                print(f"    model dir with config.json only, dummy weights; spec: {json.dumps(probe_spec(a, '<dir>'))}")
            if stage == "convert":
                print(f"    output {out}" + ("" if a.variant else " (variant name assumed bf16 until the profiler has run)"))
            continue
        if stage in ("layers", "skew"):
            work.mkdir(parents=True, exist_ok=True)
            _run(cmd, a.fork, env, work / f"{stage}.log")
            state.finish(stage, cmd)
        elif stage == "memory":
            _memory(a, gpus, work, cmd, spec_path, mem_json, env, state)
        elif stage == "convert":
            variant = find_variant(a)
            out = output_path(a, variant)
            cmd = convert_command(a, variant, out, cluster_path if a.tp > 1 else None, prov_path)
            _convert(a, gpus, work, cmd, out, cluster_path, mem_json, prov_path, env, state, variant)
    if dry:
        print("dry run: nothing executed")
        return 0
    if "convert" in stages:
        print(f"\nCommit the profile:\n  git add {out}\n  git commit -m 'Add {a.model} on {a.hardware}, TP {a.tp} profile'\n"
              "then `python -m augur.cost.profiles` lists it; --cost auto uses it.")
    return 0


def _memory(a, gpus, work, cmd, spec_path, mem_json, env, state) -> None:
    visible = len(gpus)
    if visible < a.tp:
        print(f"[memory] skipped: TP {a.tp} needs {a.tp} visible GPUs, {visible} here. Run "
              f"`python -m augur.profile --model {a.model} --hardware {a.hardware} --tp {a.tp} --stages memory` "
              "on a host with that many; the profile is built without these numbers.")
        return
    work.mkdir(parents=True, exist_ok=True)
    cfg_dir = Path(tempfile.mkdtemp(prefix="augur-probe-"))
    try:
        shutil.copyfile(fork_config_path(a.fork, a.model), cfg_dir / "config.json")
        spec_path.write_text(json.dumps(probe_spec(a, str(cfg_dir))))
        penv = dict(env, VLLM_LOGGING_LEVEL="INFO")
        text = _run(cmd, ".", penv, work / "memory.log")
    finally:
        shutil.rmtree(cfg_dir, ignore_errors=True)
    mem = parse_vllm_memory(text)
    need = {"weights_gib", "consumed_gib", "peak_activation_gib", "cuda_graph_gib", "available_kv_gib"}
    if not need <= set(mem):
        raise ProfileError(f"could not read {sorted(need - set(mem))} from vLLM's startup log ({work / 'memory.log'}); "
                           "the log format may have changed")
    mem.update({"tp": a.tp, "gpu_memory_utilization": a.gpu_memory_utilization, "max_model_len": a.max_model_len,
                "max_num_batched_tokens": a.max_batch_tokens, "max_num_seqs": a.max_num_seqs,
                "source": "vLLM startup log, dummy weights, no tokenizer"})
    mem_json.write_text(json.dumps(mem, indent=1, sort_keys=True))
    print(f"[memory] {json.dumps(mem, sort_keys=True)}")
    state.finish("memory", cmd)


def provenance(a, gpus, mem_json: Path, work: Path) -> dict:
    """What the profile records about how it was measured (the converter adds the fork commit and CSV hashes)."""
    prov: dict = {"augur_profile": {
        "command": " ".join(sys.argv),
        "max_kv": a.max_kv, "max_batch_tokens": a.max_batch_tokens, "max_num_seqs": a.max_num_seqs,
        "max_model_len": a.max_model_len, "gpu_memory_utilization": a.gpu_memory_utilization,
        "skew_sweep": not a.no_skew and (work / "skew.done.json").is_file(),
        "finished": time.strftime("%Y-%m-%d"),
    }}
    if gpus:
        g = gpus[0]
        prov["augur_profile"].update({"gpu": g["name"], "gpu_memory_gib": g["memory_gib"], "driver": g["driver"]})
    if mem_json.is_file():
        prov["vllm_memory"] = json.loads(mem_json.read_text())
    return prov


def _convert(a, gpus, work, cmd, out, cluster_path, mem_json, prov_path, env, state, variant) -> None:
    work.mkdir(parents=True, exist_ok=True)
    if a.tp > 1 and not a.cluster_config:
        cluster_path.write_text(json.dumps(cluster_config(a), indent=2))
    prov_path.write_text(json.dumps(provenance(a, gpus, mem_json, work), indent=1, sort_keys=True))
    _run(cmd, ".", env, work / "convert.log")  # exit code != 0 when the self-check finds a mismatch
    state.finish("convert", cmd, output=str(out))


if __name__ == "__main__":
    raise SystemExit(main())
