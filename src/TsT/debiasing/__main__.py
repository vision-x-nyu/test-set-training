"""
Command-line interface for Iterative Bias Pruning (IBP).

Run with: python -m TsT.debiasing --benchmark vsi --alloc per_format --budget 1000 --output_dir DIR
"""

import argparse
import logging
import sys
from pathlib import Path

EPILOG = """\
allocation (--alloc):
  global      one group, --budget questions
  per_format  one group per answer format (mc, num); --budget split in proportion to group size,
              or explicit --budgets mc=100,num=100
  per_type    one group per question type; --frac F removes round(F * n) from every type,
              --budgets_from FILE uses the per-type counts of the ids in FILE (one id per line), or
              explicit --budgets TYPE=N,TYPE=N

examples:
  python -m TsT.debiasing --benchmark vsi --alloc per_format --budget 1000 --output_dir outputs/ibp_pf_b1000
  python -m TsT.debiasing --benchmark vsi --alloc per_type --frac 0.54 --output_dir outputs/ibp_pt_uniform54
  python -m TsT.debiasing --benchmark vsi --alloc per_type \\
      --budgets_from reproduce/data/vsi_bench_debiased_v1_removed_ids.txt --output_dir outputs/ibp_pt_v1budgets
  python -m TsT.debiasing --benchmark mmmu --mode llm --alloc global --budget 100 --batch_size 20 \\
      --early_stop 0.05 --output_dir outputs/ibp_mmmu   # TsT-LLM; one GPU, `llm` extra

outputs (in --output_dir): removed_ids.txt and kept_ids.txt (one question id per line) and summary.json
(settings, dataset revision and row-order fingerprint, per-group budgets and per-iteration bias traces).

exit status: 0 on success; 1 if TsT failed or the loaded data do not match the requested dataset
revision; 2 on invalid arguments.
"""


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m TsT.debiasing",
        description=(
            "Iterative Bias Pruning (IBP): repeatedly run TsT on a benchmark and remove the questions with the "
            "highest held-out scores (the most exploitable without the image), batch by batch."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", "-b", required=True, help="Benchmark name (vsi, cvb, mmmu, video_mme, mmstar)")
    parser.add_argument("--mode", choices=["rf", "llm"], default="rf", help="TsT-RF (CPU; default) or TsT-LLM (GPU)")
    parser.add_argument(
        "--output_dir", "-o", required=True, help="Directory for removed_ids.txt, kept_ids.txt, summary.json"
    )

    a = parser.add_argument_group("budget")
    a.add_argument("--alloc", choices=["global", "per_format", "per_type"], default="per_format")
    a.add_argument("--budget", type=int, default=None, help="Total questions to remove (global, per_format)")
    a.add_argument("--frac", type=float, default=None, help="Fraction of every question type to remove (per_type)")
    a.add_argument(
        "--budgets_from",
        default=None,
        help="File of question ids (one per line); its per-type counts become the per_type budgets",
    )
    a.add_argument(
        "--budgets", default=None, help="Explicit per-group budgets, e.g. mc=100,num=100 (per_format, per_type)"
    )
    a.add_argument("--batch_size", type=int, default=50, help="Questions removed per iteration (default: 50)")
    a.add_argument(
        "--early_stop", type=float, default=None, help="Stop a group when its max bias score is at or below this"
    )
    a.add_argument(
        "--min_per_group", type=int, default=None, help="Override the benchmark's group floor (VSI-Bench: 10 per type)"
    )

    r = parser.add_argument_group("ranking")
    r.add_argument(
        "--score",
        choices=["s", "delta"],
        default=None,
        help="Rank by the held-out score s(x) (default for rf) or by s(x) minus zero-shot (delta; default for llm)",
    )
    r.add_argument(
        "--tie_break",
        choices=["id", "legacy"],
        default="id",
        help="Order of tied scores: by question id (default) or the proceedings' unstable sort (legacy; parity "
        "only, and exact only on x86-64 CPUs where numpy uses its AVX-512 sort)",
    )

    t = parser.add_argument_group("TsT settings (passed to every TsT run)")
    t.add_argument("--revision", default=None, help="Dataset revision (default: the benchmark's pinned revision)")
    t.add_argument("--feature_set", default="default", help="TsT-RF feature preset (default: leak-free 'default')")
    t.add_argument("--n_splits", "-k", type=int, default=5)
    t.add_argument("--repeats", type=int, default=1)
    t.add_argument("--random_state", "-s", type=int, default=42)
    t.add_argument("--fold_seed", type=int, default=None)
    t.add_argument("--group_col", default=None, help="Grouped folds by this column")
    t.add_argument("--target_col", default=None)
    t.add_argument("--verbose", "-v", action="store_true", help="Log every iteration")

    llm = parser.add_argument_group("TsT-LLM (--mode llm; defaults are the paper's settings, App. C.1)")
    llm.add_argument("--llm_model", default="Qwen/Qwen2-7B-Instruct")
    llm.add_argument("--llm_model_revision", default=None, help="default: the paper's revision for Qwen2-7B-Instruct")
    llm.add_argument("--llm_train_batch_size", type=int, default=4)
    llm.add_argument("--llm_eval_batch_size", type=int, default=4)
    llm.add_argument("--llm_epochs", type=int, default=1)
    return parser


class _PrefixFormatter(logging.Formatter):
    """Plain INFO messages; 'WARNING: ' / 'ERROR: ' prefixes otherwise (unless already present)."""

    def format(self, record):
        msg = super().format(record)
        if record.levelno >= logging.WARNING and not msg.lower().startswith(record.levelname.lower()):
            msg = f"{record.levelname}: {msg}"
        return msg


class _OncePerMessage(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        if record.levelno < logging.WARNING:
            return True
        key = (record.levelno, record.getMessage())
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


def _fmt(x, spec=".3f"):
    return "" if x is None else format(x, spec)


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_PrefixFormatter("%(message)s"))
    handler.addFilter(_OncePerMessage())  # TsT runs once per iteration; show each warning once
    tst_logger = logging.getLogger("TsT")
    tst_logger.handlers = [handler]
    tst_logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
    tst_logger.propagate = False

    given = [name for name in ("budget", "frac", "budgets_from", "budgets") if getattr(args, name) is not None]
    allowed = {
        "global": ["budget"],
        "per_format": ["budget", "budgets"],
        "per_type": ["frac", "budgets_from", "budgets"],
    }
    if len(given) != 1 or given[0] not in allowed[args.alloc]:
        parser.error(f"--alloc {args.alloc} needs exactly one of " + ", ".join(f"--{a}" for a in allowed[args.alloc]))
    explicit = None
    if args.budgets is not None:
        try:
            explicit = {
                k.strip(): int(v) for k, v in (item.split("=") for item in args.budgets.split(",") if item.strip())
            }
        except ValueError:
            parser.error("--budgets must look like mc=100,num=100")
        if not explicit or any(v < 0 for v in explicit.values()) or not any(explicit.values()):
            parser.error("--budgets needs at least one positive budget and no negative ones")
    if args.budgets_from is not None and not Path(args.budgets_from).is_file():
        parser.error(f"--budgets_from: no such file: {args.budgets_from}")
    if args.min_per_group is not None and args.min_per_group < 0:
        parser.error("--min_per_group must be >= 0")
    if args.early_stop is not None and not (args.early_stop == args.early_stop):  # NaN
        parser.error("--early_stop must be a number")
    if args.n_splits < 2 or args.repeats < 1 or args.batch_size < 1:
        parser.error("--n_splits must be >= 2, --repeats >= 1 and --batch_size >= 1")
    if args.mode == "rf" and args.score == "delta":
        parser.error("--score delta needs zero-shot scores, which only TsT-LLM has (--mode llm)")
    if args.mode == "llm" and args.feature_set != "default":
        parser.error("--feature_set applies to TsT-RF only")

    from ..core.benchmark import BenchmarkRegistry, DatasetRevisionError
    from ..evaluation import ModelEvaluationError
    from .strategies import MissingScoreError
    from .allocation import budgets_from_ids
    from .ibp import debias_benchmark, read_id_list

    try:
        bench = BenchmarkRegistry.get_benchmark(args.benchmark)
        bench.get_ibp_strategy()
        if args.mode == "rf" and not bench.supports_rf:
            parser.error(f"TsT-RF is not available for benchmark '{bench.name}'; use --mode llm")
        if args.mode == "rf":
            bench.check_feature_set(args.feature_set)
    except (ValueError, NotImplementedError) as e:
        parser.error(str(e))

    from ..evaluation import LOCKED_VERSIONS, version_mismatches

    mismatches = version_mismatches()
    if mismatches:
        got = ", ".join(f"{pkg} {v['installed']} (locked: {v['locked']})" for pkg, v in mismatches.items())
        print(
            f"WARNING: {got}. The published selections were computed with "
            + ", ".join(f"{p} {v}" for p, v in LOCKED_VERSIONS.items())
            + "; other versions can change the scores and so the removed questions.",
            file=sys.stderr,
        )

    llm_config = None
    if args.mode == "llm":
        from ..evaluators.llm import LLMRunConfig, missing_requirements

        problem = missing_requirements()
        if problem:
            print(f"error: {problem}.", file=sys.stderr)
            return 2

        llm_config = LLMRunConfig(
            model_name=args.llm_model,
            model_revision=args.llm_model_revision,
            train_batch_size=args.llm_train_batch_size,
            eval_batch_size=args.llm_eval_batch_size,
            num_epochs=args.llm_epochs,
        )

    revision = bench.resolve_revision(args.revision, mode=args.mode)
    try:
        df = bench.load_data(revision=revision)
    except DatasetRevisionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    budgets = explicit
    source = "--budgets" if explicit is not None else None
    if args.budgets_from:
        try:
            budgets = budgets_from_ids(df, read_id_list(args.budgets_from), bench.id_col)
        except ValueError as e:
            parser.error(f"--budgets_from: {e}")
        source = Path(args.budgets_from).name

    rev = revision[:7] if revision and len(revision) == 40 else revision
    if args.budget is not None:
        plan = f"budget {args.budget}"
    elif args.frac is not None:
        plan = f"frac {args.frac}"
    else:
        plan = f"budgets {budgets}" if explicit is not None else f"budgets from {source}"
    print(
        f"IBP | TsT-{args.mode.upper()} | {bench.name} ({bench.hf_repo} @ {rev}) | alloc {args.alloc}, {plan} | "
        f"batch {args.batch_size} | score {args.score or ('s' if args.mode == 'rf' else 'delta')} | "
        f"tie-break {args.tie_break} | seed {args.random_state}",
        flush=True,
    )
    print(
        f"{'group':30} {'n':>5} {'budget':>7} {'removed':>8} {'iters':>6} {'max_bias0':>10} {'mean_bias_last':>15}",
        flush=True,
    )

    def report(g):
        s = g.summary()
        print(
            f"{s['group']:30} {s['n']:5d} {s['budget']:7d} {s['removed']:8d} {s['n_iterations']:6d} "
            f"{_fmt(s['max_bias_first']):>10} {_fmt(s['mean_bias_last_iteration']):>15}",
            flush=True,
        )

    try:
        run = debias_benchmark(
            bench,
            mode=args.mode,
            alloc=args.alloc,
            budget=args.budget,
            frac=args.frac,
            budgets=budgets,
            budgets_source=source,
            batch_size=args.batch_size,
            score=args.score,
            tie_break=args.tie_break,
            min_per_group=args.min_per_group,
            early_stop_threshold=args.early_stop,
            revision=revision,
            feature_set=args.feature_set,
            n_splits=args.n_splits,
            repeats=args.repeats,
            random_state=args.random_state,
            fold_seed=args.fold_seed,
            group_col=args.group_col,
            target_col=args.target_col,
            llm_config=llm_config,
            df=df,
            verbose=args.verbose,
            on_group_done=report,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (ModelEvaluationError, MissingScoreError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    paths = run.save(args.output_dir)
    print(
        f"removed {run.n_removed} of {run.n_rows} ({100 * run.n_removed / run.n_rows:.1f}%), kept {len(run.kept_ids)}"
    )
    print(f"wrote {paths['removed_ids']}, {paths['kept_ids']} and {paths['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
