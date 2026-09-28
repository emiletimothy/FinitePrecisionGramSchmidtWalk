"""Gram-Schmidt walk stability experiments.

    python run_experiments.py --smoke          # n=50, R=200, then a timing projection
    python run_experiments.py                  # full grid (resumable)
    python run_experiments.py --report-only    # rebuild table and figures from raw/
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from pathlib import Path

from gsw_stability import report, runner

HERE = Path(__file__).resolve().parent


def int_map(s):
    return {int(k): int(v) for k, v in (p.split(":") for p in s.split(","))} if s else {}


def csv_list(s):
    return [x for x in s.split(",") if x]


def r_rules(s):
    """"lstsq@500=25,lstsq/float16@200=300" -> {(impl, dtype or None, n): R}"""
    rules = {}
    for p in csv_list(s):
        lhs, R = p.split("=")
        target, n = lhs.split("@")
        impl, _, dtype = target.partition("/")
        rules[(impl, dtype or None, int(n))] = int(R)
    return rules


def make_R_for(rules, R_by_n, R):
    def R_for(impl, dtype, n):
        for k in ((impl, dtype, n), (impl, None, n)):
            if k in rules:
                return rules[k]
        return R_by_n.get(n, R)
    return R_for


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="n=50, R=200, 5 diagnostic runs, into results/smoke")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--ns", type=lambda s: [int(x) for x in csv_list(s)], default=[50, 200, 500])
    ap.add_argument("--families", type=csv_list, default=list(runner.FAMILIES))
    ap.add_argument("--dtypes", type=csv_list, default=list(runner.DTYPES))
    ap.add_argument("--impls", type=csv_list, default=[], help="restrict implementations")
    ap.add_argument("--R", type=int, default=2000, help="runs per configuration")
    ap.add_argument("--R-by-n", type=int_map, default={}, help="e.g. 500:1000")
    ap.add_argument("--R-rules", type=r_rules, default=r_rules("lstsq@500=300"),
                    help='per-implementation overrides "impl[/dtype]@n=R", comma separated, '
                         'e.g. "lstsq@500=25,lstsq/float16@200=300,chol_rk10@500=200"')
    ap.add_argument("--diag-runs", type=int, default=20)
    ap.add_argument("--diag-runs-by-n", type=int_map, default={500: 5})
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estimate", action="store_true", help="only print the timing projection")
    ap.add_argument("--status", action="store_true",
                    help="estimate time left for a running grid (pass the same flags it was started with)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    full_ns = args.ns
    full_R_for = make_R_for(args.R_rules, args.R_by_n, args.R)
    if args.smoke:
        args.ns, args.R, args.R_by_n, args.R_rules = [50], 200, {}, {}
        args.diag_runs, args.diag_runs_by_n = 5, {}
    outdir = Path(args.outdir or HERE / "results" / ("smoke" if args.smoke else ""))
    R_for = make_R_for(args.R_rules, args.R_by_n, args.R)

    def diag_for(n):
        return args.diag_runs_by_n.get(n, args.diag_runs)

    if args.status:
        runner.status(runner.build_tasks(args.ns, args.families, args.dtypes, args.impls,
                                         R_for, diag_for, args.seed, outdir), args.workers)
        return

    if args.estimate:
        runner.project(args.ns, args.families, args.dtypes, args.impls, R_for, args.workers, args.seed)
        return

    if not args.report_only:
        tasks = runner.build_tasks(args.ns, args.families, args.dtypes, args.impls,
                                   R_for, diag_for, args.seed, outdir)
        runner.execute(tasks, args.workers)

    rows, md = report.write(outdir)
    print(md)
    if not args.no_plots:
        from gsw_stability import plots
        plots.make_all(outdir)

    if args.smoke:
        runner.project(full_ns, args.families, args.dtypes, args.impls, full_R_for, args.workers, args.seed)


if __name__ == "__main__":
    main()
