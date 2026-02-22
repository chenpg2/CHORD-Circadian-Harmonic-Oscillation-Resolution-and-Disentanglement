"""
CLI entry point for CHORD.

Registered as the ``chord`` console script in *pyproject.toml*.
"""

from __future__ import annotations

import sys


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="chord",
        description="CHORD: Circadian Harmonic Oscillation Resolver & Disentangler",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ---- chord detect ----------------------------------------------------
    detect_parser = subparsers.add_parser("detect", help="Detect rhythmic genes")
    detect_parser.add_argument(
        "input",
        help="CSV file: first column = gene names, remaining columns = expression values",
    )
    detect_parser.add_argument("-o", "--output", default="chord_results.csv",
                               help="Output CSV path (default: chord_results.csv)")
    detect_parser.add_argument("-t", "--timepoints",
                               help="Comma-separated timepoints in hours (e.g. 0,2,4,...,46)")
    detect_parser.add_argument("-m", "--method",
                               choices=["bhdt", "pinod", "both", "auto"], default="auto",
                               help="Detection method (default: auto)")
    detect_parser.add_argument("--device", default="cpu",
                               help="PyTorch device for PINOD (default: cpu)")
    detect_parser.add_argument("-j", "--jobs", type=int, default=1,
                               help="Parallel jobs for BHDT (default: 1)")
    detect_parser.add_argument("-v", "--verbose", action="store_true",
                               help="Print progress information")

    # ---- chord benchmark -------------------------------------------------
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("-n", "--n-genes", type=int, default=100,
                              help="Number of genes per scenario (default: 100)")
    bench_parser.add_argument("-o", "--output", default="chord_benchmark.csv",
                              help="Output CSV path (default: chord_benchmark.csv)")
    bench_parser.add_argument("-v", "--verbose", action="store_true")
    bench_parser.add_argument("--real-data", action="store_true",
                              help="Run real-data benchmark using public GEO "
                                   "datasets instead of synthetic data")
    bench_parser.add_argument("--datasets", type=str, nargs="*", default=None,
                              help="Datasets for real-data mode (default: all). "
                                   "Options: hughes2009, zhu2023_wt, zhu2023_ko, "
                                   "mure2018")
    bench_parser.add_argument("--cache-dir", type=str,
                              default="~/.chord_cache",
                              help="Cache directory for GEO data "
                                   "(default: ~/.chord_cache)")

    # ---- chord version ---------------------------------------------------
    subparsers.add_parser("version", help="Show version")

    # ---- dispatch --------------------------------------------------------
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "version":
        _cmd_version()
    elif args.command == "detect":
        _cmd_detect(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


# ======================================================================= #
#  Subcommand implementations                                              #
# ======================================================================= #

def _cmd_version() -> None:
    from chord import __version__
    print(f"chord-rhythm {__version__}")


def _cmd_detect(args) -> None:
    import numpy as np
    import pandas as pd
    from chord.pipeline import run

    # --- read input CSV ---------------------------------------------------
    df_in = pd.read_csv(args.input, index_col=0)
    gene_names = df_in.index.tolist()
    expr = df_in.values.astype(np.float64)

    # --- parse timepoints -------------------------------------------------
    if args.timepoints is not None:
        timepoints = np.array([float(x) for x in args.timepoints.split(",")],
                              dtype=np.float64)
    else:
        # Assume columns are numeric timepoints
        try:
            timepoints = np.array([float(c) for c in df_in.columns], dtype=np.float64)
        except ValueError:
            print(
                "Error: cannot infer timepoints from column headers. "
                "Please supply -t/--timepoints explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

    if len(timepoints) != expr.shape[1]:
        print(
            f"Error: {len(timepoints)} timepoints but expression matrix has "
            f"{expr.shape[1]} columns.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- run pipeline -----------------------------------------------------
    result = run(
        expr,
        timepoints,
        gene_names=gene_names,
        methods=args.method,
        pinod_device=args.device,
        n_jobs=args.jobs,
        verbose=args.verbose,
    )

    result.to_csv(args.output, index=False)
    print(f"Results written to {args.output}  ({len(result)} genes)")


def _cmd_benchmark(args) -> None:
    import numpy as np
    import pandas as pd

    if args.real_data:
        _cmd_benchmark_real(args)
        return

    from chord.simulation.generator import (
        pure_circadian,
        independent_superposition,
        sawtooth_harmonic,
    )
    from chord.pipeline import run

    n = args.n_genes
    verbose = args.verbose

    if verbose:
        print(f"Generating synthetic benchmark ({n} genes per scenario) ...")

    rng = np.random.RandomState(42)
    all_expr = []
    all_names = []
    all_truth = []

    scenarios = [
        ("circadian", pure_circadian, "circadian_only"),
        ("independent_superposition", independent_superposition, "independent_ultradian"),
        ("sawtooth_harmonic", sawtooth_harmonic, "harmonic"),
    ]

    for label, gen_fn, truth_cls in scenarios:
        for i in range(n):
            result = gen_fn(seed=rng.randint(0, 2**31))
            all_expr.append(result["y"])
            all_names.append(f"{label}_{i}")
            all_truth.append(truth_cls)

    t = pure_circadian()["t"]
    expr = np.vstack(all_expr)

    if verbose:
        print(f"Running CHORD on {expr.shape[0]} synthetic genes ...")

    df = run(expr, t, gene_names=all_names, methods="auto", verbose=verbose)
    df["truth"] = all_truth

    # Accuracy
    if "classification" in df.columns:
        correct = (df["classification"] == df["truth"]).sum()
        total = len(df)
        print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    df.to_csv(args.output, index=False)
    print(f"Benchmark results written to {args.output}")


def _cmd_benchmark_real(args) -> None:
    """Run real-data benchmark using public GEO datasets."""
    from chord.benchmarks.run_benchmark import (
        run_real_data_benchmark,
        summarize_real_data_benchmark,
        _print_real_data_summary,
    )

    verbose = args.verbose

    if verbose:
        print("Running real-data benchmark on public GEO datasets ...")

    results = run_real_data_benchmark(
        datasets=args.datasets,
        cache_dir=args.cache_dir,
        verbose=verbose,
    )

    if len(results) == 0:
        print("No results collected. Check that GEO data is accessible.")
        return

    summary = summarize_real_data_benchmark(results)
    _print_real_data_summary(summary)

    results.to_csv(args.output, index=False)
    print(f"\nReal-data benchmark results written to {args.output}")


if __name__ == "__main__":
    main()
