if __name__ == "__main__":
    import argparse

    from evaluation.evaluate import evaluate
    from evaluation.subset import EvalSubsetConfig

    parser = argparse.ArgumentParser(
        description="Compare seizure detection predictions to reference annotations."
    )
    parser.add_argument("ref", help="Root folder with reference BIDS annotations.")
    parser.add_argument("hyp", help="Root folder with hypothesis annotations.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for results.csv, onset.csv, and results.txt (default: current directory).",
    )
    parser.add_argument(
        "--eval-subset",
        type=int,
        default=None,
        help=(
            "Limit SzCORE (+ default onset) scoring scope. When unset, SzCORE uses full eval. "
            "Use with MCD subset outputs so missing hyp files are not scored as FN."
        ),
    )
    parser.add_argument(
        "--eval-subset-mode",
        default="random",
        choices=["random", "research"],
    )
    parser.add_argument("--eval-subset-seed", type=int, default=42)
    parser.add_argument(
        "--onset-subset",
        type=int,
        default=None,
        help=(
            "Limit lead/lag onset scoring scope. Default: inherit --eval-subset when set, "
            "else full eval."
        ),
    )
    parser.add_argument(
        "--onset-subset-mode",
        default="random",
        choices=["random", "research"],
    )
    parser.add_argument("--onset-subset-seed", type=int, default=42)
    args = parser.parse_args()

    eval_subset = EvalSubsetConfig(
        subset=args.eval_subset,
        subset_mode=args.eval_subset_mode,
        subset_seed=args.eval_subset_seed,
    )
    onset_subset = None
    if args.onset_subset is not None:
        onset_subset = EvalSubsetConfig(
            subset=args.onset_subset,
            subset_mode=args.onset_subset_mode,
            subset_seed=args.onset_subset_seed,
        )

    evaluate(
        args.ref,
        args.hyp,
        output_dir=args.output_dir,
        eval_subset=eval_subset,
        onset_subset=onset_subset,
    )
