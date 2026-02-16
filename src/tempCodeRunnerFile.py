def run_bayesian_dag(processed_dir):
    from features import bayesian as bayesian_module
    edges_csv = Path(processed_dir) / "hfacs_dag_edges.csv"
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_bayes = bayesian_module.load_edges_csv(str(edges_csv))
    # Save the full Bayesian edge table
    df_bayes.to_csv(out_dir / "hfacs_bayesian_dag_edges.csv", index=False)
    bayesian_module.draw_dag_pdf(
        df_bayes,
        str(out_dir / "bayesian_dag.pdf"),
        "Bayesian HFACS DAG (thickness=weight)",
    )

    # Prune using the function from bayesian.py
    pruned = bayesian_module.threshold_prune_edges(
        df_bayes,
        alpha=1.0,
        beta=1.0,
        min_keep_score=-3.0,
    )
    pruned.to_csv(out_dir / "hfacs_bayesian_dag_edges_threshold_pruned.csv", index=False)
    bayesian_module.draw_dag_pdf(
        pruned,
        str(out_dir / "bayesian_dag_threshold_pruned.pdf"),
        "Bayesian HFACS DAG (after threshold pruning)",
    )


# HFACS chain function
# from models.prediction import compute_full_chain


def read_csv_robust(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] pandas read_csv failed for {csv_path}: {e}")
        print("[INFO] Retrying with Python csv parser fallback...")
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return pd.D