
with skip_run("skip", "bayesian_hillclimb_dag") as check:
    if check():
        print("[INFO] Running GES DAG discovery with causal-learn...")
        run_hfacs_causal_learn_ges(
            config_path="./configs/config.yaml",
            data_path="./data/processed/step3_hfacs_categories.csv",
            output_dir="./data/processed",
            sink_nodes=("Error", "Violation"),
            score_func="local_score_BDeu",
        )
        print("[INFO] GES DAG discovery complete.")
