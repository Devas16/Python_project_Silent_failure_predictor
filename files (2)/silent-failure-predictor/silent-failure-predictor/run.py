"""
run.py
------
Unified CLI entry point for the Silent Failure Predictor.

Commands:
    python run.py generate        — Generate synthetic dataset
    python run.py train           — Full training pipeline
    python run.py train --skip-ae — Train without Autoencoder
    python run.py predict         — Score a single sample (demo)
    python run.py api             — Launch FastAPI server
    python run.py dashboard       — Launch Streamlit dashboard

Examples:
    python run.py generate
    python run.py train
    python run.py api
    python run.py dashboard
"""

import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def cmd_generate(args):
    """Generate (or regenerate) the synthetic dataset."""
    from src.data_generator import generate_dataset
    df = generate_dataset(save=True)
    print(f"\n✅  Dataset generated: {len(df)} rows  |  "
          f"Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)\n")


def cmd_train(args):
    """Run the full training pipeline."""
    from src.train import main as train_main
    # Inject --skip-autoencoder flag into sys.argv for train.main to pick up
    if args.skip_ae:
        sys.argv = ["train.py", "--skip-autoencoder"]
    else:
        sys.argv = ["train.py"]
    train_main()


def cmd_predict(args):
    """Demo: score a single hardcoded anomalous sample."""
    print("\n🔬  Demo prediction — simulating a cascade-failure sample …\n")

    from src.data_generator import load_dataset
    from src.preprocessor import Preprocessor
    from src.isolation_forest_model import IsolationForestDetector

    model_path  = "models/isolation_forest.pkl"
    scaler_path = "models/scaler.pkl"

    if not os.path.exists(model_path):
        print("❌  Model not found. Run `python run.py train` first.")
        sys.exit(1)

    det  = IsolationForestDetector(); det.load(model_path)
    prep = Preprocessor();            prep.load(scaler_path)

    import pandas as pd
    sample = pd.DataFrame([{
        "cpu_usage":          92.0,
        "memory_usage":       89.5,
        "response_time_ms":  1850.0,
        "disk_io_mbps":      175.0,
        "network_latency_ms": 220.0,
        "error_rate_pct":     14.2,
    }])

    X      = prep.transform(sample)
    labels, scores = det.predict(X)

    is_anom = bool(labels[0])
    score   = float(scores[0])

    print(f"  CPU Usage          : {sample['cpu_usage'].iloc[0]} %")
    print(f"  Memory Usage       : {sample['memory_usage'].iloc[0]} %")
    print(f"  Response Time      : {sample['response_time_ms'].iloc[0]} ms")
    print(f"  Disk I/O           : {sample['disk_io_mbps'].iloc[0]} MB/s")
    print(f"  Network Latency    : {sample['network_latency_ms'].iloc[0]} ms")
    print(f"  Error Rate         : {sample['error_rate_pct'].iloc[0]} %")
    print()
    print(f"  Anomaly Score      : {score:.4f}")
    print(f"  Prediction         : {'🚨 ANOMALY — Silent failure detected!' if is_anom else '✅ Normal'}")
    print()


def cmd_api(args):
    """Start the FastAPI REST server via uvicorn."""
    from src.config_loader import load_config
    cfg  = load_config()["api"]
    host = cfg.get("host", "0.0.0.0")
    port = str(cfg.get("port", 8000))

    print(f"\n🚀  Launching API at http://{host}:{port}")
    print(f"    Docs : http://{host}:{port}/docs\n")

    subprocess.run(
        ["uvicorn", "app.api:app",
         "--host", host, "--port", port, "--reload"],
        check=True,
    )


def cmd_dashboard(args):
    """Start the Streamlit dashboard."""
    print("\n🖥️   Launching Streamlit dashboard …\n")
    subprocess.run(
        ["streamlit", "run", "app/dashboard.py"],
        check=True,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "run.py",
        description = "Silent Failure Predictor — unified CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="Generate synthetic dataset")

    train_p = sub.add_parser("train", help="Train ML models")
    train_p.add_argument(
        "--skip-ae", action="store_true",
        help="Skip Autoencoder (no TensorFlow needed)"
    )

    sub.add_parser("predict", help="Demo: predict on a single sample")
    sub.add_parser("api",     help="Launch FastAPI REST server")
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    return parser


COMMANDS = {
    "generate" : cmd_generate,
    "train"    : cmd_train,
    "predict"  : cmd_predict,
    "api"      : cmd_api,
    "dashboard": cmd_dashboard,
}


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    COMMANDS[args.command](args)
