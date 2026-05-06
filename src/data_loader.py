"""
Download, load, and profile all datasets.

Usage:
    python -m src.data_loader --download-all
    python -m src.data_loader --profile
    python -m src.data_loader --adversarial
    python -m src.data_loader --dataset heart_disease
"""
import json
import argparse
import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq

from src.utils import DATA_RAW, DATA_ADVERSARIAL, load_datasets_config, log, save_json, ensure_dir


def load_heart_disease(forced_redownload=False):
    """UCI Heart Disease — binary classification, ~303 rows."""
    path = DATA_RAW / "heart_disease.csv"
    if forced_redownload or not path.exists():
        log("Downloading Heart Disease dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
        df = pd.read_csv(url, names=cols, na_values="?")
        df["target"] = (df["target"] > 0).astype(int)
        ensure_dir(DATA_RAW)
        df.to_csv(path, index=False)
        log(f"Saved: {path} ({len(df)} rows)")
    else:
        df = pd.read_csv(path)
    return df, {"name": "heart_disease", "rows": len(df), "cols": len(df.columns),
                "modality": "tabular", "target": "target"}


def load_nyc_taxi(forced_redownload=False, n_rows=50_000):
    """NYC Yellow Taxi — regression, sampled to n_rows."""
    path = DATA_RAW / "nyc_taxi_sample.parquet"
    if forced_redownload or not path.exists():
        log("Downloading NYC Taxi dataset (this may take a minute)...")
        url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
        raw_path = DATA_RAW / "yellow_tripdata_2024-01.parquet"
        ensure_dir(DATA_RAW)
        import urllib.request
        urllib.request.urlretrieve(url, raw_path)
        df = next(pq.ParquetFile(raw_path).iter_batches(batch_size=n_rows)).to_pandas()
        df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 200) & (df["trip_distance"] > 0)]
        df = df.sample(n=min(n_rows, len(df)), random_state=42)
        df.to_parquet(path, index=False)
        log(f"Saved: {path} ({len(df)} rows)")
    else:
        df = pd.read_parquet(path)
    return df, {"name": "nyc_taxi", "rows": len(df), "cols": len(df.columns),
                "modality": "tabular", "target": "fare_amount"}


def load_air_quality(forced_redownload=False):
    """OpenAQ v3 PM2.5 time series for Los Angeles. Requires OPENAQ_API_KEY in environment/.env"""
    path = DATA_RAW / "air_quality.csv"
    if forced_redownload or not path.exists():
        log("Downloading Air Quality data from OpenAQ v3 API...")
        import requests
        ensure_dir(DATA_RAW)

        api_key = os.environ.get("OPENAQ_API_KEY", "")
        headers = {"Accept": "application/json", "X-API-Key": api_key}
        all_rows = []

        # Find PM2.5 sensor locations in LA bounding box
        locations_url = (
            "https://api.openaq.org/v3/locations"
            "?bbox=-118.668153,33.703935,-118.155358,34.337306"
            "&parameters_id=2&limit=10"
        )
        try:
            resp = requests.get(locations_url, timeout=30, headers=headers)
            locations = resp.json().get("results", [])
        except Exception as e:
            log(f"WARNING: OpenAQ v3 locations fetch failed: {e}")
            locations = []

        # Fetch hourly measurements for each PM2.5 sensor
        for loc in locations:
            if len(all_rows) >= 2000:
                break
            for sensor in loc.get("sensors", []):
                if sensor.get("parameter", {}).get("name") != "pm25":
                    continue
                sensor_id = sensor["id"]
                meas_url = (
                    f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
                    f"?date_from=2024-01-01T00:00:00Z&date_to=2024-04-01T00:00:00Z&limit=1000"
                )
                try:
                    r = requests.get(meas_url, timeout=30, headers=headers)
                    for m in r.json().get("results", []):
                        all_rows.append({
                            "datetime":  m.get("period", {}).get("datetimeFrom", {}).get("utc"),
                            "value":     m.get("value"),
                            "unit":      "µg/m³",
                            "location":  loc.get("name"),
                            "latitude":  loc.get("coordinates", {}).get("latitude"),
                            "longitude": loc.get("coordinates", {}).get("longitude"),
                        })
                except Exception as e:
                    log(f"WARNING: sensor {sensor_id} fetch failed: {e}")

        df = pd.DataFrame(all_rows)
        if len(df) == 0:
            log("WARNING: OpenAQ API returned no data, generating synthetic PM2.5 data")
            dates = pd.date_range("2024-01-01", periods=2000, freq="h")
            np.random.seed(42)
            values = 30 + 15 * np.sin(np.arange(2000) * 2 * np.pi / 168) + np.random.normal(0, 5, 2000)
            df = pd.DataFrame({"datetime": dates, "value": values, "unit": "µg/m³",
                               "location": "Synthetic-LA"})
        df.to_csv(path, index=False)
        log(f"Saved: {path} ({len(df)} rows)")
    else:
        df = pd.read_csv(path)
    return df, {"name": "air_quality", "rows": len(df), "cols": len(df.columns),
                "modality": "time_series", "target": "value"}

def load_amazon_reviews(forced_redownload=False, n_rows=10_000):
    """Amazon Reviews — NLP sentiment classification."""
    path = DATA_RAW / "amazon_reviews.csv"
    if forced_redownload or not path.exists():
        log("Downloading Amazon Reviews from McAuley Lab...")
        ensure_dir(DATA_RAW)
        try:
            import urllib.request
            url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl"
            jsonl_path = DATA_RAW / "All_Beauty.jsonl"
            urllib.request.urlretrieve(url, jsonl_path)
            df = pd.read_json(jsonl_path, lines=True, nrows=n_rows)
            df = df[["rating", "text", "verified_purchase", "title"]].head(n_rows)
            jsonl_path.unlink()
        except Exception as e:
            log(f"WARNING: direct download failed: {e} — generating synthetic reviews")
            np.random.seed(42)
            ratings = np.random.choice([1, 2, 3, 4, 5], size=n_rows, p=[0.05, 0.05, 0.10, 0.30, 0.50])
            texts = [f"This is a sample review with rating {r}. " * np.random.randint(3, 15) for r in ratings]
            df = pd.DataFrame({"rating": ratings, "text": texts, "verified_purchase": True})
        df.to_csv(path, index=False)
        log(f"Saved: {path} ({len(df)} rows)")
    else:
        df = pd.read_csv(path)
    return df, {"name": "amazon_reviews", "rows": len(df), "cols": len(df.columns),
                "modality": "nlp", "target": "rating"}

def load_cifar10(forced_redownload=False):
    """CIFAR-10 — image classification (HuggingFace, 5000 test samples)."""
    meta = {"name": "cifar10", "rows": 5000, "modality": "image", "target": "label"}
    if not forced_redownload:
        return None, meta
    log("Loading CIFAR-10 from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("cifar10", split="test[:5000]")
        return ds, meta
    except Exception as e:
        log(f"WARNING: CIFAR-10 load failed: {e}")
        return None, meta


def load_urbansound8k(forced_redownload=False):
    """UrbanSound8K — audio classification. Requires manual download from Zenodo."""
    log("WARNING: UrbanSound8K requires manual download from https://zenodo.org/record/1203745")
    return None, {"name": "urbansound8k", "rows": 8732, "modality": "audio", "target": "classID"}


LOADERS = {
    "heart_disease":  {"func": load_heart_disease,  "type": "tabular"},
    "nyc_taxi":       {"func": load_nyc_taxi,        "type": "tabular"},
    "air_quality":    {"func": load_air_quality,     "type": "time_series"},
    "amazon_reviews": {"func": load_amazon_reviews,  "type": "nlp"},
    "cifar10":        {"func": load_cifar10,         "type": "image"},
    "urbansound8k":   {"func": load_urbansound8k,    "type": "audio"},
}


def load_dataset_by_name(name, forced_redownload=False):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADERS.keys())}")
    return LOADERS[name]["func"](forced_redownload=forced_redownload)


def flip_labels(df, target_col, flip_pct=0.20, seed=42):
    """Randomly flip flip_pct of target labels for adversarial robustness testing."""
    rng = np.random.RandomState(seed)
    df_adv = df.copy()
    flip_idx = rng.choice(df.index, size=int(len(df) * flip_pct), replace=False)
    unique_labels = df[target_col].unique()
    for idx in flip_idx:
        others = [l for l in unique_labels if l != df_adv.loc[idx, target_col]]
        if others:
            df_adv.loc[idx, target_col] = rng.choice(others)
    return df_adv


def inject_missing(df, target_col, missing_pct=0.15, seed=42):
    """Randomly set missing_pct of feature values to NaN, leaving the target column intact."""
    rng = np.random.RandomState(seed)
    df_adv = df.copy()
    mask = rng.random(df.shape) < missing_pct
    mask[:, df.columns.get_loc(target_col)] = False
    return df_adv.mask(mask)


def create_adversarial_datasets():
    """Generate corrupted versions of all tabular datasets for D6 robustness testing."""
    ensure_dir(DATA_ADVERSARIAL)

    df_hd, _ = load_heart_disease()
    df_hd_adv = inject_missing(flip_labels(df_hd, "target", flip_pct=0.20), target_col="target", missing_pct=0.15)
    df_hd_adv.to_csv(DATA_ADVERSARIAL / "heart_disease.csv", index=False)
    log(f"Created adversarial heart_disease: {len(df_hd_adv)} rows, 20% flipped, 15% missing")

    df_taxi, _ = load_nyc_taxi()
    df_taxi_adv = inject_missing(df_taxi, target_col="fare_amount", missing_pct=0.20)
    df_taxi_adv.to_parquet(DATA_ADVERSARIAL / "nyc_taxi_sample.parquet", index=False)
    log(f"Created adversarial nyc_taxi: {len(df_taxi_adv)} rows, 20% missing")

    df_aq, _ = load_air_quality()
    df_aq_adv = inject_missing(df_aq, target_col="value", missing_pct=0.20)
    df_aq_adv.to_csv(DATA_ADVERSARIAL / "air_quality.csv", index=False)
    log(f"Created adversarial air_quality: {len(df_aq_adv)} rows, 20% missing")

    df_ar, _ = load_amazon_reviews()
    df_ar_adv = flip_labels(df_ar, "rating", flip_pct=0.15)
    df_ar_adv.to_csv(DATA_ADVERSARIAL / "amazon_reviews.csv", index=False)
    log(f"Created adversarial amazon_reviews: {len(df_ar_adv)} rows, 15% flipped")


def profile_all_datasets(forced_redownload=False):
    """Download and profile all datasets. Saves profiles to data/dataset_profiles.json."""
    profiles = {}
    for name, loader in LOADERS.items():
        log(f"Profiling {name}...")
        try:
            data, meta = loader["func"](forced_redownload=forced_redownload)
            if isinstance(data, pd.DataFrame):
                meta["missing_total"] = int(data.isnull().sum().sum())
                meta["missing_pct"]   = round(data.isnull().sum().sum() / data.size * 100, 2)
                meta["memory_mb"]     = round(data.memory_usage(deep=True).sum() / 1e6, 2)
                meta["dtypes"]        = dict(data.dtypes.astype(str).value_counts())
            profiles[name] = meta
            log(f"{name}: {meta.get('rows', '?')} rows")
        except Exception as e:
            log(f"WARNING: {name} failed: {e}")
            profiles[name] = {"name": name, "error": str(e)}

    save_json(profiles, DATA_RAW.parent / "dataset_profiles.json")
    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PrismBench Data Loader")
    parser.add_argument("--download-all", action="store_true", help="Download and profile all datasets")
    parser.add_argument("--profile",      action="store_true", help="Profile already-downloaded datasets")
    parser.add_argument("--adversarial",  action="store_true", help="Create adversarial variants")
    parser.add_argument("--dataset",      type=str,            help="Download a specific dataset by name")
    args = parser.parse_args()

    if args.download_all:
        profiles = profile_all_datasets(forced_redownload=True)
        print(json.dumps(profiles, indent=2, default=str))
    elif args.profile:
        profiles = profile_all_datasets(forced_redownload=False)
        print(json.dumps(profiles, indent=2, default=str))

    if args.adversarial:
        create_adversarial_datasets()

    if args.dataset:
        df, meta = load_dataset_by_name(args.dataset, forced_redownload=True)
        print(json.dumps(meta, indent=2, default=str))
        if isinstance(df, pd.DataFrame):
            print(f"\n{df.head()}")