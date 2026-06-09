
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.base import clone

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path("e88186124ec611f1/dataset")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = DATA_DIR / "submission.csv"

TARGET_COL = "demand"
INDEX_COL = "Index"

# Use only these columns/features (plus target/index)
BASE_COLS = ["geohash", "day", "timestamp", "RoadType", "LargeVehicles", "Weather"]


# =============================================================================
# HELPERS
# =============================================================================
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names to a consistent form without changing meaning."""
    rename_map = {}
    for c in df.columns:
        cl = c.strip()
        if cl.lower() == "road type":
            rename_map[c] = "RoadType"
        elif cl.lower() == "large vehicle":
            rename_map[c] = "LargeVehicles"
        elif cl.lower() in {"time", "timestamp"}:
            rename_map[c] = "timestamp"
        else:
            rename_map[c] = cl
    return df.rename(columns=rename_map)


def _parse_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour/minute from timestamp/time if available."""
    out = df.copy()

    if "timestamp" not in out.columns:
        return out

    ts = out["timestamp"]

    # Try to parse as datetime first; if that fails, try HH:MM or HH:MM:SS-like strings
    parsed = pd.to_datetime(ts, errors="coerce")
    if parsed.notna().any():
        out["hour"] = parsed.dt.hour.fillna(0).astype(int)
        out["minute"] = parsed.dt.minute.fillna(0).astype(int)
    else:
        ts_str = ts.astype(str)
        parts = ts_str.str.split(":", expand=True)
        out["hour"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
        if parts.shape[1] > 1:
            out["minute"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
        else:
            out["minute"] = 0

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["day_sin"] = np.sin(2 * np.pi * out["day"] / 7.0)
    out["day_cos"] = np.cos(2 * np.pi * out["day"] / 7.0)

    out["is_weekend"] = (out["day"] >= 6).astype(int)
    out["is_weekday"] = (out["day"] < 6).astype(int)
    out["is_rush_hour"] = (((out["hour"] >= 7) & (out["hour"] <= 10)) | ((out["hour"] >= 17) & (out["hour"] <= 20))).astype(int)
    out["is_night"] = (((out["hour"] >= 21) | (out["hour"] <= 5))).astype(int)

    return out


def _fit_smoothed_target_encoding(train_series: pd.Series, y: pd.Series, smoothing: float = 20.0):
    """Return mapping dict for target encoding."""
    global_mean = y.mean()
    stats = y.groupby(train_series).agg(["mean", "count"])
    enc_map = ((stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)).to_dict()
    return enc_map, global_mean


def _apply_map(series: pd.Series, mapping: dict, default_value: float) -> pd.Series:
    return series.map(mapping).fillna(default_value).astype(float)


def _build_numeric_features(df: pd.DataFrame, enc_maps=None, fit=False, y=None):
    """
    Build a numeric-only feature matrix using only:
    geohash, day, timestamp, RoadType, LargeVehicles, Weather
    plus derived interactions/features.
    """
    data = _standardize_columns(df)
    data = _parse_time_features(data)

    # Fill basic missing values
    for col in ["geohash", "RoadType", "LargeVehicles", "Weather"]:
        if col in data.columns:
            data[col] = data[col].astype(str).fillna("missing")
    if "day" in data.columns:
        data["day"] = pd.to_numeric(data["day"], errors="coerce").fillna(0).astype(int)

    # Frequency encodings
    if fit:
        freq_maps = {}
        for col in ["geohash", "RoadType", "LargeVehicles", "Weather"]:
            freq_maps[col] = data[col].value_counts(normalize=True).to_dict()
        enc_maps = {"freq": freq_maps}
    elif enc_maps is None:
        raise ValueError("enc_maps must be provided for non-fit mode.")

    freq_maps = enc_maps["freq"]

    # Target encodings
    if fit:
        te_maps = {}
        global_mean = float(y.mean()) if y is not None else 0.0
        for col, smoothing in [("geohash", 30.0), ("RoadType", 25.0), ("LargeVehicles", 20.0), ("Weather", 25.0)]:
            te_maps[col], _ = _fit_smoothed_target_encoding(data[col], y, smoothing=smoothing)
        enc_maps["te"] = te_maps
        enc_maps["global_mean"] = global_mean
    te_maps = enc_maps["te"]
    global_mean = float(enc_maps.get("global_mean", 0.0))

    feat = pd.DataFrame(index=data.index)

    # Raw numeric/time features
    feat["day"] = data["day"].astype(float)
    feat["hour"] = data.get("hour", 0).astype(float)
    feat["minute"] = data.get("minute", 0).astype(float)
    feat["hour_sin"] = data.get("hour_sin", 0)
    feat["hour_cos"] = data.get("hour_cos", 0)
    feat["day_sin"] = data.get("day_sin", 0)
    feat["day_cos"] = data.get("day_cos", 0)
    feat["is_weekend"] = data.get("is_weekend", 0)
    feat["is_weekday"] = data.get("is_weekday", 0)
    feat["is_rush_hour"] = data.get("is_rush_hour", 0)
    feat["is_night"] = data.get("is_night", 0)

    # Frequency encodings
    feat["geohash_freq"] = _apply_map(data["geohash"], freq_maps["geohash"], 0.0)
    feat["road_freq"] = _apply_map(data["RoadType"], freq_maps["RoadType"], 0.0)
    feat["large_freq"] = _apply_map(data["LargeVehicles"], freq_maps["LargeVehicles"], 0.0)
    feat["weather_freq"] = _apply_map(data["Weather"], freq_maps["Weather"], 0.0)

    # Target encodings
    feat["geohash_te"] = _apply_map(data["geohash"], te_maps["geohash"], global_mean)
    feat["road_te"] = _apply_map(data["RoadType"], te_maps["RoadType"], global_mean)
    feat["large_te"] = _apply_map(data["LargeVehicles"], te_maps["LargeVehicles"], global_mean)
    feat["weather_te"] = _apply_map(data["Weather"], te_maps["Weather"], global_mean)

    # Simple interactions from the allowed columns only
    feat["geo_road_te"] = feat["geohash_te"] * feat["road_te"]
    feat["geo_weather_te"] = feat["geohash_te"] * feat["weather_te"]
    feat["road_weather_te"] = feat["road_te"] * feat["weather_te"]
    feat["road_large_te"] = feat["road_te"] * feat["large_te"]
    feat["time_weather"] = feat["hour_sin"] * feat["weather_te"]
    feat["time_road"] = feat["hour_sin"] * feat["road_te"]
    feat["rush_weather"] = feat["is_rush_hour"] * feat["weather_te"]

    # Missing values protection
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return feat, enc_maps


def _make_base_models():
    models = {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.85,
            random_state=42,
            n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=450,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.9,
            random_state=42,
        ),
        "hist_gb": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        ),
    }

    if LGB_AVAILABLE:
        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=42,
            n_jobs=-1,
        )

    if CB_AVAILABLE:
        models["catboost"] = cb.CatBoostRegressor(
            iterations=4000,
            learning_rate=0.02,
            depth=8,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )

    return models


def _fit_predict_single_model(model, X_train, y_train, X_test, cv):
    """Manual OOF + test prediction for stacking."""
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))

    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        oof[val_idx] = m.predict(X_val)
        test_pred += m.predict(X_test) / cv.get_n_splits()

    return oof, test_pred


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("STACKED TRAFFIC DEMAND PREDICTION (USING ONLY SELECTED COLUMNS)")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = _standardize_columns(train_df)
    test_df = _standardize_columns(test_df)

    needed_train = [INDEX_COL, TARGET_COL] + BASE_COLS
    needed_test = [INDEX_COL] + BASE_COLS

    train_df = train_df[needed_train].copy()
    test_df = test_df[needed_test].copy()

    print(f"\nLoaded train: {train_df.shape} | test: {test_df.shape}")
    print(f"Target stats -> min: {train_df[TARGET_COL].min():.4f}, max: {train_df[TARGET_COL].max():.4f}, mean: {train_df[TARGET_COL].mean():.4f}")

    y = pd.to_numeric(train_df[TARGET_COL], errors="coerce").fillna(0.0).astype(float)

    X_train, enc_maps = _build_numeric_features(train_df.drop(columns=[TARGET_COL]), fit=True, y=y)
    X_test, _ = _build_numeric_features(test_df, enc_maps=enc_maps, fit=False)

    # Ensure identical columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    print(f"\nFeature matrix: {X_train.shape[1]} columns")
    print("Features:", ", ".join(X_train.columns[:10]), "..." if X_train.shape[1] > 10 else "")

    models = _make_base_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 80)
    print("LEVEL 0: BASE MODELS")
    print("=" * 80)

    oof_predictions = []
    test_predictions = []
    model_scores = []

    for name, model in models.items():
        print(f"Training {name} ... ", end="", flush=True)
        oof, pred = _fit_predict_single_model(model, X_train, y, X_test, cv)
        score = r2_score(y, oof)
        print(f"OOF R² = {score:.5f}")
        oof_predictions.append(oof)
        test_predictions.append(pred)
        model_scores.append((name, score))

    meta_X_train = np.column_stack(oof_predictions)
    meta_X_test = np.column_stack(test_predictions)

    print("\n" + "=" * 80)
    print("LEVEL 1: META LEARNER")
    print("=" * 80)

    meta_model = Ridge(alpha=1.0, random_state=42)
    meta_model.fit(meta_X_train, y)
    meta_oof = meta_model.predict(meta_X_train)
    meta_score = r2_score(y, meta_oof)
    final_pred = meta_model.predict(meta_X_test)

    # Blend with the best base model a bit for stability
    best_base_idx = int(np.argmax([s for _, s in model_scores]))
    final_pred = 0.75 * final_pred + 0.25 * meta_X_test[:, best_base_idx]

    final_pred = np.clip(final_pred, 0, None)

    print(f"\nMeta OOF R² = {meta_score:.5f}")
    print("\nBase model scores:")
    for name, score in sorted(model_scores, key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {score:.5f}")

    submission = pd.DataFrame({
        INDEX_COL: test_df[INDEX_COL].values,
        TARGET_COL: final_pred
    })
    submission.to_csv(SUBMISSION_PATH, index=False)

    print("\n" + "=" * 80)
    print(f"Saved submission to: {SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")
    print("First 10 rows:")
    print(submission.head(10).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
