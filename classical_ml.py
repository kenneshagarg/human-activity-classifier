"""
Classical Machine Learning models for Human Activity Recognition
Baseline benchmarks: Logistic Regression, SVM (RBF), Random Forest, XGBoost
These are used as reference points for deep learning model performance
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from config import *
from evaluate import ModelEvaluator

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(seconds):
    """Format elapsed seconds as mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"

def _section(title, width=65):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def _subsection(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print("─" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# ClassicalMLModels
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalMLModels:
    """Container for classical ML baseline models with proper specifications"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.evaluator = ModelEvaluator()
        self.feature_names = None

        # Load feature names if available
        feature_names_path = COMBINED_PROCESSED_PATH / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"  [OK] Loaded {len(self.feature_names)} feature names from feature_names.json")
        else:
            print("  [WARN] feature_names.json not found — feature importance labels will be 'feat_<idx>'")

    # ── Logistic Regression ────────────────────────────────────────────────

    def train_logistic_regression(self, X_train, y_train, X_val, y_val, model_name="LR"):
        """Train Logistic Regression with one-vs-rest and L2 regularization + CV"""
        t0 = time.time()
        print(f"\n[{model_name}] Starting training...")
        print(f"  Strategy : One-vs-Rest, L2 regularization, liblinear solver")
        print(f"  Grid     : C in [0.01, 0.1, 1, 10, 100]  (5-fold CV)")
        print(f"  Train    : {X_train.shape[0]:,} samples × {X_train.shape[1]} features")

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        print(f"  [OK] Features scaled (mean=0, std=1)")

        # estimator__C because param is on the inner LR inside OneVsRestClassifier
        param_grid = {'estimator__C': [0.01, 0.1, 1, 10, 100]}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        lr = OneVsRestClassifier(
            LogisticRegression(penalty='l2', solver='lbfgs', random_state=42, max_iter=1000)
        )

        print(f"  Running GridSearchCV (5 C values × 5 folds = 25 fits)...")
        grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(X_train_scaled, y_train)

        model = grid_search.best_estimator_
        best_C = grid_search.best_params_['estimator__C']
        print(f"  [OK] Best C          : {best_C}")
        print(f"  [OK] CV accuracy     : {grid_search.best_score_:.4f}")

        val_pred     = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1       = f1_score(y_val, val_pred, average='macro')
        print(f"  [OK] Val accuracy    : {val_accuracy:.4f}")
        print(f"  [OK] Val F1 (macro)  : {val_f1:.4f}")
        print(f"  [OK] Elapsed         : {_fmt(time.time() - t0)}")

        self.models[model_name]  = model
        self.scalers[model_name] = scaler
        return model, scaler, val_accuracy, val_f1

    # ── SVM ────────────────────────────────────────────────────────────────

    def train_svm_rbf(self, X_train, y_train, X_val, y_val, model_name="SVM"):
        """Train SVM with RBF kernel and hyperparameter tuning"""
        t0 = time.time()
        print(f"\n[{model_name}] Starting training...")
        print(f"  Strategy : RBF kernel, C × γ grid (3×3 = 9 combos, 3-fold CV = 27 fits)")
        print(f"  NOTE     : SVM on 89k samples is slow — expected 60–120 min")
        print(f"  Train    : {X_train.shape[0]:,} samples × {X_train.shape[1]} features")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        print(f"  [OK] Features scaled")

        # Reduced grid — full grid is infeasible at 89k samples
        param_grid = {
            'C':     [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.1]
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        svm = SVC(
            kernel='rbf',
            random_state=42,
            probability=True
        )

        print(f"  Running GridSearchCV (27 fits)... [this will take a while]")
        grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        model       = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"  [OK] Best C          : {best_params['C']}")
        print(f"  [OK] Best gamma      : {best_params['gamma']}")
        print(f"  [OK] CV accuracy     : {grid_search.best_score_:.4f}")

        val_pred     = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1       = f1_score(y_val, val_pred, average='macro')
        print(f"  [OK] Val accuracy    : {val_accuracy:.4f}")
        print(f"  [OK] Val F1 (macro)  : {val_f1:.4f}")
        print(f"  [OK] Elapsed         : {_fmt(time.time() - t0)}")

        self.models[model_name]  = model
        self.scalers[model_name] = scaler
        return model, scaler, val_accuracy, val_f1

    # ── Random Forest ──────────────────────────────────────────────────────

    def train_random_forest(self, X_train, y_train, X_val, y_val, model_name="RF"):
        """Train Random Forest with bootstrap sampling and √d features per split"""
        t0 = time.time()
        n_features   = X_train.shape[1]
        max_features = int(np.sqrt(n_features))

        print(f"\n[{model_name}] Starting training...")
        print(f"  Strategy : 100 trees, bootstrap=True, max_features=√{n_features}={max_features}")
        print(f"  Train    : {X_train.shape[0]:,} samples × {n_features} features")

        rf = RandomForestClassifier(
            n_estimators=100,
            max_features=max_features,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

        print(f"  Fitting Random Forest...")
        rf.fit(X_train, y_train)
        print(f"  [OK] Training complete")

        val_pred     = rf.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1       = f1_score(y_val, val_pred, average='macro')
        print(f"  [OK] Val accuracy    : {val_accuracy:.4f}")
        print(f"  [OK] Val F1 (macro)  : {val_f1:.4f}")

        # Feature importance (Gini)
        feature_importance = rf.feature_importances_
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        print(f"  Top 10 features by Gini importance:")
        for rank, idx in enumerate(top_indices):
            name = self.feature_names[idx] if self.feature_names else f"feat_{idx}"
            print(f"    {rank+1:2d}. {name}: {feature_importance[idx]:.4f}")

        print(f"  [OK] Elapsed         : {_fmt(time.time() - t0)}")

        self.models[model_name]  = rf
        self.scalers[model_name] = None
        return rf, None, val_accuracy, val_f1

    # ── XGBoost ────────────────────────────────────────────────────────────

    def train_xgboost(self, X_train, y_train, X_val, y_val, model_name="XGBoost"):
        """Train XGBoost for upper-bound comparison with hand-crafted features"""
        t0 = time.time()
        print(f"\n[{model_name}] Starting training...")
        print(f"  Strategy : Gradient boosting, 3×3×3 grid × 5-fold CV = 135 fits")
        print(f"  Train    : {X_train.shape[0]:,} samples × {X_train.shape[1]} features")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        print(f"  [OK] Features scaled")

        param_grid = {
            'n_estimators':  [50, 100, 200],
            'max_depth':     [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # n_jobs=1 inside XGBClassifier to avoid thread over-subscription with GridSearchCV n_jobs=-1
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=N_CLASSES,
            random_state=42,
            n_jobs=1,             # <-- intentionally 1; GridSearchCV handles outer parallelism
            eval_metric='mlogloss'
        )

        print(f"  Running GridSearchCV (135 fits)... [may take 15–30 min]")
        grid_search = GridSearchCV(xgb_model, param_grid, cv=cv, scoring='accuracy',
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        model       = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"  [OK] Best params     : {best_params}")
        print(f"  [OK] CV accuracy     : {grid_search.best_score_:.4f}")

        val_pred     = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1       = f1_score(y_val, val_pred, average='macro')
        print(f"  [OK] Val accuracy    : {val_accuracy:.4f}")
        print(f"  [OK] Val F1 (macro)  : {val_f1:.4f}")

        # Feature importance
        feature_importance = model.feature_importances_
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        print(f"  Top 10 features:")
        for rank, idx in enumerate(top_indices):
            name = self.feature_names[idx] if self.feature_names else f"feat_{idx}"
            print(f"    {rank+1:2d}. {name}: {feature_importance[idx]:.4f}")

        print(f"  [OK] Elapsed         : {_fmt(time.time() - t0)}")

        self.models[model_name]  = model
        self.scalers[model_name] = scaler
        return model, scaler, val_accuracy, val_f1


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_classical_ml_pipeline():
    """Run complete classical ML pipeline with all 4 baseline models"""
    pipeline_start = time.time()
    _section("CLASSICAL ML BASELINE PIPELINE")
    print("Training 4 baseline models: LR, SVM, RF, XGBoost")
    print(f"Results dir : {METRICS_PATH}")

    # ── 1. Feature extraction / loading ───────────────────────────────────
    _subsection("STEP 1 / 5 — Load or Extract Features")

    t0 = time.time()
    try:
        print("  Looking for pre-extracted feature files...")
        X_train = np.load(COMBINED_PROCESSED_PATH / "X_train_features.npy")
        y_train = np.load(COMBINED_PROCESSED_PATH / "y_train.npy")
        X_val   = np.load(COMBINED_PROCESSED_PATH / "X_val_features.npy")
        y_val   = np.load(COMBINED_PROCESSED_PATH / "y_val.npy")
        X_test  = np.load(COMBINED_PROCESSED_PATH / "X_test_features.npy")
        y_test  = np.load(COMBINED_PROCESSED_PATH / "y_test.npy")
        print(f"  [OK] Loaded pre-extracted features ({_fmt(time.time() - t0)})")
    except FileNotFoundError as e:
        print(f"  Feature files not found ({e.filename}). Running feature extraction...")
        from features import extract_features_for_classical_ml
        X_train, X_val, X_test = extract_features_for_classical_ml()
        y_train = np.load(COMBINED_PROCESSED_PATH / "y_train.npy")
        y_val   = np.load(COMBINED_PROCESSED_PATH / "y_val.npy")
        y_test  = np.load(COMBINED_PROCESSED_PATH / "y_test.npy")
        print(f"  [OK] Feature extraction done ({_fmt(time.time() - t0)})")

    print(f"\n  X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}   y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}  y_test  : {y_test.shape}")
    assert X_train.shape[1] == N_FEATURES, \
        f"Expected {N_FEATURES} features, got {X_train.shape[1]}. Re-run feature extraction."

    # ── 2. Cross-dataset features ──────────────────────────────────────────
    _subsection("STEP 2 / 5 — Load or Extract Cross-Dataset Features (UCI HAR)")

    t0 = time.time()
    cross_data_available = False
    X_cross_test = None
    y_cross_test = None

    try:
        X_cross_test = np.load(COMBINED_PROCESSED_PATH / "X_cross_test_features.npy")
        y_cross_test = np.load(COMBINED_PROCESSED_PATH / "y_cross_test.npy")
        cross_data_available = True
        print(f"  [OK] Loaded cross-test features: {X_cross_test.shape} ({_fmt(time.time() - t0)})")
    except FileNotFoundError:
        print("  X_cross_test_features.npy not found. Extracting from raw windows...")
        try:
            X_cross_raw  = np.load(COMBINED_PROCESSED_PATH / "X_cross_test.npy")
            y_cross_test = np.load(COMBINED_PROCESSED_PATH / "y_cross_test.npy")
            print(f"  Loaded raw cross-test windows: {X_cross_raw.shape}")

            from features import FeatureExtractor
            extractor    = FeatureExtractor()
            X_cross_test = extractor.extract_features_batch(X_cross_raw)
            np.save(COMBINED_PROCESSED_PATH / "X_cross_test_features.npy", X_cross_test)
            cross_data_available = True
            print(f"  [OK] Cross-test features extracted and saved: {X_cross_test.shape} ({_fmt(time.time() - t0)})")
        except FileNotFoundError as e:
            print(f"  [WARN] Cross-dataset raw file missing ({e.filename}). Skipping cross-dataset eval.")

    # ── 3. Train + Evaluate Each Model ───────────────────────────────────
    _subsection("STEP 3 / 3 — Train & Evaluate Each Model")

    ml_models   = ClassicalMLModels()
    all_results = {}   # {model_name: {model, scaler, val_acc, val_f1, status}}
    test_results  = {}
    cross_results = {}
    train_order = [
        ("LR",  "Logistic Regression (lower bound reference)", ml_models.train_logistic_regression),
        ("SVM", "SVM RBF kernel (nonlinear baseline)",         ml_models.train_svm_rbf),
        ("RF",  "Random Forest (bootstrap ensemble)",          ml_models.train_random_forest),
    ]
    if XGBOOST_AVAILABLE:
        train_order.append(("XGBoost", "XGBoost (upper bound for hand-crafted features)", ml_models.train_xgboost))
    else:
        print("  [SKIP] XGBoost not available (libomp missing). Run LR, SVM, RF only.")

    n_models = len(train_order)
    for step_num, (model_name, description, train_fn) in enumerate(train_order, 1):
        print(f"\n>>> [{step_num}/{n_models}] {model_name} — {description}")
        try:
            model, scaler, val_acc, val_f1 = train_fn(X_train, y_train, X_val, y_val, model_name)
            all_results[model_name] = {
                "model": model, "scaler": scaler,
                "val_acc": val_acc, "val_f1": val_f1,
                "status": "SUCCESS"
            }
        except Exception as exc:
            print(f"\n  [FAIL] {model_name} FAILED with error:")
            print(f"         {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {
                "model": None, "scaler": None,
                "val_acc": None, "val_f1": None,
                "status": f"FAILED: {type(exc).__name__}: {exc}"
            }
            test_results[model_name] = {"accuracy": None, "f1_macro": None}
            cross_results[model_name] = {"accuracy": None, "f1_macro": None}
            print(f"  Continuing to next model...")
            continue

        # ── Test set evaluation (immediately after training) ──
        X_test_eval = scaler.transform(X_test) if scaler else X_test
        try:
            test_pred = model.predict(X_test_eval)
            test_acc  = accuracy_score(y_test, test_pred)
            test_f1   = f1_score(y_test, test_pred, average='macro')
            test_results[model_name] = {"accuracy": test_acc, "f1_macro": test_f1}
            print(f"  [OK] Test Acc: {test_acc:.4f}  Test F1: {test_f1:.4f}")
        except Exception as exc:
            print(f"  [{model_name}] Test eval FAILED: {exc}")
            test_results[model_name] = {"accuracy": None, "f1_macro": None}

        # ── Cross-dataset evaluation (immediately after training) ──
        if cross_data_available:
            X_cross_eval = scaler.transform(X_cross_test) if scaler else X_cross_test
            try:
                cross_pred = model.predict(X_cross_eval)
                cross_acc  = accuracy_score(y_cross_test, cross_pred)
                cross_f1   = f1_score(y_cross_test, cross_pred, average='macro')
                cross_results[model_name] = {"accuracy": cross_acc, "f1_macro": cross_f1}
                print(f"  [OK] Cross-Dataset Acc: {cross_acc:.4f}  Cross F1: {cross_f1:.4f}")
            except Exception as exc:
                print(f"  [{model_name}] Cross eval FAILED: {exc}")
                cross_results[model_name] = {"accuracy": None, "f1_macro": None}
        else:
            cross_results[model_name] = {"accuracy": None, "f1_macro": None}

        # ── Per-model summary box ──
        print(f"\n  ┌─── {model_name} RESULTS ───")
        print(f"  │ Val  Acc: {val_acc:.4f}   F1: {val_f1:.4f}")
        tr = test_results[model_name]
        print(f"  │ Test Acc: {tr['accuracy']:.4f}   F1: {tr['f1_macro']:.4f}" if tr['accuracy'] else f"  │ Test: N/A")
        cr = cross_results[model_name]
        if cr['accuracy'] is not None:
            print(f"  │ Cross Acc: {cr['accuracy']:.4f}  F1: {cr['f1_macro']:.4f}")
        print(f"  └{'─' * 30}")

    # ── Save results ───────────────────────────────────────────────────────
    _subsection("Saving Results")

    def _safe(val):
        return round(val, 4) if val is not None else None

    baseline_summary = {
        "validation_results": {
            n: {"accuracy": _safe(i["val_acc"]), "f1_macro": _safe(i["val_f1"]),
                "status": i["status"]}
            for n, i in all_results.items()
        },
        "test_results": {
            n: {"accuracy": _safe(v["accuracy"]), "f1_macro": _safe(v["f1_macro"])}
            for n, v in test_results.items()
        },
        "cross_dataset_results": {
            n: {"accuracy": _safe(v["accuracy"]), "f1_macro": _safe(v["f1_macro"])}
            for n, v in cross_results.items()
        } if cross_results else None,
        "model_descriptions": {
            "LR":      "Logistic Regression — one-vs-rest, L2 regularization (lower bound)",
            "SVM":     "SVM RBF — nonlinear boundaries, C×γ tuning",
            "RF":      "Random Forest — bootstrap ensemble, √d features per split",
            "XGBoost": "XGBoost — gradient boosted trees (upper bound for hand-crafted features)"
        }
    }

    out_path = METRICS_PATH / "classical_ml_results.json"
    with open(out_path, "w") as f:
        json.dump(baseline_summary, f, indent=2)
    print(f"  [OK] Results saved to {out_path}")

    # Also save as a tidy CSV
    rows = []
    for name in all_results:
        row = {"Model": name, "Status": all_results[name]["status"]}
        row["Val_Acc"]   = _safe(all_results[name]["val_acc"])
        row["Val_F1"]    = _safe(all_results[name]["val_f1"])
        row["Test_Acc"]  = _safe(test_results.get(name, {}).get("accuracy"))
        row["Test_F1"]   = _safe(test_results.get(name, {}).get("f1_macro"))
        row["Cross_Acc"] = _safe(cross_results.get(name, {}).get("accuracy"))
        row["Cross_F1"]  = _safe(cross_results.get(name, {}).get("f1_macro"))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    csv_path = METRICS_PATH / "classical_ml_results.csv"
    df.to_csv(csv_path)
    print(f"  [OK] CSV saved to {csv_path}")

    # ── Final summary ──────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    _section("FINAL SUMMARY")
    print(f"  Total runtime : {_fmt(total_elapsed)}\n")

    col_w = 10
    header = f"{'Model':<12} {'Status':<10} {'Val Acc':>{col_w}} {'Val F1':>{col_w}} {'Test Acc':>{col_w}} {'Test F1':>{col_w}}"
    if cross_results:
        header += f" {'Cross Acc':>{col_w}} {'Cross F1':>{col_w}}"
    print(header)
    print("─" * len(header))

    for name in all_results:
        status = "OK" if all_results[name]["status"] == "SUCCESS" else "FAIL"
        def _fmt_val(v): return f"{v:.4f}" if v is not None else "  N/A  "
        line = (
            f"{name:<12} {status:<10}"
            f" {_fmt_val(all_results[name]['val_acc']):>{col_w}}"
            f" {_fmt_val(all_results[name]['val_f1']):>{col_w}}"
            f" {_fmt_val(test_results.get(name, {}).get('accuracy')):>{col_w}}"
            f" {_fmt_val(test_results.get(name, {}).get('f1_macro')):>{col_w}}"
        )
        if cross_results:
            line += (
                f" {_fmt_val(cross_results.get(name, {}).get('accuracy')):>{col_w}}"
                f" {_fmt_val(cross_results.get(name, {}).get('f1_macro')):>{col_w}}"
            )
        print(line)

    print("\n  Models that SUCCEEDED:")
    succeeded = [n for n, i in all_results.items() if i["status"] == "SUCCESS"]
    failed    = [n for n, i in all_results.items() if i["status"] != "SUCCESS"]
    for n in succeeded:
        print(f"    [OK] {n}")

    if failed:
        print("\n  Models that FAILED:")
        for n in failed:
            print(f"    [FAIL] {n} — {all_results[n]['status']}")
    else:
        print("\n  All models completed successfully.")

    print(f"\n  Output files:")
    print(f"    {out_path}")
    print(f"    {csv_path}")
    print(f"\n  Done. Total runtime: {_fmt(total_elapsed)}")

    return all_results


if __name__ == "__main__":
    run_classical_ml_pipeline()
