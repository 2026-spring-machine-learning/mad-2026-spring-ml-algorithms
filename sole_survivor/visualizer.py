import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sole_survivor.data_loader import FEATURES, TARGET

GRAPHS_DIR = os.path.join(os.path.dirname(__file__), "graphs")


def _save_and_show(fig, filename):
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    fig.savefig(os.path.join(GRAPHS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df, save_path=None):
    cols = FEATURES + [TARGET]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix — Expert Ratings & Survival Score")
    fig.tight_layout()

    _save_and_show(fig, save_path or "correlation_heatmap.png")


def plot_feature_vs_target(df, save_path=None):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        ax.scatter(df[feat], df[TARGET], alpha=0.6, s=30)
        z = np.polyfit(df[feat], df[TARGET], 1)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 200)
        ax.plot(x_line, np.polyval(z, x_line), color="red", linewidth=1.5)
        ax.set_xlabel(feat)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{feat} vs {TARGET}")

    fig.suptitle("Individual Feature vs Survival Score", fontsize=14, y=1.01)
    fig.tight_layout()

    _save_and_show(fig, save_path or "feature_vs_target.png")


def plot_predicted_vs_actual(y_test, y_pred, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5)

    lo = min(np.min(y_test), np.min(y_pred)) - 5
    hi = max(np.max(y_test), np.max(y_pred)) + 5
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual Survival Score")
    ax.set_ylabel("Predicted Survival Score")
    ax.set_title("Predicted vs Actual Survival Score")
    ax.legend()
    fig.tight_layout()

    _save_and_show(fig, save_path or "predicted_vs_actual.png")


def plot_residuals(y_test, y_pred, save_path=None):
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_pred, residuals, alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Predicted Survival Score")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot")
    fig.tight_layout()

    _save_and_show(fig, save_path or "residuals.png")


def plot_feature_importance(model, feature_names, save_path=None):
    coefficients = model.coef_
    sorted_idx = np.argsort(np.abs(coefficients))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        coefficients[sorted_idx],
        color=["#d9534f" if c < 0 else "#5cb85c" for c in coefficients[sorted_idx]],
    )
    ax.set_xlabel("Regression Coefficient")
    ax.set_title("Feature Importance (Linear Regression Coefficients)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    fig.tight_layout()

    _save_and_show(fig, save_path or "feature_importance.png")
