import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import statsmodels.api as sm


def train_model(X_train, y_train):
    algorithm = lm.LinearRegression()
    model = algorithm.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r_squared = model.score(X_test, y_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)

    results = {"r_squared": r_squared, "rmse": rmse, "mae": mae}

    print("=== Model Evaluation (Test Set) ===")
    print(f"  R-squared : {r_squared:.4f}")
    print(f"  RMSE      : {rmse:.4f}")
    print(f"  MAE       : {mae:.4f}")

    return results


def get_statistical_summary(X, y, feature_names):
    """Full OLS summary via statsmodels for p-values, F-stat, and adj. R²."""
    X_with_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_with_const).fit()

    print("\n=== Statsmodels OLS Summary ===")
    print(ols_model.summary())

    print("\n=== Per-Feature p-values ===")
    for name, pval in zip(feature_names, ols_model.pvalues[1:]):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:20s}  p = {pval:.4f}  {sig}")

    print(f"\n  F-statistic       : {ols_model.fvalue:.4f}")
    print(f"  F-stat p-value    : {ols_model.f_pvalue:.4e}")
    print(f"  Adjusted R-squared: {ols_model.rsquared_adj:.4f}")

    return ols_model
