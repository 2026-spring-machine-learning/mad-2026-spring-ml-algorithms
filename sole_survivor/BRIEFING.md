# Sole Survivor — Briefing

## 1. Did the Experts Do a Good Job?

**Yes.** A linear regression on all 9 expert ratings explains **82% of the variance** in the final Survival Score (R² = 0.820, p < 1e-29).

The three most important predictors:

| Feature         | Effect                        |
|-----------------|-------------------------------|
| PhysicalFitness | Higher fitness → higher score |
| MentalToughness | Higher toughness → higher score |
| Stubbornness    | More stubborn → lower score   |

The remaining features (Leadership, SurvivalSkills, etc.) weren't individually significant, but the model as a whole works well.

## 2. Model Performance

| Metric      | Value  |
|-------------|--------|
| R²          | 0.820  |
| Adjusted R² | 0.802  |
| RMSE        | 7.85   |
| MAE         | 5.69   |

Residuals look clean — no obvious patterns, and roughly normally distributed.

## 3. Charts

Saved in `sole_survivor/graphs/`:

- `correlation_heatmap.png`
- `feature_vs_target.png`
- `predicted_vs_actual.png`
- `residuals.png`
- `feature_importance.png`

## 4. Next Season — Top 3 Picks

| Rank | Name  | Predicted Score |
|------|-------|-----------------|
| 1    | Nico  | 84.39           |
| 2    | Byron | 70.76           |
| 3    | Jonah | 66.00           |

## 5. How to Run

```bash
python3 -m sole_survivor.main
```