# Sales Prediction using Python

Predicts future sales based on advertising spend across TV, Radio, and
Newspaper platforms, and analyzes which platform drives sales the most —
to deliver actionable insights for marketing strategy.

## Task

- Predict future sales based on factors like advertising spend, target segment, and platform.
- Prepare data through cleaning, transformation, and feature selection.
- Use regression or time series models to forecast sales.
- Analyze how changes in advertising impact sales outcomes.
- Deliver actionable insights for business marketing strategies.

## Dataset

Source: [Kaggle — Advertising.csv](https://www.kaggle.com/datasets/bumba5341/advertisingcsv),
downloaded via `kagglehub`. Columns: TV, Radio, Newspaper (ad spend,
in thousands of dollars) and Sales (in thousands of units).

If the CSV isn't found locally, the script falls back to a small
synthetic sample so the pipeline can still be demonstrated end to end.

## Project structure

```
CodeAlpha_SalesPrediction/
├── data/
│   └── Advertising.csv
├── src/
│   └── sales_prediction.py
├── outputs/
│   ├── correlation_heatmap.png
│   ├── spend_vs_sales.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── platform_impact.png
├── requirements.txt
└── README.md
```

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the dataset:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("bumba5341/advertisingcsv")
   ```
   Copy the CSV into `data/`, then update `CSV_PATH` in the script.
3. Run the script:
   ```
   python src/sales_prediction.py
   ```

## What the script does

1. **Loads** the dataset and previews it.
2. **Cleans** it: drops stray index columns and missing rows.
3. **Explores** it with a correlation heatmap and per-platform spend-vs-sales scatter plots.
4. **Trains** three regression models: Linear Regression, Ridge Regression,
   and Random Forest.
5. **Evaluates** each with MAE, RMSE, and R² score, and compares them side by side.
6. **Analyzes platform impact** — shows which advertising channel (TV,
   Radio, or Newspaper) has the strongest effect on sales, via feature
   importance or model coefficients.
7. **Summarizes insights** for marketing decisions.

## Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | **0.91** | **1.22** | **1.49** |
| Ridge Regression | 0.91 | 1.22 | 1.49 |
| Random Forest | 0.84 | 1.62 | 1.97 |

**TV advertising has the strongest correlation with sales**, followed by
Radio; Newspaper spend has a comparatively weak effect — suggesting
marketing budget is best prioritized toward TV and Radio.

## Concepts covered

- Data cleaning and preparation for regression.
- Comparing multiple regression algorithms (linear vs regularized vs ensemble).
- Regression evaluation metrics: MAE, RMSE, R².
- Interpreting feature importance/coefficients to guide business decisions.

## Author

Joyshree Maity — EIILM Kolkata, Data Science — CodeAlpha Internship
