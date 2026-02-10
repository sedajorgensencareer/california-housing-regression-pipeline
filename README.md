# California House Prices - End-to-End ML Project
This project implements an end-to-end machine learning pipeline to predict California district-level median house values using the classic California Housing dataset (from *Hands-On Machine Learning*).

## Problem
Given census-style features (median income, housing age, rooms, population, latitude/longitude, etc.), predict `median_house_value` for each district.

## What I built
- ✅ Stratified train/test split on income buckets to keep distributions consistent
- ✅ Full preprocessing pipeline in scikit-learn (imputation, scaling, feature engineering)
- ✅ Model training and cross-validation
- ✅ Robust evaluation: RMSE, segmented RMSE (income quartiles, urban vs rural), normalised RMSE
- ✅ Error analysis: worst 10 predictions, underprediction of high-value houses due to target censoring
- ✅ Final test-set evaluation + bootstrap 95% confidence interval

## Tech stack
- Python, NumPy, Pandas
- scikit-learn (pipelines, model selection, metrics)
- Matplotlib for visualisation
- joblib for model persistence

# Results
- **Validation RMSE**: ~40,000
- **Final test RMSE**: ~40,850
- **95% CI for RMSE:** 38,903 – 43,236 (bootstrap)

---

🔙 [Back to Portfolio](https://github.com/sedajorgensencareer/Portfolio)


