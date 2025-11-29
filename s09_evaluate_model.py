from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.calibration import cross_val_predict
from sklearn.metrics import root_mean_squared_error
from s01_load_data import load_housing_data
from s03_create_test_set import create_stratified_test_set

# Load the saved model
final_model = load("final_model_c.joblib")

strat_train_set, strat_test_set = create_stratified_test_set(load_housing_data())

X_train = strat_train_set.drop("median_house_value", axis=1)
y_train = strat_train_set["median_house_value"].copy()

val_predictions = cross_val_predict(
    final_model,
    X = X_train,
    y = y_train,
    cv = 3
)

rmse = root_mean_squared_error(y_train, val_predictions)
print("Validation RMSE: ", rmse)


lower_income_mask = X_train["median_income"] < X_train["median_income"].quantile(0.25)
higher_income_mask = X_train["median_income"] > X_train["median_income"].quantile(0.75)
rmse_lower_income = root_mean_squared_error(y_train[lower_income_mask], val_predictions[lower_income_mask])
rmse_higher_income = root_mean_squared_error(y_train[higher_income_mask], val_predictions[higher_income_mask])
print("RMSE for lower income: ", rmse_lower_income)
print("RMSE for higher income: ", rmse_higher_income)


urban = X_train["population"] > 5000
rural = X_train["population"] <= 5000
rmse_urban = root_mean_squared_error(y_train[urban], val_predictions[urban])
rmse_rural = root_mean_squared_error(y_train[rural], val_predictions[rural])
print("RMSE for urban areas: ", rmse_urban)
print("RMSE for rural areas: ", rmse_rural)


std_lower_income = y_train[lower_income_mask].std()
std_higher_income = y_train[higher_income_mask].std()
std_urban = y_train[urban].std()
std_rural = y_train[rural].std()

print(
    "LI: ", std_lower_income,
    "\nHI: ", std_higher_income,
    "\nU:  ", std_urban,
    "\nR:  ", std_rural
)


def normalized_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) / y_true.std()

nrmse_lower_income = normalized_rmse(y_train[lower_income_mask], val_predictions[lower_income_mask])
nrmse_higher_income = normalized_rmse(y_train[higher_income_mask], val_predictions[higher_income_mask])
nrmse_urban = normalized_rmse(y_train[urban], val_predictions[urban])
nrmse_rural = normalized_rmse(y_train[rural], val_predictions[rural])
print("Normalised root mean squared error lower income: ", nrmse_lower_income)
print("Normalised root mean squared error lower income: ", nrmse_higher_income)
print("Normalised root mean squared error urban: ", nrmse_urban)
print("Normalised root mean squared error rural: ", nrmse_rural)




abs_errors = (y_train - val_predictions).abs()
worst_indexes = abs_errors.sort_values(ascending=False).head(10).index
worst = strat_train_set.loc[worst_indexes].copy()
worst["error"] = abs_errors.loc[worst_indexes]
val_pred_series = pd.Series(val_predictions, index=X_train.index)
worst["predicted"] = val_pred_series.loc[worst_indexes]
print(worst)



