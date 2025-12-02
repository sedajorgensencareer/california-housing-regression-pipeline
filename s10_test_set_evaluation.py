"""Final test-set evaluation with RMSE and confidence interval."""

from joblib import load
import numpy as np
from sklearn.metrics import root_mean_squared_error
from s03_create_test_set import create_stratified_test_set
from s01_load_data import load_housing_data
from scipy.stats import bootstrap

final_model = load("final_model_c.joblib")
strat_train_set, strat_test_set = create_stratified_test_set(load_housing_data())

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

test_rmse = root_mean_squared_error(y_test, final_predictions)
print("Final Test RMSE:", test_rmse)

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = bootstrap([squared_errors], 
                        rmse, 
                        confidence_level=confidence,
                        random_state=42)

rmse_lower, rmse_upper = boot_result.confidence_interval

print(f"95% CI for RMSE: ({rmse_lower}, {rmse_upper})")