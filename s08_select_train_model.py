import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from s01_load_data import add_income_cat, load_housing_data
from s03_create_test_set import create_stratified_test_set
from s07_full_pipeline import full_pipeline

strat_train_set, strat_test_set = create_stratified_test_set(load_housing_data())
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# lin_reg = make_pipeline(preprocessing, LinearRegression())
# lin_reg.fit(housing, housing_labels)
# lin_housing_predictions = lin_reg.predict(housing)

# print(lin_housing_predictions[:5].round(-2))
# print(housing_labels.iloc[:5].values)

# lin_rmse = root_mean_squared_error(housing_labels, lin_housing_predictions)
# print(lin_rmse)


# tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
# tree_reg.fit(housing, housing_labels)
# tree_housing_predictions = tree_reg.predict(housing)

# tree_rmse = root_mean_squared_error(housing_labels, tree_housing_predictions)
# print(tree_rmse)


# tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
#                                 scoring="neg_root_mean_squared_error", cv=10)

# print(pd.Series(tree_rmses).describe())


# forest_reg = make_pipeline(preprocessing, 
#                            RandomForestRegressor(random_state=42, n_jobs=-1),)
# forest_rmses = -cross_val_score(forest_reg, 
#                                 housing, 
#                                 housing_labels,
#                                 scoring="neg_root_mean_squared_error", 
#                                 cv=10,
#                                 n_jobs=-1)

# forest_reg.fit(housing, housing_labels)
# forest_reg_predictions = forest_reg.predict(housing)
# print(pd.Series(forest_rmses).describe())
# print(root_mean_squared_error(housing_labels, forest_reg_predictions))

# from sklearn.model_selection import GridSearchCV



# param_grid = [
#     {
#         'preprocessing__geo__n_clusters': [5, 8, 10],
#         'random_forest__max_features': [4, 6, 8]},
#     {
#         'preprocessing__geo__n_clusters': [10, 15],
#         'random_forest__max_features': [6, 8, 10]
#     }]

# grid_search = GridSearchCV(full_pipeline, 
#                            param_grid, 
#                            cv=3,
#                            scoring='neg_root_mean_squared_error',
#                            n_jobs=-1
#                            )

# grid_search.fit(housing, housing_labels)
# # print(str(full_pipeline.get_params().keys())[:1000] + "...")

# print(grid_search.best_params_)
# # print(grid_search.best_estimator_)
# cv_res = pd.DataFrame(grid_search.cv_results_)
# cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# cv_res = cv_res[["param_preprocessing__geo__n_clusters",
#                  "param_random_forest__max_features", "split0_test_score",
#                  "split1_test_score", "split2_test_score", "mean_test_score"]]
# score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
# cv_res.columns = ["n_clusters", "max_features"] + score_cols
# cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

# print(cv_res.head())


## Full RandomSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
# import pandas as pd

# param_distribs = {
#     'preprocessing__geo__n_clusters': randint(low=3, high=50),
#     'random_forest__max_features': randint(low=2, high=20),
# }

# rnd_search = RandomizedSearchCV(
#     estimator=full_pipeline,
#     param_distributions=param_distribs,
#     n_iter=20,                                 # you choose how many combos to try
#     scoring="neg_root_mean_squared_error",
#     cv=3,
#     random_state=42,
#     n_jobs=-1
# )

# rnd_search.fit(housing, housing_labels)

# cv_res = pd.DataFrame(rnd_search.cv_results_)
# cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# cv_res = cv_res[[
#     "param_preprocessing__geo__n_clusters",
#     "param_random_forest__max_features",
#     "split0_test_score",
#     "split1_test_score",
#     "split2_test_score",
#     "mean_test_score"
# ]]

# score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
# cv_res.columns = ["n_clusters", "max_features"] + score_cols
# cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

# print(cv_res.head())



from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint

param_distribs = {
    'preprocessing__geo__n_clusters': randint(low=3, high=50),
    'random_forest__max_features': randint(low=2, high=20),
    'feature_selection__threshold': ["mean", "median"],
}

halving_search = HalvingRandomSearchCV(
    estimator=full_pipeline,
    param_distributions=param_distribs,
    factor=2,                                 # how aggressively to halve
    scoring="neg_root_mean_squared_error",
    cv=3,
    min_resources=200,
    random_state=42
)

halving_search.fit(housing, housing_labels)

cv_res = pd.DataFrame(halving_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

cv_res = cv_res[[
    "param_preprocessing__geo__n_clusters",
    "param_random_forest__max_features",
    "param_feature_selection__threshold",
    "split0_test_score",
    "split1_test_score",
    "split2_test_score",
    "mean_test_score",
]]

score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features", "threshold"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

print(cv_res.head())

final_model = halving_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
print(feature_importances.round(2))



for importance, name in sorted(zip(feature_importances, final_model["preprocessing"].get_feature_names_out()), reverse=True):
    print(importance, name)





from joblib import dump
dump(final_model, "final_model_c.joblib")