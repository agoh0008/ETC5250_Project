library(vip)
library(tidymodels)
library(rsample)
library(recipes)
library(parsnip)
library(tune)
library(dials)
library(workflows)
library(yardstick)


water <- read_csv("data/water_train.csv")
water_test <- read_csv("data/water_test.csv")


# preprocessing "recipe"
preprocessing_recipe <- 
  recipes::recipe(status_id ~ ., data = water[,-1]) %>% # Choose the combination of variables
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  # combine low frequency factor levels
  recipes::step_other(all_nominal(), threshold = 0.01) %>%
  # remove no variance predictors which provide no predictive information 
  recipes::step_nzv(all_nominal()) %>%
  prep()




xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(), min_n = tune(),
  loss_reduction = tune(),                     ## first three: model complexity
  sample_size = tune(), mtry = tune(),         ## randomness
  learn_rate = tune()                          ## step size
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_spec


xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), water[,-1]), # Use the selection of variables in water dataset
  learn_rate(),
  size = 30
)



xgb_wf <- workflow() %>%
  add_formula(status_id ~ .) %>%
  add_model(xgb_spec)

xgb_wf


set.seed(1148)
vb_folds <- vfold_cv(water[,-1], strata = status_id,v=5)

vb_folds


doParallel::registerDoParallel()

set.seed(1148)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = vb_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE,verbose = TRUE)
)

xgb_res




show_best(xgb_res, "accuracy")

best_acc<- select_best(xgb_res, "accuracy")
best_acc



final_xgb <- finalize_workflow(
  xgb_wf,
  best_acc
)

final_xgb


final_xgb %>%
  fit(data = water[,-1]) %>% # Use the selection of variables in water dataset
  pull_workflow_fit() %>%
  vip(geom = "point")


xgb_fit_tune <- final_xgb %>%
  fit(data = water[,-1]) # Use the selection of variables in water dataset


water_ts_pred <- water_test |> 
  bind_cols(predict(xgb_fit_tune, new_data=water_test, type="prob")) |> 
  mutate(pstatus_id = if_else(.pred_n >=0.5,"n","y")) 
write_csv(water_ts_pred[,c("ID", "pstatus_id")], file="predictions.csv")



