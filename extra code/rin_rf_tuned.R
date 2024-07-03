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


tune_spec <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")






tune_wf <- workflow() %>%
  add_model(tune_spec) %>%
  add_formula(as.factor(status_id)~.)


rf_grid <- grid_regular(
  mtry(range = c(10, 30)),
  min_n(range = c(5, 8)),
  levels = 5
)

rf_grid

set.seed(1148)

water$report_date <- year(water$report_date)
rf_folds <- vfold_cv(water[,-1],v = 5) # Use the selection of variables in water dataset




doParallel::registerDoParallel()
set.seed(1148)

rf_res <- tune_grid(
    tune_wf,
    resamples = rf_folds,
    grid = rf_grid
  )
rf_res



rf_res %>% 
  collect_metrics() %>% head() %>% kbl()




best_acc <- select_best(rf_res, "accuracy")



final_rf <- finalize_model(
  tune_spec,
  best_acc
)

final_rf



rf_fit_tune <- final_rf %>%
  fit(status_id~.,data = water[,-1]) # Use the selection of variables in water dataset



water_test$report_date <- year(water_test$report_date)

water_ts_pred <- water_test |> 
  bind_cols(predict(rf_fit_tune, new_data=water_test, type="prob")) |> 
  mutate(pstatus_id = if_else(.pred_n >=0.5,"n","y")) 
write_csv(water_ts_pred[,c("ID", "pstatus_id")], file="predictions.csv")



