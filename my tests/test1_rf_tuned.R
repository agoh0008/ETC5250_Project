library(vip)
library(tidyverse)
library(tidymodels)
library(rsample)
library(recipes)
library(parsnip)
library(tune)
library(dials)
library(workflows)
library(yardstick)
library(readr)
library(lubridate)
library(knitr)
library(kableExtra)
library(caret)

water <- read_csv("water_train.csv") %>% 
  mutate(status_id = factor(status_id))
water_test <- read_csv("water_test.csv")

# Feature Engineering (for train)

# Transforming Variables to be Numerical

# Drop the original report_date variable
# water <- water[, !names(water) %in% "report_date"]

# Water Tech Category

# Perform one-hot encoding
water_tech_dummies <- model.matrix(~ water_tech_category - 1, data = water)

# Combine dummy variables with the original dataset
water <- cbind(water, water_tech_dummies)

# Drop the original water_tech_category variable
water <- water[, !names(water) %in% "water_tech_category"]

# Is Urban
# Convert is_urban to numerical (0 or 1)
water$is_urban <- as.numeric(water$is_urban)

# Pay
# Convert pay to numerical (0 or 1)
water$pay <- ifelse(water$pay == "yes", 1, 0)

water <- subset(water, select = -c(pay, install_year, staleness_score, 
                                   `water_tech_categoryPublic Tapstand`, water_point_population, is_urban,
                                   lat, lon))


# Feature Engineering (for water_test)

# Transforming Variables to be Numerical

# Drop the original report_date variable
# water_test <- water_test[, !names(water_test) %in% "report_date"]

# Water Tech Category

# Perform one-hot encoding
water_tech_dummies <- model.matrix(~ water_tech_category - 1, data = water_test)

# Combine dummy variables with the original dataset
water_test <- cbind(water_test, water_tech_dummies)

# Drop the original water_tech_category variable
water_test <- water_test[, !names(water_test) %in% "water_tech_category"]

# Is Urban
# Convert is_urban to numerical (0 or 1)
water_test$is_urban <- as.numeric(water_test$is_urban)

# Pay
# Convert pay to numerical (0 or 1)
water_test$pay <- ifelse(water_test$pay == "yes", 1, 0)

water_test <- subset(water_test, select = -c(pay, install_year, staleness_score, 
                                   `water_tech_categoryPublic Tapstand`, water_point_population, is_urban,
                                   lat, lon))


# Model

# Hyperparameter tuning
tune_spec <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune(),
) %>%
  set_mode("classification") %>%
  set_engine("ranger")






tune_wf <- workflow() %>%
  add_model(tune_spec) %>%
  add_formula(as.factor(status_id)~.)


# New grid for tuning
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



