library(vip)
library(tidymodels)
library(rsample)
library(recipes)
library(parsnip)
library(tune)
library(dials)
library(workflows)
library(yardstick)
library(readr)
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

library(smotefamily)
library(performanceEstimation)
library(ROSE)
library(ggplot2)
library(caret)
library(nnet)
library(dplyr)


# EDA

# Prepping Dataset

water <- read_csv("water_train.csv") |>
  mutate(status_id = factor(status_id))

water_test <- read_csv("water_test.csv")

water <- water |>
  mutate_if(is.character, as.factor)

water_test <- water_test |>
  mutate_if(is.character, as.factor)

water <- water |>
  select(-ID, -report_date)

water_test <- water_test %>%
  select(-ID, -report_date)


# SMOTE

# Apply SMOTE to balance the classes in status_id
# fixing up the data to make it usuable for SMOTE
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

# removing report_date
# water <- water[, !names(water) %in% "report_date"]


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


# Filter 'install_year' in 'water' dataset
water <- water %>%
  filter(install_year <= 2023)

# Filter 'install_year' in 'water_test' dataset
water_test <- water_test %>%
  filter(install_year <= 2023)











# SMOTE


# balanced_data <- SMOTE(water[, -which(names(water) == "status_id")], 
#                       target = water$status_id, 
#                       K = 5,  
#                       dup_size = 50)

# Combine the original data with the oversampled data
# balanced_water <- rbind(water, balanced_data$data)



# Model 



set.seed(123)
water_folds <- vfold_cv(water, strata = status_id)
water_folds

ranger_recipe <- 
  recipe(formula = status_id ~ ., data = water)  

# diff code
rf_spec <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_mode("classification") %>%
  set_engine("randomForest")

# create workflow
rf_wflow <- workflow() |>
  add_recipe(ranger_recipe) |>
  add_model(rf_spec)

rf_grid <- grid_regular(
  mtry(range = c(1, ncol(water))),
  trees(range = c(1000, 2000)),
  min_n(range = c(1, 10)),
  levels = 10
)

rf_results <- tune_grid(
  rf_wflow,
  resamples = water_folds,
  grid = rf_grid
)


# Select the best model
best_rf <- rf_results %>% select_best(metric = "accuracy")


# Finalize the model
final_rf_spec <- rand_forest(trees = best_rf$trees) %>%
  set_mode("classification") %>%
  set_engine("randomForest")

final_rf_wflow <- workflow() %>%
  add_recipe(ranger_recipe) %>%
  add_model(final_rf_spec)

final_rf_fit <- final_rf_wflow %>%
  fit(data = water)


# Variable Importance

vip::vip(final_rf_fit)

# Extract the fitted model from the workflow
rf_model <- pull_workflow_fit(final_rf_fit)

# Extract variable importance
var_importance <- randomForest::importance(rf_model$fit)

# Convert to data frame
var_importance <- as.data.frame(var_importance)

# Assuming the importance score is in a column named 'MeanDecreaseAccuracy'
var_importance <- var_importance %>%
  arrange(desc(MeanDecreaseGini))

# Print variable importance
print(var_importance)




# predictions
test_pred <- final_rf_fit |> predict(new_data = water_test)





submission <- data.frame(ID = read_csv("water_test.csv")$ID, status_id = test_pred$.pred_class)

write_csv(submission, "predictions.csv")





