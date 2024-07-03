# Load necessary libraries
library(tidymodels)
library(dplyr)
library(randomForest)
library(readr)
library(themis)
library(lubridate)

set.seed(123)

# Load the data
# Load the data
water <- read_csv("water_train.csv")
water_test <- read_csv("water_test.csv")

# Convert report_date to Date format if necessary
water$report_date <- as.Date(water$report_date)

# Extract the year
water$report_date <- year(water$report_date)

# Do the same for the test set
water_test$report_date <- as.Date(water_test$report_date)
water_test$report_date <- year(water_test$report_date)

# Filter data based on install_year
water <- water %>%
  filter(install_year <= 2024)

water_test <- water_test %>%
  filter(install_year <= 2024)

#water <- water %>% 
#  select(-is_urban, -water_point_population, -lat, -lon)

#water_test <- water_test %>% 
#  select(-is_urban, -water_point_population, -lat, -lon)

# Convert character columns to factors and remove ID and report_date
water_factor <- water |>
  mutate_if(is.character, as.factor) |>
  mutate_if(is.logical, as.factor) |>
  select(-ID)

water_test <- water_test |>
  mutate_if(is.character, as.factor) |>
  mutate_if(is.logical, as.factor) |>
  select(-ID)

# Create a recipe with SMOTE
ranger_recipe <- recipe(status_id ~ ., data = water_factor) |>
  step_unknown(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.03) |>
  step_downsample(status_id) %>% 
  step_dummy(all_nominal_predictors()) # one-hot encoding

# Define the model specification
rf_spec <- rand_forest(trees = 1000) |>
  set_mode("classification") |>
  set_engine("randomForest")

# Create the workflow
rf_wflow <- workflow() |>
  add_recipe(ranger_recipe) |>
  add_model(rf_spec)

# Train the model
rf_fit <- rf_wflow |>
  fit(data = water_factor)

# Make predictions
test_pred <- rf_fit |>
  predict(new_data = water_test)

# Prepare the submission file
submission <- data.frame(ID = read_csv("water_test.csv")$ID, status_id = test_pred$.pred_class)
write_csv(submission, "thurs_submission1.csv")


# Local Testing (for accuracy)

# Split the data into training and validation sets
set.seed(123)
data_split <- initial_split(water_factor, prop = 0.75, strata = status_id)
water_train <- training(data_split)
water_validation <- testing(data_split)

# Train the model on the new training set
rf_fit_local <- rf_wflow %>%
  fit(data = water_train)

# Make predictions on the validation set
validation_pred <- rf_fit_local %>%
  predict(new_data = water_validation)

# Add the actual values to the prediction data frame
validation_pred <- validation_pred %>%
  bind_cols(truth = water_validation$status_id)

# Compute accuracy
accuracy <- accuracy(validation_pred, truth = truth, estimate = .pred_class)

# Cross-validation (5-Fold)
set.seed(123)
cv_data <- vfold_cv(water_factor, v = 5, strata = status_id)

rf_fit_cv <- rf_wflow %>%
  fit_resamples(resamples = cv_data)

cv_results <- collect_metrics(rf_fit_cv)