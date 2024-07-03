library(tidyverse)
library(randomForest)
library(rsample)
library(parsnip)
library(tidymodels)

set.seed(123)

water_train <- read_csv("water_train.csv") |>
  mutate(status_id = factor(status_id))

water_test <- read_csv("water_test.csv")

# Convert report_date to "YYYY" (training):

water_train$report_date <- as.Date(water_train$report_date)
water_train$report_date <- year(water_train$report_date)

# Convert report_date to "YYYY" (test):
water_test$report_date <- as.Date(water_test$report_date)
water_test$report_date <- year(water_test$report_date)

# Transform `distance` variables and filter `install_year` outliers (train):
water_train <- water_train |> 
  mutate_if(is.character, as.factor) |>
  mutate_if(is.logical, as.factor) |>
  mutate(
    distance_to_primary_road = sqrt(distance_to_primary_road),
    distance_to_secondary_road = sqrt(distance_to_secondary_road),
    distance_to_tertiary_road = sqrt(distance_to_tertiary_road)
  ) |>
  filter(install_year <= 2024)


# Transform `distance` variables and filter `install_year` outliers (test):

water_test <- water_test |>
  mutate_if(is.character, as.factor) |>
  mutate_if(is.logical, as.factor) |>
  mutate(
    distance_to_primary_road = sqrt(distance_to_primary_road),
    distance_to_secondary_road = sqrt(distance_to_secondary_road),
    distance_to_tertiary_road = sqrt(distance_to_tertiary_road)
  ) |>
  filter(install_year <= 2024)


# Remove `is_urban` variable (train)

water_train <- water_train |>
  select(-is_urban)


# Remove `is_urban` variable (test)

water_test <- water_test |>
  select(-is_urban)


# Fit random forest model to training data

randf_fit <- randomForest(status_id ~ ., data = water_train[,-1],
                       importance = TRUE)

randf_fit

# Predict on test data

water_test_pred <- water_test |>
  bind_cols(predict(randf_fit, newdata=water_test, type="prob")) |>
  mutate(pstatus_id = predict(randf_fit, newdata=water_test,
                              type="response",
                              cutoff = c(0.4, 0.6)))

write_csv(water_test_pred[,c("ID", "pstatus_id")], file="predictions.csv")
