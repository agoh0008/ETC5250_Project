library(tidyverse)
library(randomForest)
library(rsample)
library(parsnip)
library(tidymodels)

set.seed(123)

water <- read_csv("water_train.csv") |>
  mutate(status_id = factor(status_id))

water_test <- read_csv("water_test.csv")

# Convert report_date to "YYYY" (training): 

water$report_date <- as.Date(water$report_date)
water$report_date <- year(water$report_date)

# Convert report_date to "YYYY" (test):
water_test$report_date <- as.Date(water_test$report_date)
water_test$report_date <- year(water_test$report_date)

# Transform `distance` variables and filter `install_year` outliers (train):
water <- water %>%
  mutate_if(is.character, as.factor) |>
  mutate_if(is.logical, as.factor) |>
  mutate(
    distance_to_primary_road = sqrt(distance_to_primary_road),
    distance_to_secondary_road = sqrt(distance_to_secondary_road),
    distance_to_tertiary_road = sqrt(distance_to_tertiary_road)
  ) |>
  filter(install_year <= 2024)


# Transform `distance` variables and filter `install_year` outliers (test):

water_test <- water_test %>%
  mutate_if(is.character, as.factor) |>
  mutate_if(is.logical, as.factor) |>
  mutate(
    distance_to_primary_road = sqrt(distance_to_primary_road),
    distance_to_secondary_road = sqrt(distance_to_secondary_road),
    distance_to_tertiary_road = sqrt(distance_to_tertiary_road)
  ) |>
  filter(install_year <= 2024)


# Remove `is_urban` variable (train)

water <- water %>%
  select(-is_urban)


# Remove `is_urban` variable (test)

water_test <- water_test %>%
  select(-is_urban)


# Fit random forest model to training data

rf_fit <- randomForest(status_id ~ ., data = water[,-1],
                       importance = TRUE)

rf_fit


# Predict on test data

water_ts_pred <- water_test |>
  bind_cols(predict(rf_fit, newdata=water_test, type="prob")) |>
  mutate(pstatus_id = predict(rf_fit, newdata=water_test,
                              type="response",
                              cutoff = c(0.4, 0.6)))

write_csv(water_ts_pred[,c("ID", "pstatus_id")], file="predictions.csv")