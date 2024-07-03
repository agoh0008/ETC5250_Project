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


water <- read_csv("water_train.csv") |>
  mutate(status_id = factor(status_id))
water_test <- read_csv("water_test.csv")

water_factor <- water |>
  mutate_if(is.character, as.factor)

water_factor <- water_factor |>
  select(-ID, -report_date)

water_test <- water_test %>%
  select(-ID, -report_date)

set.seed(123)
water_folds <- vfold_cv(water_factor, strata = status_id)
water_folds

ranger_recipe <- 
  recipe(formula = status_id ~ ., data = water_factor)  

# diff code
rf_spec <- rand_forest(trees = 2000) %>%
  set_mode("classification") %>%
  set_engine("randomForest")

# create workflow
rf_wflow <- workflow() |>
  add_recipe(ranger_recipe) |>
  add_model(rf_spec)

# train model

rf_fit <- rf_wflow |>
  fit(data = water_factor)
# predictions
test_pred <- rf_fit |> predict(new_data = water_test)

submission <- data.frame(ID = read_csv("water_test.csv")$ID, status_id = test_pred$.pred_class)

write_csv(submission, "submissionfixedss.csv")


