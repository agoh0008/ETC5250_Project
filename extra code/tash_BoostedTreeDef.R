
# Default Boosted and Random Forest 

library(readr)
library(tidyverse)
library(dplyr)
library(rsample)
library(parsnip)
library(discrim)
library(kableExtra)
library(yardstick)
library(tidyverse) 
library(randomForest)



# read in the data
water_train <- read_csv("water_train.csv")
water_test <- read_csv("water_test.csv")

# Boosted tree

# Transforming Variables to be Numerical
# Drop the original report_date variable
water_train <- water_train[, !names(water_train) %in% "report_date"]
# Water Tech Category

# Perform one-hot encoding
water_tech_dummies <- model.matrix(~ water_tech_category - 1, data = water_train)

# Combine dummy variables with the original dataset
water_train <- cbind(water_train, water_tech_dummies)

# Drop the original water_tech_category variable
water_train <- water_train[, !names(water_train) %in% "water_tech_category"]

# Is Urban
# Convert is_urban to numerical (0 or 1)
water_train$is_urban <- as.numeric(water_train$is_urban)

# Pay
# Convert pay to numerical (0 or 1)
water_train$pay <- ifelse(water_train$pay == "yes", 1, 0)

# removing 
water_train <- subset(water_train, select = -c(pay, install_year, staleness_score, `water_tech_categoryPublic Tapstand`))

set.seed(1148)

water_std <- water_train %>%
  mutate_at(vars(-1), ~ if(is.numeric(.)) (.-mean(.))/sd(.) else .) 

water_split <- initial_split(water_std, 2/3, 
                             strata = status_id)

w_train <- training(water_split)

w_test <- testing(water_split)

set.seed(1148)
# dont use status id
# converting status_id to factor
w_train <- w_train %>%
  mutate(status_id = factor(status_id)) %>%
  select(-ID) 

# boosted tree
bf_spec <- boost_tree() |>
  set_mode("classification") |>
  set_engine("xgboost") 

bf_fit_w <- bf_spec |> 
  fit(status_id ~., data = w_train)

# fit the model , create confusion tables and accuracy 

bf_ts_pred <- w_test |>
  mutate(pstatus_id = predict(bf_fit_w, 
                              w_test)$.pred_class)

# Convert status_id in the test set to factor 
bf_ts_pred <- bf_ts_pred |> mutate(status_id = factor(status_id))

# calc balance accuracy
boosted_balaccuracy <- bal_accuracy(bf_ts_pred, status_id, pstatus_id)
# calc accuracy
#boosted_accuracy <- accuracy(bf_ts_pred, status_id, pstatus_id) #need to fix

# confusion matrix
boosted_confusion_matrix <- bf_ts_pred |>
  count(status_id, pstatus_id) |>
  group_by(status_id) |>
  mutate(Accuracy = n[status_id==pstatus_id]/sum(n)) |>
  pivot_wider(names_from = "pstatus_id", 
              values_from = n, values_fill = 0) |>
  select(status_id, y, n, Accuracy)


boosted_tm <- bind_cols(predict(bf_fit_w, water_test),
                        predict(bf_fit_w, water_test, type = "prob"))