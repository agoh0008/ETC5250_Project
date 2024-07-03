library(tidyverse) 
library(randomForest)

set.seed(123)

water <- read_csv("water_train.csv") |> 
  mutate(status_id = factor(status_id)) 

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

water <- water %>% 
  select(-is_urban)

water_test <- water_test %>% 
  select(-is_urban)

rf_fit <- randomForest(status_id ~ ., data = water[,-1], 
                       importance = TRUE) 

rf_fit 

water_ts_pred <- water_test |> 
  bind_cols(predict(rf_fit, newdata=water_test, type="prob")) |> 
  mutate(pstatus_id = predict(rf_fit, newdata=water_test, 
                              type="response",
                              cutoff = c(0.4, 0.6))) 

write_csv(water_ts_pred[,c("ID", "pstatus_id")], file="predictions1.csv")

# For RMD

read_csv("predictions1.csv")