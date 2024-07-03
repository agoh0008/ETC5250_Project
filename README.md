# ETC5250: Water Availability Prediction Kaggle Project

This repository contains the work and materials for the Kaggle Project of the ETC5250 unit (Introduction to Machine Learning).

Water is a scarce commodity in many parts of the world. Accurately predicting its availability while reducing the need to routinely check would enable lower costs in monitoring that might be better allocated to creating new water resources.

## Aim of the Project
The goal is to accurately predict water availability for the test set.

This challenge is motivated by an analysis by Julia Silge, ["Predict availability in #TidyTuesday water sources with random forest models"](https://juliasilge.com/blog/water-sources/).

## About the Data

The data is downloaded from https://www.waterpointdata.org and represents a subset from a region in Africa. The actual spatial coordinates are disguised. The data has been cleaned, with missing values (small number) imputed, and only reliable variables included.

There files are provided:
- water-train.csv: the training set of data
- water-test.csv: the test set, with no labels, with the response variable `status_id` needing to be predicted.
- sample-submissions.csv: the format of the data that is used to make a submission to Kaggle.

To learn more about the variables provided, please refer [here](https://www.waterpointdata.org/).

