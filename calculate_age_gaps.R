library(dplyr)

predictions <- read.csv("lightgbm_test_predictions.csv")

age_gaps <- predictions %>%
  group_by(Organ_Type, Fold_Number, Repetition_Number) %>%
  mutate(Age_Gap_Resid = Predicted_Age - predict(loess(Predicted_Age ~ Age, span = 1))) %>%
  group_by(visit_id, Organ_Type) %>%
  summarise(Avg_Age_Gap = mean(Age_Gap_Resid)) %>%
  group_by(Organ_Type) %>%
  mutate(Z_Age_Gap = as.numeric(scale(Avg_Age_Gap))) %>%
  ungroup()

write.csv(age_gaps, "lightgbm_age_gaps.csv", row.names = FALSE)
