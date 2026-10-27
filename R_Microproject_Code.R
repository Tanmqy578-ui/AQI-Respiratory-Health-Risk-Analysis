# ============================================================
# Data Science Using R Microproject
# Title: AQI-Based Respiratory Health Risk Analysis and Prediction Using R
# Student: Tanmay Arjun Salunkhe | Roll No.: 3032
# Environment: Posit Cloud / RStudio
# ============================================================

# 1. Import dataset
# Keep air_quality_health_impact_data.csv in the same folder as this script.
file_name <- "air_quality_health_impact_data.csv"

if (!file.exists(file_name)) {
  stop("Dataset not found. Add 'air_quality_health_impact_data.csv' to this folder, then run again.")
}

aqi_data <- read.csv(file_name, stringsAsFactors = FALSE)

# 2. Inspect the dataset
cat("\n--- First six rows ---\n")
print(head(aqi_data))
cat("\n--- Structure ---\n")
str(aqi_data)
cat("\n--- Dimensions ---\n")
print(dim(aqi_data))
cat("\n--- Column names ---\n")
print(names(aqi_data))
cat("\n--- Summary ---\n")
print(summary(aqi_data))

# 3. Data quality checks and cleaning
cat("\nMissing values by column:\n")
print(colSums(is.na(aqi_data)))
cat("\nDuplicate rows:", sum(duplicated(aqi_data)), "\n")

aqi_data <- unique(aqi_data)
required_cols <- c("AQI", "PM2_5", "PM10", "NO2", "SO2", "O3", "RespiratoryCases")
missing_cols <- setdiff(required_cols, names(aqi_data))
if (length(missing_cols) > 0) {
  stop(paste("Required columns missing:", paste(missing_cols, collapse = ", ")))
}

model_data <- aqi_data[, required_cols]
for (nm in required_cols) {
  model_data[[nm]] <- suppressWarnings(as.numeric(model_data[[nm]]))
}
model_data <- model_data[complete.cases(model_data), ]
model_data <- model_data[apply(model_data, 1, function(x) all(is.finite(x))), ]

cat("\nRows available for modelling:", nrow(model_data), "\n")

# 4. Descriptive analysis
cat("\nDescriptive statistics for modelling variables:\n")
print(summary(model_data))
if ("HealthImpactClass" %in% names(aqi_data)) {
  cat("\nRecord count by HealthImpactClass:\n")
  print(table(aqi_data$HealthImpactClass, useNA = "ifany"))
}

# 5. Visualizations
dir.create("Output_Screenshots", showWarnings = FALSE)

png("Output_Screenshots/Figure1_AQI_Histogram.png", width = 1000, height = 700)
hist(model_data$AQI, breaks = 10, col = "lightblue",
     main = "Distribution of AQI", xlab = "AQI")
dev.off()

png("Output_Screenshots/Figure2_PM25_vs_RespiratoryCases.png", width = 1000, height = 700)
plot(model_data$PM2_5, model_data$RespiratoryCases, pch = 19,
     col = "steelblue", main = "PM2.5 vs Respiratory Cases",
     xlab = "PM2.5", ylab = "Respiratory Cases")
dev.off()

if ("HealthImpactClass" %in% names(aqi_data)) {
  temp <- aqi_data[, c("HealthImpactClass", "RespiratoryCases")]
  temp$RespiratoryCases <- suppressWarnings(as.numeric(temp$RespiratoryCases))
  temp <- temp[complete.cases(temp), ]
  avg_cases <- aggregate(RespiratoryCases ~ HealthImpactClass, data = temp, FUN = mean)
  png("Output_Screenshots/Figure3_Average_Cases_by_Health_Class.png", width = 1000, height = 700)
  barplot(avg_cases$RespiratoryCases, names.arg = avg_cases$HealthImpactClass,
          col = "lightgreen", main = "Average Respiratory Cases by Health Impact Class",
          xlab = "Health Impact Class", ylab = "Average Respiratory Cases")
  dev.off()
}

# 6. Correlation analysis
cat("\nCorrelation between PM2.5 and respiratory cases:\n")
print(cor(model_data$PM2_5, model_data$RespiratoryCases))

# 7. Train/test split and multiple linear regression
set.seed(123)
train_index <- sample(seq_len(nrow(model_data)), size = floor(0.8 * nrow(model_data)))
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

health_model <- lm(RespiratoryCases ~ AQI + PM2_5 + PM10 + NO2 + SO2 + O3,
                   data = train_data)

cat("\n--- Regression model summary ---\n")
print(summary(health_model))

predicted_cases <- predict(health_model, newdata = test_data)
actual_cases <- test_data$RespiratoryCases

mae <- mean(abs(actual_cases - predicted_cases))
rmse <- sqrt(mean((actual_cases - predicted_cases)^2))
test_r_squared <- 1 - sum((actual_cases - predicted_cases)^2) /
  sum((actual_cases - mean(actual_cases))^2)

cat("\n--- Model Evaluation ---\n")
cat("Training records:", nrow(train_data), "\n")
cat("Testing records:", nrow(test_data), "\n")
cat("MAE:", round(mae, 4), "\n")
cat("RMSE:", round(rmse, 4), "\n")
cat("Test R-squared:", round(test_r_squared, 4), "\n")

# 8. Actual vs predicted plot
png("Output_Screenshots/Figure4_Actual_vs_Predicted.png", width = 1000, height = 700)
plot(actual_cases, predicted_cases, pch = 19, col = "steelblue",
     main = "Actual vs Predicted Respiratory Cases",
     xlab = "Actual Respiratory Cases", ylab = "Predicted Respiratory Cases")
abline(0, 1, col = "red", lwd = 2)
dev.off()

# Save evaluation metrics for the report
metrics <- data.frame(
  Metric = c("MAE", "RMSE", "Test R-squared", "Training records", "Testing records"),
  Result = c(round(mae, 4), round(rmse, 4), round(test_r_squared, 4),
             nrow(train_data), nrow(test_data))
)
write.csv(metrics, "Model_Evaluation_Results.csv", row.names = FALSE)
cat("\nEvaluation table saved to Model_Evaluation_Results.csv\n")
