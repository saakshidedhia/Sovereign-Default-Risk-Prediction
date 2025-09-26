
# Packages
library(readxl)
library(dplyr)
library(purrr)
library(tidyselect)
library(knitr)
library(kableExtra)
library(tidyr)
library(plm)
library(sandwich)
library(lmtest)

setwd("C:/Users/Ashley/OneDrive/Desktop/501") # Set file directory

data <- read_xlsx("./data-final-vars.xlsx", sheet = "Sheet1", col_names = TRUE,
                  na = "NaN")

pdata_test <- pdata.frame(data, index = c("country_clean", "year"))

# List of all variables
all_vars <- names(pdata_test)

# Variables you want to exclude from regression
vars_to_exclude <- c("systemic_crisis", "country_clean", "year")

# Create list of independent variables
independent_vars <- setdiff(all_vars, vars_to_exclude)

# Build the regression formula
formula <- as.formula(
  paste("systemic_crisis ~", paste(independent_vars, collapse = " + "))
)

# Run the logistic regression with fixed effects and clustered SEs
logit_panel <- feglm(
  formula,
  data = pdata_test,
  family = binomial(link = "logit"),
  cluster = ~country_clean,
  fixef = "year"
)

# View summary
summary(logit_panel)

##### LASSO
library(glmnet)

# Create model matrix 
X <- model.matrix(systemic_crisis ~ total_debt_service_._of_gni +
                    st_debt_._of_total_reserves + 
                    st_debt_._of_total_external_debt +
                    public_and_publicly_guaranteed_debt_service_._of_gni +
                    current_acct_balance_._of_gdp + inflation_rate +
                    Volume_of_exports_of_goods_and_services +
                    gdp_growth_rate + general_govt_revenue_._of_gdp +
                    foreign_direct_investment_net_inflows_._of_gdp +
                    general_govt_total_expenditure_._of_gdp +
                    govt_gross_debt_._of_gdp + total_reserves_._of_total_external_debt,
                  data = pdata_test)

# Drop intercept column 
X <- X[, -1]

# Outcome variable
y <- pdata_test$systemic_crisis

set.seed(123)

# Fit LASSO with Cross-validation
cvfit <- cv.glmnet(X, y, family = "binomial", alpha = 1)

# View the lambda that minimizes CV error
cvfit$lambda.min
coef(cvfit, s = "lambda.min")

# Get coefficient vector at optimal lambda
lasso_coefs <- coef(cvfit, s = "lambda.min")

# Convert LASSO selected vars to a tidy data frame
library(tibble)
library(dplyr)

selected_vars <- lasso_coefs %>%
  as.matrix() %>%
  as_tibble(rownames = "variable") %>%
  rename(coef = `s1`) %>%
  filter(coef != 0 & variable != "(Intercept)") %>%
  pull(variable)

# Build formula string
rhs <- paste(selected_vars, collapse = " + ")
formula_lasso <- as.formula(paste("systemic_crisis ~", rhs))

# Run FE logit on LASSO selected vars
library(fixest)

logit_lasso_fe <- feglm(
  formula_lasso,
  data = pdata_test,
  family = binomial(link = "logit"),
  fixef = "year",
  cluster = ~country_clean
)
summary(logit_lasso_fe)

# Create report table of LASSO-Logit results
library(modelsummary)

# Readable coefficient names
coef_map = c(
  total_debt_service_._of_gni = "Total Debt Service as % of GNI",
  public_and_publicly_guaranteed_debt_service_._of_gni = "Public Debt Service as % of GNI",
  current_acct_balance_._of_gdp = "Current Account Balance as % of GDP",
  general_govt_revenue_._of_gdp = "General Government Revenue as % of GDP",
  foreign_direct_investment_net_inflows_._of_gdp = "FDI Net Inflows as % of GDP",
  gdp_growth_rate = "GDP Growth Rate",
  inflation_rate = "Inflation Rate"
)

# Generate LaTeX table
modelsummary(
  logit_lasso_fe,
  output = "lasso_results.tex",
  title = "LASSO-Selected Logistic Regression",
  coef_map = coef_map,
  statistic = "conf.int",
  gof_omit = "AIC|BIC|R2",
  stars = c("*" = .1, "**" = .05, "***" = .01),
  notes = "Standard errors clustered at the country level. Year fixed effects included."
)


################################################################################
# Old code for ROC, Confusion Matrix

# # Store predicted probabilities
# predicted_probs <- predict(logit_panel, type = "response")
# 
# # Rough ROC curve, print AUC
# roc_obj <- roc(pdata_clean$systemic_crisis, predicted_probs)
# plot(roc_obj)
# auc(roc_obj)
# 
# # Confusion Matrix
# predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)
# cm <- confusionMatrix(factor(predicted_classes),
#                 factor(pdata_clean$systemic_crisis),
#                 positive = "1")
# 
# 
# # Pull the table
# cm_table <- as.data.frame(cm$table)
# 
# # Make a nice confusion matrix table
# cm_table %>%
#   kable(caption = "Confusion Matrix", align = "c") %>%
#   kable_styling(full_width = FALSE, position = "center")
# 
# # Pull key statistics
# stats_summary <- data.frame(
#   Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision (PPV)"),
#   Value = c(
#     cm$overall["Accuracy"],
#     cm$byClass["Sensitivity"],
#     cm$byClass["Specificity"],
#     cm$byClass["Pos Pred Value"]
#   )
# )
# 
# # Format it
# stats_summary %>%
#   kable(caption = "Model Performance Metrics", digits = 3, align = "c") %>%
#   kable_styling(full_width = FALSE, position = "center")
# 
# # Rename columns for nicer labels
# colnames(cm_table) <- c("Predicted", "Actual", "Freq")
# 
# # Tag correct/incorrect
# cm_table <- cm_table %>%
#   mutate(
#     Outcome = case_when(
#       Predicted == Actual ~ "Correct",
#       Predicted != Actual ~ "Incorrect"
#     )
#   )
# 
# ggplot(cm_table, aes(x = Actual, y = Predicted, fill = Outcome)) +
#   geom_tile(aes(alpha = Freq), color = "black", linewidth = 0.8) +
#   geom_text(aes(label = Freq), size = 6, color = "black") +
#   scale_fill_manual(values = c("Correct" = "#66c2a5",   # greenish for correct
#                                "Incorrect" = "#fc8d62")) + # orangish for incorrect
#   scale_alpha(range = c(0.4, 1)) +  # lighter tiles for lower counts
#   labs(
#     title = "Confusion Matrix",
#     x = "Actual Class",
#     y = "Predicted Class",
#     fill = "Prediction",
#     alpha = "Frequency"
#   ) +
#   guides(alpha = "none") +
#   theme_minimal(base_size = 16) +
#   theme(
#     plot.title = element_text(hjust = 0.5, face = "bold"),
#     axis.title = element_text(face = "bold"),
#     legend.title = element_text(face = "bold")
#   )
# 
# ggsave("confusion_matrix_plot.png", plot = last_plot(), width = 6, height = 5, dpi = 600)  # Very high resolution
# 
# # Fancy ROC curve
# roc_data <- data.frame(
#   specificity = rev(roc_obj$specificities),
#   sensitivity = rev(roc_obj$sensitivities)
# )
# 
# roc_curve <- ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity)) +
#   geom_line(linewidth = 1.5) +
#   geom_abline(linetype = "dashed", color = "gray") +
#   labs(
#        title = "ROC Curve",
#        x = "False Positive Rate (1 - Specificity)",
#        y = "True Positive Rate (Sensitivity)"
#        ) +
#   theme_minimal(base_size = 15)
# 
# ggsave("ROC-Logit.png", plot = roc_curve, device = png, width = 10, height = 8, dpi = 600)
# ################################################################################


