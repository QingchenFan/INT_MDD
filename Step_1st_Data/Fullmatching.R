library(MatchIt)

# =========================
# 1) Read data
# =========================
hc <- read.csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/AgeSex/HC_INT20_7net_agesex.csv")
mdd <- read.csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/AgeSex/MDD_INT20_7net_agesex.csv")

hc$Group <- 0
mdd$Group <- 1

df <- rbind(hc, mdd)
df$Group <- as.factor(df$Group)

# =========================
# 2) Full matching
# =========================
m.out <- matchit(Group ~ age + sex,
                 data = df,
                 method = "full",
                 estimand = "ATT")   # 常用 ATT (treat=MDD)

summary(m.out)

# =========================
# 3) Extract matched data
# (includes weights + subclass)
# =========================
matched_df <- match.data(m.out)

cat("Original HC:", sum(df$Group == 0), "\n")
cat("Original MDD:", sum(df$Group == 1), "\n")
cat("Matched HC:", sum(matched_df$Group == 0), "\n")
cat("Matched MDD:", sum(matched_df$Group == 1), "\n")

# 保存匹配数据
write.csv(matched_df,
          "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/AgeSex/fullmatched_dataset.csv",
          row.names = FALSE)

# =========================
# 4) Check age difference after matching (weighted regression)
# =========================
age_fit <- lm(age ~ Group, data = matched_df, weights = weights)
summary(age_fit)

cat("\nWeighted age model:\n")
print(summary(age_fit))

# =========================
# 5) Weighted regression for each network
# =========================
networks <- c("subcortical", "Visual", "Somatomotor", "Dorsal_Attention",
              "Ventral_Attention", "Limbic", "Frontoparietal", "Default")

results <- data.frame(Network = networks,
                      Beta = NA,
                      t = NA,
                      p = NA)

for(i in 1:length(networks)){

  net <- networks[i]

  # weighted linear regression
  fit <- lm(as.formula(paste(net, "~ Group")),
            data = matched_df,
            weights = weights)

  sm <- summary(fit)

  # Group effect: Group1 corresponds to MDD vs HC (since Group is factor 0/1)
  beta <- sm$coefficients["Group1", "Estimate"]
  tval <- sm$coefficients["Group1", "t value"]
  pval <- sm$coefficients["Group1", "Pr(>|t|)"]

  results$Beta[i] <- beta
  results$t[i] <- tval
  results$p[i] <- pval
}

# =========================
# 6) FDR correction
# =========================
results$p_FDR <- p.adjust(results$p, method = "fdr")

print(results)

# =========================
# 7) Save results
# =========================
write.csv(results,
          "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/AgeSex/fullmatching_network_results_FDR.csv",
          row.names = FALSE)
