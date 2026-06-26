library(MatchIt)

# =========================
# 1) Read data
# =========================
hc <- read.csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/AgeSex/HC_INT20_7net_agesex.csv")
mdd <- read.csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_INT_agesex.csv")

hc$Group <- 0
mdd$Group <- 1

df <- rbind(hc, mdd)
df$Group <- as.factor(df$Group)

# =========================
# 2) 1:1 Matching (without replacement)
# =========================
m.out <- matchit(Group ~ age + sex,
                 data = df,
                 method = "optimal",
                 ratio = 1,
                 )

summary(m.out)

# =========================
# 3) Extract matched data
# =========================
matched_df <- match.data(m.out)

cat("Matched HC:", sum(matched_df$Group == 0), "\n")
cat("Matched MDD:", sum(matched_df$Group == 1), "\n")

# 保存匹配后的数据
write.csv(matched_df,
          "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/matched_1to1_dataset.csv",
          row.names = FALSE)

# =========================
# 4) 检验匹配后年龄是否仍有差异
# =========================
t_age <- t.test(age ~ Group, data = matched_df)

cat("\nAge comparison after matching:\n")
print(t_age)

# =========================
# 5) 网络比较
# =========================
networks <- c("subcortical", "Visual", "Somatomotor", "Dorsal_Attention",
              "Ventral_Attention", "Limbic", "Frontoparietal", "Default")

results <- data.frame(Network = networks,
                      Mean_HC = NA,
                      Mean_MDD = NA,
                      t = NA,
                      p = NA)

for(i in 1:length(networks)){

  net <- networks[i]

  hc_values <- matched_df[matched_df$Group == 0, net]
  mdd_values <- matched_df[matched_df$Group == 1, net]

  results$Mean_HC[i] <- mean(hc_values, na.rm = TRUE)
  results$Mean_MDD[i] <- mean(mdd_values, na.rm = TRUE)

  test <- t.test(mdd_values, hc_values)  # Welch t-test

  results$t[i] <- test$statistic
  results$p[i] <- test$p.value
}

# =========================
# 6) FDR 校正
# =========================
results$p_FDR <- p.adjust(results$p, method = "fdr")

print(results)

# 保存结果
#write.csv(results,
#          "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/AgeSex/matched_1to1_network_comparison_FDR.csv",
#          row.names = FALSE)
