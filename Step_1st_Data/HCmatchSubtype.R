# 加载所需的 R 包
library(MatchIt)
library(dplyr)
library(stats) # 用于 t 检验和 p.adjust

# 1. 读取数据
# 请确保此时的 mdd_data 包含了 age, sex, MDD 这几列！
hc_data <- read.csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/AgeSex/HC_INT20_7net_agesex.csv")
mdd_data <- read.csv("/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_INT_7net_agesex.csv")

# 2. 定义包含的列：人口学特征 + 8 个网络特征
networks <- c("subcortical", "Visual", "Somatomotor", "Dorsal_Attention",
              "Ventral_Attention", "Limbic", "Frontoparietal", "Default")

common_cols <- c("subID", "age", "sex", "MDD", networks)

# 提取需要的列并合并数据集
# (若报错找不到列，请检查您的 mdd_data 中是否缺失 age/sex/MDD)
hc_subset <- hc_data[, common_cols]
mdd_subset <- mdd_data[, common_cols]

# 合并数据集
combined_data <- rbind(hc_subset, mdd_subset)

# 3. 数据预处理
# 将 MDD 转换为 0 和 1 的二元变量 (1 = MDD患者, 0 = HC健康对照)
combined_data$is_MDD <- ifelse(combined_data$MDD == 2, 1, 0)
combined_data$sex <- as.factor(combined_data$sex)

# 4. 运行 MatchIt 进行倾向性评分匹配
set.seed(123) # 设置随机种子
match_out <- matchit(is_MDD ~ age + sex,
                     data = combined_data,
                     method = "nearest",
                     distance = "glm",
                     ratio = 1,          # 1:1 匹配
                     caliper = 0.2)      # 卡钳值

# 5. 提取匹配后的数据集
matched_data <- match.data(match_out)
print("========== 匹配后的各组样本量 ==========")
print(table(matched_data$is_MDD))

# ----------------- 新增统计分析部分 -----------------

# 6. 比较匹配后的年龄是否存在差异 (两独立样本 t 检验)
print("========== 年龄差异检验 ==========")
age_ttest <- t.test(age ~ is_MDD, data = matched_data)
print(age_ttest)
# 注意：如果这里的 p-value > 0.05，说明两组年龄匹配良好，无显著统计学差异。

# 7. 比较 8 个网络连接是否存在组间差异
# 创建一个空数据框来存储最终的统计结果
results <- data.frame(
  Network = character(),
  Mean_HC = numeric(),
  Mean_MDD = numeric(),
  t_value = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

# 使用循环对每个网络执行 t 检验
for (net in networks) {
  # 动态生成公式，例如: "Visual ~ is_MDD"
  formula <- as.formula(paste(net, "~ is_MDD"))

  # 执行 t 检验
  test_res <- t.test(formula, data = matched_data)

  # 获取两组的均值 (由于 is_MDD 的 levels 是 0和1，estimate[1]为HC，estimate[2]为MDD)
  mean_hc <- test_res$estimate[1]
  mean_mdd <- test_res$estimate[2]

  # 保存当前网络的计算结果
  results <- rbind(results, data.frame(
    Network = net,
    Mean_HC = mean_hc,
    Mean_MDD = mean_mdd,
    t_value = test_res$statistic,
    p_value = test_res$p.value
  ))
}

# 8. 对 8 个网络的 p 值进行 FDR (False Discovery Rate) 校正
results$p_fdr <- p.adjust(results$p_value, method = "fdr")

# 去除行名并打印最终结果表
rownames(results) <- NULL
print("========== 8个网络组间差异分析及 FDR 校正结果 ==========")
print(results)

# 9. 保存匹配数据及统计结果
# 保存匹配后的受试者数据
#write.csv(matched_data, "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/matched_subjects.csv", row.names = FALSE)

# 保存 8 个网络的统计结果表
#write.csv(results, "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/network_comparison_results.csv", row.names = FALSE)
