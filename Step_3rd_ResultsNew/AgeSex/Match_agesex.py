import pandas as pd
from scipy import stats


def balance_groups(mdd_path, hc_path, p_threshold=0.05):
    # 1. 读取数据
    mdd = pd.read_csv(mdd_path)
    hc = pd.read_csv(hc_path)

    print(f"【初始状态】MDD 被试数: {len(mdd)}, HC 被试数: {len(hc)}")

    # 初始检验
    t_stat, p_val_age = stats.ttest_ind(mdd['age'], hc['age'])

    # 构建性别列联表 (避免 concat 时索引重复报错，使用 list 转换)
    sex_group = ['MDD'] * len(mdd) + ['HC'] * len(hc)
    sex_values = list(mdd['sex']) + list(hc['sex'])
    contingency = pd.crosstab(pd.Series(sex_group), pd.Series(sex_values))
    chi2, p_val_sex, _, _ = stats.chi2_contingency(contingency)

    print(f"初始年龄 P 值: {p_val_age:.4f}")
    print(f"初始性别 P 值: {p_val_sex:.4f}\n")
    print("-" * 30)

    # 2. 迭代剔除策略
    mdd_filtered = mdd.copy()
    dropped_count = 0
    dropped_records = []  # 新增：用于记录被剔除的被试信息

    while True:
        # 计算当前的 P 值
        t_stat, p_age = stats.ttest_ind(mdd_filtered['age'], hc['age'])

        sex_group = ['MDD'] * len(mdd_filtered) + ['HC'] * len(hc)
        sex_values = list(mdd_filtered['sex']) + list(hc['sex'])
        c = pd.crosstab(pd.Series(sex_group), pd.Series(sex_values))
        chi2, p_sex, _, _ = stats.chi2_contingency(c)

        # 如果年龄和性别都没有显著差异 (p > threshold)，则停止剔除
        if p_age > p_threshold and p_sex > p_threshold:
            break

        # 如果年龄仍有显著差异，剔除 MDD 中年龄最大的人 (因为 MDD 整体偏大)
        if p_age <= p_threshold:
            idx_to_drop = mdd_filtered['age'].idxmax()

            # 新增：在剔除前，记录该被试的完整信息
            dropped_subject = mdd_filtered.loc[idx_to_drop]
            dropped_records.append(dropped_subject)

            # 执行剔除
            mdd_filtered = mdd_filtered.drop(index=idx_to_drop)
            dropped_count += 1

        # 安全机制：如果剔除过多仍无法平衡，避免死循环
        if dropped_count > 100:
            print("警告：剔除过多被试，可能数据差异过大。")
            break

    # 3. 输出最终结果与保存
    print(f"【剔除完成】共剔除了 {dropped_count} 名 MDD 被试。")

    # 新增：将被剔除的被试列表转化为 DataFrame，打印并保存为 CSV
    if dropped_count > 0:
        dropped_df = pd.DataFrame(dropped_records)
        print("\n【被剔除的被试名单 (subID)】:")
        # 打印被剔除的 subID 列表
        print(dropped_df['subID'].tolist())

        # 保存被剔除的被试到独立的 CSV
        dropped_csv_path = 'MDD_dropped_subjects.csv'
        dropped_df.to_csv(dropped_csv_path, index=False)
        print(f"--> 被剔除的详细信息已保存至: {dropped_csv_path}")

    print(f"\n【最终状态】MDD 被试数: {len(mdd_filtered)}, HC 被试数: {len(hc)}")
    print(f"最终年龄 P 值: {p_age:.4f} (均值: MDD={mdd_filtered['age'].mean():.2f}, HC={hc['age'].mean():.2f})")
    print(f"最终性别 P 值: {p_sex:.4f}")

    # 将清洗后的数据保存为新文件
    mdd_filtered.to_csv('MDD_agesex_balanced.csv', index=False)

    return mdd_filtered, hc


MDDpath = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/sourceData/MDD_agesex.csv'
HCpath = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_Match/sourceData/HC_agesex.csv'
# 运行代码
mdd_balanced, hc_balanced = balance_groups(MDDpath, HCpath)