import numpy as np
import nibabel as nib
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm

# 1. 加载数据（保持您的原始路径）
Subtype1_Data = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_INT_agesex.csv')
Subtype2_Data = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_INT_agesex.csv')

# 2. 合并数据并添加组别标签
Subtype1_Data['Group'] = 'Subtype1'
Subtype2_Data['Group'] = 'Subtype2'
combined_df = pd.concat([Subtype1_Data, Subtype2_Data], ignore_index=True)

# 3. 提取脑区列名（自动排除所有非脑区特征列，包括亚型2特有的 site, sitenum, MDD）
exclude_cols = ['subID', 'age', 'sex', 'site', 'sitenum', 'MDD', 'Group']
brainRegion = [col for col in combined_df.columns if col not in exclude_cols]

box = []
roi = []
tvalue = []

print("开始进行控制年龄、性别的两亚型（Subtype1 vs Subtype2）脑区差异分析...")

# 4. 循环 246 个脑区，使用 OLS 协方差分析
for i, region in enumerate(brainRegion):

    # 使用 Q() 函数包裹脑区名称，防止如 "46d_L" 这种以数字开头的列名导致公式解析报错
    # C(Group) 和 C(sex) 强制声明其为分类变量
    formula = f'Q("{region}") ~ C(Group) + age + C(sex)'

    # 拟合 OLS 模型
    model = smf.ols(formula, data=combined_df).fit()

    # 提取 Subtype2 相较于 Subtype1 的 t 值和 p 值
    t = model.tvalues['C(Group)[T.Subtype2]']
    p = model.pvalues['C(Group)[T.Subtype2]']

    roi.append(region)
    tvalue.append(t)
    box.append(p)

    # 打印未经校正显著的脑区
    if p < 0.05:
        print(f'ROI: {i + 1} ({region})  P-value: {p:.5f}  T-value: {t:.4f}')

# 转换为 numpy 数组
pvalue = np.array(box)
tvalue = np.array(tvalue)  # 【新增修改】转换为 numpy 数组，方便后续按索引直接取 T 值

# 5. FDR 多重比较校正 (Benjamini-Hochberg)
rejected, fdr_pvalue, _, _ = smm.multipletests(pvalue, alpha=0.05, method='fdr_bh')

# 6. 创建 DataFrame 并保存到 CSV 文件
result_df = pd.DataFrame({
    'ROI': roi,
    'tvalue': tvalue,
    'pvalue': pvalue,
    'fdr-pvalue': fdr_pvalue,
})

csv_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Ttest_Region_Subtype1vs2GLM.csv'
result_df.to_csv(csv_path, index=False)
print(f"统计结果 CSV 已保存至: {csv_path}")

# ==========================================
# 7. 脑图数据写入 (修改：映射 T 值，并由 FDR p 值决定显著性)
# ==========================================
tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
template = nib.load(tpath)
label = template.get_fdata()
label[label > 210] -= 210

print("开始将 FDR 显著的 T 值映射至大脑皮层...")
# 映射回对应的 label (假设脑区在 CSV 和模板中的排序为 1-246 严格对应)
for i in range(1, fdr_pvalue.shape[0] + 1):
    index = np.where(label == i)

    # 【核心修改】通过判断该脑区的 FDR p 值是否显著来填充 T 值
    if fdr_pvalue[i - 1] <= 0.05:
        # 如果 FDR 显著，填充当前脑区对应的 T 值
        label[:, index] = tvalue[i - 1]
    else:
        # 如果不显著，填充为 np.nan 剔除背景
        label[:, index] = np.nan

# 8. 保存 .dscalar.nii 文件
scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['Tvalue'])  # 【修改】将标签改为 Tvalue 更直观
brain_model_axis = template.header.get_axis(1)
scalar_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
scalar_img = nib.Cifti2Image(label, header=scalar_header)

output_nii = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Ttest_Region_Subtype1vs2GLM_Tvalue.dscalar.nii'
scalar_img.to_filename(output_nii)
print(f"画图文件 (dscalar) 已保存至: {output_nii}")