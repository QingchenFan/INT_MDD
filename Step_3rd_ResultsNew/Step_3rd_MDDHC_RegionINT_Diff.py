import numpy as np
import nibabel as nib
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm

# 1. 加载数据
HCData = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_agesex.csv')
MDDData = pd.read_csv('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/MDD_INT20_agesex.csv')

# 2. 合并数据并添加组别标签（为了使用 OLS 控制协变量）
HCData['Group'] = 'HC'
MDDData['Group'] = 'MDD'
combined_df = pd.concat([HCData, MDDData], ignore_index=True)

# 3. 提取脑区列名（自动排除 subID, age, sex, Group 等非脑区特征列）
covariates = ['subID', 'age', 'sex', 'Group']
brainRegion = [col for col in combined_df.columns if col not in covariates]

box = []
roi = []
tvalue = []

print("开始进行控制年龄、性别的 246 脑区差异分析...")

# 4. 循环 246 个脑区，使用 OLS 协方差分析替代独立的 T 检验
for i, region in enumerate(brainRegion):

    # 使用 Q() 函数包裹脑区名称，防止如 "46d_L" 这种以数字开头的列名导致公式解析报错
    # C(Group) 和 C(sex) 强制声明其为分类变量
    formula = f'Q("{region}") ~ C(Group) + age + C(sex)'

    # 拟合 OLS 模型
    model = smf.ols(formula, data=combined_df).fit()

    # 提取 MDD 相较于 HC 的 t 值和 p 值
    t = model.tvalues['C(Group)[T.MDD]']
    p = model.pvalues['C(Group)[T.MDD]']

    roi.append(region)
    tvalue.append(t)
    box.append(p)

    # 打印未经校正显著的脑区
    if p < 0.05:
        print(f'ROI: {i + 1} ({region})  P-value: {p:.5f}  T-value: {t:.4f}')

# 转换为 numpy 数组
pvalue = np.array(box)

# 5. FDR 多重比较校正 (Benjamini-Hochberg)
rejected, fdr_pvalue, _, _ = smm.multipletests(pvalue, alpha=0.05, method='fdr_bh')

# 6. 创建 DataFrame 并保存到 CSV 文件
result_df = pd.DataFrame({
    'ROI': roi,
    'tvalue': tvalue,
    'pvalue': pvalue,
    'fdr-pvalue': fdr_pvalue,
})

csv_path = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/Ttest_Region_HCMDDGLM.csv'
result_df.to_csv(csv_path, index=False)
print(f"统计结果 CSV 已保存至: {csv_path}")

# ==========================================
# 7. 脑图数据写入 (保持原逻辑)
# ==========================================
tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
template = nib.load(tpath)
label = template.get_fdata()
label[label > 210] -= 210

# 使用 fdr_pvalue 映射，并将不显著 (> 0.05) 的脑区设为 NaN（剔除背景或无差异区）
data = fdr_pvalue.copy()
data = np.where(data > 0.05, np.nan, data)

# 映射回对应的 label (假设脑区在 CSV 和模板中的排序为 1-246 严格对应)
for i in range(1, data.shape[0] + 1):
    index = np.where(label == i)
    # 因为 label 可能是 (1, 91282) 这种 2D 形状，保持你原本的操作
    label[:, index] = data[i - 1]

# 8. 保存 .dscalar.nii 文件
scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['IntValue'])
brain_model_axis = template.header.get_axis(1)
scalar_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
scalar_img = nib.Cifti2Image(label, header=scalar_header)

output_nii = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/Ttest_Region_HCMDDGLM.dscalar.nii'
scalar_img.to_filename(output_nii)
print(f"画图文件 (dscalar) 已保存至: {output_nii}")