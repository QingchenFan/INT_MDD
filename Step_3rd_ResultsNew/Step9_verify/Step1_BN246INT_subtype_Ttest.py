import numpy as np
import nibabel as nib
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
import os

# 1. 加载包含 mean_fd 的数据（请确保路径指向最新包含 mean_fd 的 CSV 文件）
path_sub1 = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/subtype1_DZ_INT_FD.csv'
path_sub2 = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/subtype2_DZ_INT_FD.csv'

Subtype1_Data = pd.read_csv(path_sub1)
Subtype2_Data = pd.read_csv(path_sub2)

# 2. 合并数据并添加组别标签
Subtype1_Data['Group'] = 'Subtype1'
Subtype2_Data['Group'] = 'Subtype2'
combined_df = pd.concat([Subtype1_Data, Subtype2_Data], ignore_index=True)

# 3. 提取脑区列名（自动排除所有非脑区特征列）
# 【修改处 1】：将 'mean_fd' 加入排除列表中
exclude_cols = ['subID', 'mean_fd', 'age', 'sex', 'site', 'sitenum', 'MDD', 'Group']
brainRegion = [col for col in combined_df.columns if col not in exclude_cols]

box = []
roi = []
tvalue = []

print("开始进行控制年龄、性别及 mean_fd 的两亚型（Subtype1 vs Subtype2）脑区差异分析...")

# 4. 循环脑区，使用 OLS 协方差分析
for i, region in enumerate(brainRegion):

    # 【修改处 2】：在公式中加入 mean_fd 作为协变量
    formula = f'Q("{region}") ~ C(Group) + age + C(sex) + mean_fd'

    try:
        # 拟合 OLS 模型
        model = smf.ols(formula, data=combined_df).fit()

        # 提取 Subtype2 相较于 Subtype1 的 t 值和 p 值
        t = model.tvalues.get('C(Group)[T.Subtype2]', np.nan)
        p = model.pvalues.get('C(Group)[T.Subtype2]', np.nan)

        roi.append(region)
        tvalue.append(t)
        box.append(p)

        # 打印未经校正显著的脑区
        if not np.isnan(p) and p < 0.05:
            print(f'ROI: {i + 1} ({region})  P-value: {p:.5f}  T-value: {t:.4f}')

    except Exception as e:
        print(f"Warning: 分析脑区 {region} 时出错 - {e}")

pvalue = np.array(box)
tvalue = np.array(tvalue)

# 5. FDR 多重比较校正 (Benjamini-Hochberg)
mask = ~np.isnan(pvalue)
fdr_pvalue = np.ones_like(pvalue)
rejected = np.zeros_like(pvalue, dtype=bool)

if np.any(mask):
    rejected[mask], fdr_pvalue[mask], _, _ = smm.multipletests(pvalue[mask], alpha=0.05, method='fdr_bh')

# 6. 创建 DataFrame 并保存到 CSV 文件
result_df = pd.DataFrame({
    'ROI': roi,
    'tvalue': tvalue,
    'pvalue': pvalue,
    'fdr-pvalue': fdr_pvalue,
})

out_dir = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/Step1_subtype_Ttest'
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, 'Ttest_Region_Subtype1vs2GLM_covFD.csv') # 【修改处 3】：给输出文件改个名字以示区别

result_df.to_csv(csv_path, index=False)
print(f"\n统计结果 CSV 已保存至: {csv_path}")

# ==========================================
# 7. 脑图数据写入 (映射 T 值，并由 FDR p 值决定显著性)
# ==========================================
tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'

if os.path.exists(tpath):
    template = nib.load(tpath)
    label = template.get_fdata()
    label[label > 210] -= 210

    print("开始将 FDR 显著的 T 值映射至大脑皮层...")

    for i in range(1, fdr_pvalue.shape[0] + 1):
        index = np.where(label == i)
        if fdr_pvalue[i - 1] <= 0.05:
            label[:, index] = tvalue[i - 1]
        else:
            label[:, index] = np.nan

    # 8. 保存 .dscalar.nii 文件
    scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['Tvalue'])
    brain_model_axis = template.header.get_axis(1)
    scalar_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    scalar_img = nib.Cifti2Image(label, header=scalar_header)

    output_nii = os.path.join(out_dir, 'Ttest_Region_Subtype1vs2GLM_covFD_Tvalue.dscalar.nii') # 【修改处 4】：区分输出结果文件名
    scalar_img.to_filename(output_nii)
    print(f"画图文件 (dscalar) 已保存至: {output_nii}")
else:
    print(f"\n注意：未找到 CIFTI 模板文件，无法生成 dscalar 脑图文件。请检查路径：{tpath}")