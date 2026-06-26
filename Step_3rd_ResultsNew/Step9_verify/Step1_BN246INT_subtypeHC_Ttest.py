import numpy as np
import nibabel as nib
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
import os


def run_brain_glm_analysis(subtype_data_path, hc_data_path, subtype_label, output_dir):
    """
    通用脑区差异分析函数（控制年龄、性别及 mean_fd，支持 FDR 校正与脑图导出）
    """

    # 1. 加载数据
    print(f"\n==============================================")
    print(f"开始运行统计分析: {subtype_label} vs HC")
    print(f"==============================================")
    Subtype_Data = pd.read_csv(subtype_data_path)
    HC_Data = pd.read_csv(hc_data_path)

    # 2. 合并数据并添加组别标签
    Subtype_Data['Group'] = subtype_label
    HC_Data['Group'] = 'HC'
    combined_df = pd.concat([Subtype_Data, HC_Data], ignore_index=True)

    # 3. 提取脑区列名（自动排除所有非脑区特征列）
    # 【修改 1】：加入了 'mean_fd'，避免将其作为脑区进行分析
    exclude_cols = ['subID', 'mean_fd', 'age', 'sex', 'site', 'sitenum', 'MDD', 'Group']
    brainRegion = [col for col in combined_df.columns if col not in exclude_cols]

    box = []
    roi = []
    tvalue = []

    print(f"开始进行控制年龄、性别及 mean_fd 的两组（{subtype_label} vs HC）脑区 GLM 差异分析...")

    # 4. 循环 246 个脑区，使用 OLS 协方差分析
    for i, region in enumerate(brainRegion):
        # 【修改 2】：在公式中加入 mean_fd 作为协变量
        formula = f'Q("{region}") ~ C(Group) + age + C(sex) + mean_fd'

        try:
            # 拟合 OLS 模型
            model = smf.ols(formula, data=combined_df).fit()

            # 提取当前亚型相较于 HC 的 t 值和 p 值
            t = model.tvalues.get(f'C(Group)[T.{subtype_label}]', np.nan)
            p = model.pvalues.get(f'C(Group)[T.{subtype_label}]', np.nan)

            roi.append(region)
            tvalue.append(t)
            box.append(p)

            # 打印未经校正显著的脑区
            if not np.isnan(p) and p < 0.05:
                print(f'ROI: {i + 1} ({region})  P-value: {p:.5f}  T-value: {t:.4f}')

        except Exception as e:
            print(f"Warning: 分析脑区 {region} 时出错 - {e}")
            roi.append(region)
            tvalue.append(np.nan)
            box.append(np.nan)

    # 转换为 numpy 数组
    pvalue = np.array(box)
    tvalue = np.array(tvalue)

    # 5. FDR 多重比较校正 (Benjamini-Hochberg)
    # 排除潜在的 NaN 数据进行校正以防报错
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

    # 【修改 3】：修改了输出文件名，加上 _covFD 后缀以示区分
    csv_name = f'Ttest_Region_{subtype_label}vsHC_GLM_covFD.csv'
    csv_path = os.path.join(output_dir, csv_name)
    os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(csv_path, index=False)
    print(f"统计结果 CSV 已保存至: {csv_path}")

    # ==========================================
    # 7. 脑图数据写入
    # ==========================================
    tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'

    if os.path.exists(tpath):
        template = nib.load(tpath)
        label = template.get_fdata()
        label[label > 210] -= 210

        print("开始将 FDR 显著的 T 值映射至大脑皮层...")
        for i in range(1, fdr_pvalue.shape[0] + 1):
            index = np.where(label == i)
            label[:, index] = tvalue[i - 1]   # 不考虑显著性
            # if fdr_pvalue[i - 1] <= 0.05:
            #     label[:, index] = tvalue[i - 1]
            # else:
            #     label[:, index] = np.nan

        # 8. 保存 .dscalar.nii 文件
        scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['Tvalue'])
        brain_model_axis = template.header.get_axis(1)
        scalar_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
        scalar_img = nib.Cifti2Image(label, header=scalar_header)

        # 【修改 4】：修改了绘图文件名，加上 _covFD
        nii_name = f'Ttest_Region_{subtype_label}vsHC_GLM_covFD_Tvalue_NoFDR.dscalar.nii'
        output_nii = os.path.join(output_dir, nii_name)
        scalar_img.to_filename(output_nii)
        print(f"画图文件 (dscalar) 已保存至: {output_nii}")
    else:
        print(f"\n注意：未找到 CIFTI 模板文件，无法生成 dscalar 脑图文件。请检查路径：{tpath}")


# ==============================================================================
#  路径配置与运行
# ==============================================================================

# 【修改 5】：更新了输入文件的路径，指向带有 FD 的新文件
HC_PATH = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/HC_DZ_INT_agesex_FD.csv'
BASE_DIR = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/Step1_subtypeHC_Ttest'

SUBTYPE1_PATH = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/subtype1_DZ_INT_FD.csv'
SUBTYPE2_PATH = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_verify/subtype2_DZ_INT_FD.csv'

# ------ 跑 Subtype1 现有的比较 ------
# run_brain_glm_analysis(
#     subtype_data_path=SUBTYPE1_PATH,
#     hc_data_path=HC_PATH,
#     subtype_label='Subtype1',
#     output_dir=BASE_DIR
# )

# ------ 跑 Subtype2 现有的比较 ------
run_brain_glm_analysis(
    subtype_data_path=SUBTYPE2_PATH,
    hc_data_path=HC_PATH,
    subtype_label='Subtype2',
    output_dir=BASE_DIR
)