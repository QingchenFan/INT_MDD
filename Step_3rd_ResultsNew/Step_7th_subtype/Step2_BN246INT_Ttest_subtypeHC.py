import numpy as np
import nibabel as nib
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
import os


def run_brain_glm_analysis(subtype_data_path, hc_data_path, subtype_label, output_dir):
    """
    通用脑区差异分析函数（控制年龄、性别，支持 FDR 校正与脑图导出）

    参数:
    - subtype_data_path: 亚型数据的 CSV 路径 (如 subtype1 或 subtype2)
    - hc_data_path: 健康对照组 HC 的 CSV 路径
    - subtype_label: 字符串标签，用于区分和命名输出文件 (例如 'Subtype1' 或 'Subtype2')
    - output_dir: 结果文件保存的目录路径
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
    exclude_cols = ['subID', 'age', 'sex', 'site', 'sitenum', 'MDD', 'Group']
    brainRegion = [col for col in combined_df.columns if col not in exclude_cols]

    box = []
    roi = []
    tvalue = []

    print(f"开始进行控制年龄、性别的两组（{subtype_label} vs HC）脑区 GLM 差异分析...")

    # 4. 循环 246 个脑区，使用 OLS 协方差分析
    for i, region in enumerate(brainRegion):
        # 使用 Q() 函数包裹脑区名称，防止特殊列名解析报错
        formula = f'Q("{region}") ~ C(Group) + age + C(sex)'

        # 拟合 OLS 模型
        model = smf.ols(formula, data=combined_df).fit()

        # 提取当前亚型相较于 HC 的 t 值和 p 值
        # 提示：因字母顺序 HC 在前，statsmodels 默认将 HC 设为对照组，故此处提取 T.{subtype_label}
        t = model.tvalues[f'C(Group)[T.{subtype_label}]']
        p = model.pvalues[f'C(Group)[T.{subtype_label}]']

        roi.append(region)
        tvalue.append(t)
        box.append(p)

        # 打印未经校正显著的脑区
        if p < 0.05:
            print(f'ROI: {i + 1} ({region})  P-value: {p:.5f}  T-value: {t:.4f}')

    # 转换为 numpy 数组
    pvalue = np.array(box)
    tvalue = np.array(tvalue)  # 【新增修改】为了后面方便按索引取值，将 tvalue 转换为 numpy 数组

    # 5. FDR 多重比较校正 (Benjamini-Hochberg)
    rejected, fdr_pvalue, _, _ = smm.multipletests(pvalue, alpha=0.05, method='fdr_bh')

    # 6. 创建 DataFrame 并保存到 CSV 文件
    result_df = pd.DataFrame({
        'ROI': roi,
        'tvalue': tvalue,
        'pvalue': pvalue,
        'fdr-pvalue': fdr_pvalue,
    })

    csv_name = f'Ttest_Region_{subtype_label}vsHC_GLM.csv'
    csv_path = os.path.join(output_dir, csv_name)
    result_df.to_csv(csv_path, index=False)
    print(f"统计结果 CSV 已保存至: {csv_path}")

    # ==========================================
    # 7. 脑图数据写入 (修改：通过 FDR P 值过滤，最终在大脑上映射 T 值)
    # ==========================================
    tpath = '/Users/qingchen/Documents/Data/template/BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii'
    template = nib.load(tpath)
    label = template.get_fdata()
    label[label > 210] -= 210

    print("开始将 FDR 显著的 T 值映射至大脑皮层...")
    # 映射回对应的 label
    for i in range(1, fdr_pvalue.shape[0] + 1):
        index = np.where(label == i)

        # 判断当前脑区的 FDR p 值是否显著
        if fdr_pvalue[i - 1] <= 0.05:
            # 如果显著，填入对应的 t 值
            label[:, index] = tvalue[i - 1]
        else:
            # 如果不显著，填入 np.nan (或可根据需求改为 0)
            label[:, index] = np.nan

    # 8. 保存 .dscalar.nii 文件
    scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(['Tvalue'])  # 将标签改为 Tvalue 更加直观
    brain_model_axis = template.header.get_axis(1)
    scalar_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    scalar_img = nib.Cifti2Image(label, header=scalar_header)

    nii_name = f'Ttest_Region_{subtype_label}vsHC_GLM_Tvalue.dscalar.nii'
    output_nii = os.path.join(output_dir, nii_name)
    scalar_img.to_filename(output_nii)
    print(f"画图文件 (dscalar) 已保存至: {output_nii}")


# ==============================================================================
#  如何使用：您只需要在下方配置好路径，然后通过一行命令即可自由切换/依次运行
# ==============================================================================

# 公共配置
HC_PATH = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step2_Ttest/HC_INT20_agesex.csv'
BASE_DIR = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/Step2_BNA246INT_Ttest'

SUBTYPE1_PATH = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype1_INT_agesex.csv'
SUBTYPE2_PATH = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/Step4_subtype/subtype2_INT_agesex.csv'

# ------ 如果现在只想单独跑 Subtype1 现有的比较 ------
run_brain_glm_analysis(
    subtype_data_path=SUBTYPE1_PATH,
    hc_data_path=HC_PATH,
    subtype_label='Subtype1',
    output_dir=BASE_DIR
)

# ------  Subtype2 时，只需取消下方代码的注释，运行即可 ------
# run_brain_glm_analysis(
#     subtype_data_path=SUBTYPE2_PATH,
#     hc_data_path=HC_PATH,
#     subtype_label='Subtype2',
#     output_dir=BASE_DIR
# )