'''
两组 INT 脑图的映射（画 HC MDD组适用)

'''
from pathlib import Path
import csv, json, re
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from nilearn import plotting, surface
from nilearn.image import resample_to_img

# ===== 日常使用只需修改下面三个路径，然后直接点击 Run =====
CSV1 = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_FirstRe_covariate/First/subtype1_meanINT.csv"
CSV2 = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_FirstRe_covariate/First/subtype2_meanINT.csv"
OUTPUT_DIR = "/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_FirstRe_covariate/First/MeanINTmapBrain"
# =========================================================

TEMPLATE = Path('/Users/qingchen/Documents/Data/template')
COLORS = ['#00478D','#1E5F9F','#4B82B3','#7DA2C6','#B8CEDD','#FAFDFF','#E5AAB5','#D78393','#C54A61','#B50A27']
# Place near-white exactly at 0.5, rather than 5/9 as in the original code.
CMAP = LinearSegmentedColormap.from_list('INT_blue_white_red', list(zip([0,.1,.2,.3,.4,.5,.65,.77,.89,1], COLORS)), N=256)


def canonical_names(template):
    lut = template/'BrainnetomeAtlas/BN_Atlas_freesurfer/BN_Atlas_246_LUT.txt'
    entries = {}
    for line in lut.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and 1 <= int(parts[0]) <= 246:
            entries[int(parts[0])] = parts[1]
    if set(entries) != set(range(1,247)):
        raise ValueError('BNA246 LUT 必须包含编号 1–246。')
    return [entries[i] for i in range(1,247)]


def read_csv(path, names):
    """支持原始横向 CSV，或两列 region/mean 的纵向 CSV。按名称或编号重排。"""
    path = Path(path).expanduser()
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = [[cell.strip() for cell in row] for row in csv.reader(f)
                if any(cell.strip() for cell in row)]
    if len(rows)==2 and len(rows[0])==247 and len(rows[1])==247:
        keys, raw = rows[0][1:], rows[1][1:]
    elif len(rows)==2 and len(rows[0])==246 and len(rows[1])==246:
        keys, raw = rows
    elif len(rows) in (246,247) and all(len(row)==2 for row in rows):
        data_rows = rows[1:] if len(rows)==247 else rows
        keys, raw = zip(*data_rows)
    else:
        raise ValueError(f'{path}: 需要 246 个脑区；支持横向两行（脑区名、均值），或纵向两列（脑区名/编号、均值）。')
    aliases = {name:name for name in names}
    for name in names:
        aliases[name.split('/')[-1]] = name
    mapped = {}
    for key, value in zip(keys, raw):
        if key.isdigit() and 1 <= int(key) <= 246:
            name = names[int(key)-1]
        else:
            name = aliases.get(key)
        if name is None:
            raise ValueError(f'{path}: 无法识别脑区 {key!r}；请使用 BNA246 名称或编号 1–246。')
        if name in mapped:
            raise ValueError(f'{path}: 重复脑区 {name!r}。')
        try:
            mapped[name] = float(value)
        except ValueError as exc:
            raise ValueError(f'{path}: 脑区 {key!r} 的均值不是数字：{value!r}') from exc
    missing = set(names)-set(mapped)
    if missing:
        raise ValueError(f'{path}: 缺少脑区：{sorted(missing)}')
    result = np.array([mapped[name] for name in names])
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{path}: 均值含 NaN 或无穷大；请先处理缺失数据。')
    return result


def group_name(path):
    stem = Path(path).stem
    return re.sub(r'(?i)[_-]?(246[_-]?)?INT[_-]?mean$', '', stem) or stem


def read_inputs(csv1, csv2, template=TEMPLATE):
    names = canonical_names(template)
    groups = [group_name(csv1), group_name(csv2)]
    if groups[0] == groups[1]:
        groups = [groups[0]+' (1)', groups[1]+' (2)']
    values = {g:read_csv(path,names) for g,path in zip(groups,[csv1,csv2])}
    atlas=nib.load(template/'BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii')
    labels=atlas.get_fdata()[0].astype(int)
    label_names=atlas.header.get_axis(0).label[0]
    # 按名称映射表面标签；兼容右半球独立标签编号。
    aliases = {name:name for name in names}
    aliases.update({name.split('/')[-1]:name for name in names})
    key_to_roi=np.zeros(max(label_names)+1,int)
    for key,(label_name,_) in label_names.items():
        if key==0: continue
        stripped = label_name[2:] if label_name.startswith(('L_','R_')) else label_name
        name = aliases.get(stripped) or aliases.get(stripped.split('/')[-1])
        if name is None:
            raise ValueError(f'表面标签无法匹配 BNA246：{label_name}')
        key_to_roi[key]=names.index(name)+1
    labels=key_to_roi[labels]
    hemi_labels={}
    for structure,slc,bm in atlas.header.get_axis(1).iter_structures():
        if structure in ['CIFTI_STRUCTURE_CORTEX_LEFT','CIFTI_STRUCTURE_CORTEX_RIGHT']:
            hemi= 'left' if structure.endswith('LEFT') else 'right'
            full=np.zeros(bm.nvertices[structure],int)
            full[bm.vertex]=labels[slc]
            hemi_labels[hemi]=full
    vol=nib.load(template/'BrainnetomeAtlas/BN_Atlas_246_2mm.nii.gz')
    vol_labels=vol.get_fdata().astype(int)
    if not np.array_equal(np.unique(vol_labels),np.arange(247)):
        raise ValueError('体积图谱必须包含 BNA246 标签 0–246。')
    bg=nib.load(template/'MNITemplate-master/inst/extdata/MNI152_T1_2mm_Brain.nii.gz')
    bg=resample_to_img(bg,vol,interpolation='continuous',force_resample=True,copy_header=True)
    meshes={hemi:surface.load_surf_mesh(template/f'HCP_S1200_Atlas_Z4_pkXDZ/S1200.{letter}.pial_MSMAll.32k_fs_LR.surf.gii') for hemi,letter in [('left','L'),('right','R')]}
    return values,hemi_labels,meshes,vol,vol_labels,bg


def cortex(ax,mesh,labels,roi,norm,hemi,view):
    lookup=np.r_[np.nan,roi]
    vertex_values=lookup[labels]
    # Match Nilearn mean face interpolation, but apply the exact shared norm.
    face_values=np.mean(vertex_values[mesh.faces],axis=1)
    rgba=CMAP(norm(np.ma.masked_invalid(face_values)))
    rgba[~np.isfinite(face_values)]=[.90,.90,.90,1]
    plotting.plot_surf(mesh,hemi=hemi,view=view,axes=ax,colorbar=False)
    ax.collections[0].set_facecolors(rgba)
    ax.collections[0].set_edgecolors('none')
    ax.set_box_aspect(None, zoom=1.45)
    ax.set_title(f'{hemi.capitalize()} {view}',fontsize=10,pad=-8)


def make_figure(inputs,mode,out):
    values,labels,meshes,vol,vl,bg=inputs
    pooled=np.concatenate(list(values.values()))
    low,high=float(pooled.min()),float(pooled.max())
    center=float(np.median(pooled))
    norm, effective_center, fallback = shared_norm(pooled,mode)
    fig=plt.figure(figsize=(16,10),facecolor='white')
    gs=fig.add_gridspec(3,5,width_ratios=[1,1,.13,1,1],height_ratios=[1,1,1.05],left=.015,right=.985,bottom=.09,top=.92,wspace=.015,hspace=.035)
    for group,base in zip(values,[0,3]):
        roi=values[group]
        for row,view in enumerate(['lateral','medial']):
            for col,hemi in enumerate(['left','right']):
                ax=fig.add_subplot(gs[row,base+col],projection='3d')
                cortex(ax,meshes[hemi],labels[hemi],roi,norm,hemi,view)
        lookup=np.r_[np.nan,roi]
        sub=np.where(vl>210,lookup[vl],np.nan)
        # Convert only the display coordinates; colorbar labels remain raw INT.
        sub_color=(norm(np.ma.masked_invalid(sub))+1).filled(0)
        sub_img=nib.Nifti1Image(sub_color,vol.affine)
        for col,z in enumerate([-12,6]):
            ax=fig.add_subplot(gs[2,base+col])
            # 屏蔽 T1 底图中脑外的零值，让脑外区域显示为白色。
            display=plotting.plot_anat(bg,threshold=1e-6,display_mode='z',cut_coords=[z],axes=ax,figure=fig,annotate=False,draw_cross=False,colorbar=False,black_bg=False)
            display.add_overlay(sub_img,cmap=CMAP,vmin=1,vmax=2,threshold=0.5,colorbar=False)
            ax.set_title(f'Subcortex: z = {z} mm',fontsize=10)
    for x, group in zip([.25,.75],values):
        fig.text(x,.96,group,ha='center',fontsize=18,weight='bold')
    cax=fig.add_axes([.30,.04,.40,.019])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=cax,orientation='horizontal')
    ticks=[norm.vmin,(norm.vmin+effective_center)/2,effective_center,(effective_center+norm.vmax)/2,norm.vmax] if isinstance(norm,TwoSlopeNorm) else np.linspace(norm.vmin,norm.vmax,5)
    cb.set_ticks(ticks);cb.set_ticklabels([f'{t:.3f}' for t in ticks])
    cb.set_label('Raw INT (shared mapping; white = pooled ROI median)' if isinstance(norm,TwoSlopeNorm) else 'Raw INT (shared linear mapping)',fontsize=11)
    fig.savefig(out/f'INT_comparison_{mode}.png',dpi=220,facecolor='white')
    plt.close(fig)
    return {'data_min':low,'data_max':high,'pooled_median':center,'vmin':float(norm.vmin),'vcenter':effective_center,'vmax':float(norm.vmax),'fallback':fallback,'groups':{g:{'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),'sd_across_ROIs':float(v.std()),'median':float(np.median(v))} for g,v in values.items()}}

def shared_norm(pooled, mode):
    """Constant values and endpoint medians cannot be used directly in TwoSlopeNorm."""
    low, high = float(np.min(pooled)), float(np.max(pooled))
    center = float(np.median(pooled))
    if low == high:
        pad = max(abs(low)*0.01, 0.01)
        return Normalize(low-pad, high+pad), low, '所有脑区值相同；扩展色条范围，统一显示中间颜色。'
    if mode == 'pooled_median' and low < center < high:
        return TwoSlopeNorm(vmin=low,vcenter=center,vmax=high), center, None
    return Normalize(low,high), (low+high)/2, ('中位数位于端点，改用共同线性色条。' if mode=='pooled_median' else None)


def plot_int_groups(csv1, csv2, output_dir, *, template_dir=TEMPLATE):
    """仅需三个参数；默认使用本机已安装的 BNA246/HCP/MNI 模板。"""
    out = Path(output_dir).expanduser().resolve()
    inputs = read_inputs(csv1, csv2, Path(template_dir).expanduser())
    out.mkdir(parents=True,exist_ok=True)
    summary = {'source_csvs':[str(Path(p).expanduser().resolve()) for p in [csv1,csv2]],
               'template_dir':str(Path(template_dir).expanduser().resolve()),
               'normalization':{}}
    for mode in ['pooled_median','linear']:
        summary['normalization'][mode] = make_figure(inputs,mode,out)
    (out/'INT_mapping_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'绘图完成，结果保存到：{out}')
    return summary


if __name__ == '__main__':
    try:
        plot_int_groups(CSV1, CSV2, OUTPUT_DIR)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"输入错误：{exc}") from exc
