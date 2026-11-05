"""BNA246单组INT绘图：只需修改CSV_PATH后点击Run。
也可调用plot_single_int(csv_path)，或命令行传入CSV路径。
依赖：numpy、nibabel、matplotlib、nilearn。
输出：四个皮层视角和两个皮层下切片，共用本组原始INT色条。
"""
from pathlib import Path
import argparse
import csv
import json
import inspect
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from nilearn import plotting, surface, datasets
from nilearn.image import resample_to_img

# ===== 日常使用只改这一行，然后点击 Run =====
CSV_PATH = '/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_FirstRe_covariate/First_new/subtype2_meanINT.csv'
# =========================================

# 已预设本机模板；只有换电脑或移动模板时才需要修改。
TEMPLATE_DIR = Path('/Users/qingchen/Documents/Data/template')
COLORS = ['#00478D','#1E5F9F','#4B82B3','#7DA2C6','#B8CEDD','#FAFDFF','#E5AAB5','#D78393','#C54A61','#B50A27']
CMAP = LinearSegmentedColormap.from_list('INT_blue_white_red',list(zip([0,.1,.2,.3,.4,.5,.65,.77,.89,1],COLORS)),N=256)


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

def load_brain_templates(template):
    """只读取模板，不读取第二组CSV。"""
    names=canonical_names(template)
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
    bg=datasets.load_mni152_template(resolution=1)
    vol=resample_to_img(vol,bg,interpolation='nearest',force_resample=True,copy_header=True)
    vol_labels=np.rint(vol.get_fdata()).astype(int)
    meshes={hemi:surface.load_surf_mesh(template/f'HCP_S1200_Atlas_Z4_pkXDZ/S1200.{letter}.pial_MSMAll.32k_fs_LR.surf.gii') for hemi,letter in [('left','L'),('right','R')]}
    return hemi_labels,meshes,vol,vol_labels,bg

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

def plot_single_int(csv_path, output_dir=None, template_dir=TEMPLATE_DIR):
    """唯一必需参数为CSV路径；默认保存到CSV同目录/Single_group_INT/CSV文件名/。

    使用全部246个ROI的原始INT，不做Z标准化。
    蓝端点=本组最小值，白中心=本组中位数，红端点=本组最大值。
    """
    source=Path(csv_path).expanduser().resolve()
    template=Path(template_dir).expanduser().resolve()
    if not source.is_file(): raise FileNotFoundError(f'找不到CSV：{source}')
    if not template.is_dir(): raise FileNotFoundError(f'找不到模板目录：{template}')
    names=canonical_names(template)
    roi=read_csv(source,names)
    labels,meshes,vol,vl,bg=load_brain_templates(template)
    low,center,high=map(float,[roi.min(),np.median(roi),roi.max()])
    fallback=None
    if low< center <high:
        norm=TwoSlopeNorm(vmin=low,vcenter=center,vmax=high)
    elif low==high:
        pad=max(abs(low)*.01,.01)
        norm=Normalize(low-pad,high+pad)
        fallback='全部值相同，扩展色条范围，统一显示中间色。'
    else:
        norm=Normalize(low,high)
        fallback='中位数位于端点，使用本组线性色条。'
    out=Path(output_dir).expanduser().resolve() if output_dir else source.parent/'Single_group_INT'/source.stem
    out.mkdir(parents=True,exist_ok=True)
    group=source.name.split('_')[0]
    print(f'{group}: 246个ROI；min={low:.6f}, median={center:.6f}, max={high:.6f}')
    fig=plt.figure(figsize=(8,10),facecolor='white')
    gs=fig.add_gridspec(3,2,left=.015,right=.985,top=.90,bottom=.115,wspace=.035,hspace=.045,height_ratios=[1,1,1.1])
    for row,view in enumerate(['lateral','medial']):
        for col,hemi in enumerate(['left','right']):
            ax=fig.add_subplot(gs[row,col],projection='3d')
            cortex(ax,meshes[hemi],labels[hemi],roi,norm,hemi,view)
    lookup=np.r_[np.nan,roi]
    sub=np.where(vl>210,lookup[vl],np.nan)
    # 用+1颜色坐标区分背景，不会误删负INT或合法零值；色条仍显示原始INT。
    sub_colors=(norm(np.ma.masked_invalid(sub))+1).filled(0)
    sub_img=nib.Nifti1Image(sub_colors,vol.affine)
    for col,z in enumerate([-18,7]):
        ax=fig.add_subplot(gs[2,col])
        display=plotting.plot_anat(bg,threshold=1e-6,dim=.8,display_mode='z',cut_coords=[z],axes=ax,figure=fig,annotate=False,draw_cross=False,colorbar=False,black_bg=False,interpolation='bicubic',resampling_interpolation='continuous')
        # 兼容新旧Nilearn的不透明度接口。
        opacity={'transparency':1.} if 'transparency' in inspect.signature(display.add_overlay).parameters else {'alpha':1.}
        display.add_overlay(sub_img,cmap=CMAP,vmin=1,vmax=2,threshold=.5,colorbar=False,interpolation='nearest',resampling_interpolation='nearest',**opacity)
        ax.set_title(f'Subcortex: z = {z} mm',fontsize=10)
    fig.text(.5,.96,group,ha='center',fontsize=21,weight='bold')
    fig.text(.5,.926,source.parent.name,ha='center',fontsize=11)
    cax=fig.add_axes([.24,.067,.52,.018])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=cax,orientation='horizontal')
    ticks=[low,(low+center)/2,center,(center+high)/2,high] if isinstance(norm,TwoSlopeNorm) else np.linspace(norm.vmin,norm.vmax,5)
    cb.set_ticks(ticks);cb.set_ticklabels([f'{t:.3f}' for t in ticks])
    cb.set_label("Raw INT (cortex and subcortex share this group's scale)",fontsize=10)
    note="White = this group's ROI median. Color scales differ between groups." if isinstance(norm,TwoSlopeNorm) else 'Group-specific raw INT scale. Color scales differ between groups.'
    fig.text(.5,.013,note,ha='center',fontsize=8)
    filename=out/'INT_single_group.png'
    fig.savefig(filename,dpi=400,facecolor='white')
    plt.close(fig)
    info={'source_csv':str(source),'template_dir':str(template),'ROI_count':246,'raw_min':low,'raw_median':center,'raw_max':high,'colorbar':{'vmin':float(norm.vmin),'vmax':float(norm.vmax),'vcenter':center if isinstance(norm,TwoSlopeNorm) else (float(norm.vmin)+float(norm.vmax))/2},'Z_standardization':False,'fallback':fallback,'subcortical_slices_MNI_z_mm':[-18,7],'atlas_native_resolution_mm':2,'anatomical_background_resolution_mm':1}
    (out/'mapping_summary.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'ROI_raw_INT.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['ROI_ID','ROI_name','raw_INT'])
        writer.writerows((i+1,n,float(roi[i])) for i,n in enumerate(names))
    print(f'绘图完成：{filename}')
    return filename

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='单组BNA246 INT脑图：皮层与皮层下共用本组原始INT色条')
    parser.add_argument('csv_path',nargs='?',default=CSV_PATH,help='不传则使用顶部CSV_PATH')
    parser.add_argument('--output-dir',default=None,help='可选；默认保存到输入CSV对应目录')
    parser.add_argument('--template-dir',default=str(TEMPLATE_DIR),help='可选模板路径')
    args=parser.parse_args()
    plot_single_int(args.csv_path,output_dir=args.output_dir,template_dir=args.template_dir)

