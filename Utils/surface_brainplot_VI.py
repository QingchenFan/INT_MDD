
from pathlib import Path
import csv, json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from nilearn import plotting, surface

DATA_PATH = Path('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_FirstRe_covariate/First_new/subtype1_meanINT.csv')
TEMPLATE = Path('/Users/qingchen/Documents/Data/template')
COLORS = ['#00478D','#1E5F9F','#4B82B3','#7DA2C6','#B8CEDD','#FAFDFF','#E5AAB5','#D78393','#C54A61','#B50A27']
CMAP = LinearSegmentedColormap.from_list('PLS_blue_white_red',
    list(zip([0,.1,.2,.3,.4,.5,.65,.77,.89,1], COLORS)), N=256)

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

def main():
    names = canonical_names(TEMPLATE)
    roi = read_csv(DATA_PATH, names)
    atlas = nib.load(TEMPLATE/'BrainnetomeAtlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii')
    raw_labels = atlas.get_fdata()[0].astype(int)
    label_names = atlas.header.get_axis(0).label[0]
    aliases = {name:name for name in names}
    aliases.update({name.split('/')[-1]:name for name in names})
    key_to_roi = np.zeros(max(label_names)+1, int)
    for key, (label_name, _) in label_names.items():
        if key == 0:
            continue
        stripped = label_name[2:] if label_name.startswith(('L_', 'R_')) else label_name
        name = aliases.get(stripped) or aliases.get(stripped.split('/')[-1])
        if name is None:
            raise ValueError(f'表面标签无法匹配 BNA246: {label_name}')
        key_to_roi[key] = names.index(name)+1
    mapped = key_to_roi[raw_labels]
    hemi_labels = {}
    for structure, slc, bm in atlas.header.get_axis(1).iter_structures():
        if structure in ('CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT'):
            hemi = 'left' if structure.endswith('LEFT') else 'right'
            full = np.zeros(bm.nvertices[structure], int)
            full[bm.vertex] = mapped[slc]
            hemi_labels[hemi] = full
    if set(hemi_labels) != {'left', 'right'}:
        raise ValueError('CIFTI 图谱缺少左右皮层。')
    used = np.unique(np.concatenate(list(hemi_labels.values())))
    used = used[used > 0]
    if not np.array_equal(used, np.arange(1, 211)):
        raise ValueError('预期表面图谱覆盖编号 1–210 的皮层脑区。')
    vmax = float(np.max(np.abs(roi[used-1]))) or 1.
    norm = Normalize(-vmax, vmax)
    out = DATA_PATH.parent/(DATA_PATH.stem+'_cortex_style')
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    gs = fig.add_gridspec(2, 2, left=.02, right=.98, bottom=.14, top=.90,
                          wspace=.01, hspace=.04)
    for col, (hemi, letter) in enumerate([('left','L'), ('right','R')]):
        mesh_path = TEMPLATE/f'HCP_S1200_Atlas_Z4_pkXDZ/S1200.{letter}.pial_MSMAll.32k_fs_LR.surf.gii'
        mesh = surface.load_surf_mesh(mesh_path)
        labels = hemi_labels[hemi]
        if len(mesh.coordinates) != len(labels):
            raise ValueError(f'{hemi}: 模板和标签的顶点数不同。')
        for row, view in enumerate(['lateral','medial']):
            ax = fig.add_subplot(gs[row,col], projection='3d')
            cortex(ax, mesh, labels, roi, norm, hemi, view)
        np.save(out/f'{hemi}_vertex_scores.npy', np.r_[np.nan, roi][labels])
    fig.suptitle(DATA_PATH.stem, y=.96, fontsize=18, weight='bold')
    cax = fig.add_axes([.30,.065,.40,.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation='horizontal')
    ticks = np.linspace(-vmax, vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{v:.3f}' for v in ticks])
    cb.set_label('PLS score (white = 0)', fontsize=11)
    fig.savefig(out/'PLS_cortex_style.png', dpi=300, facecolor='white')
    fig.savefig(out/'PLS_cortex_style.pdf', facecolor='white')
    plt.close(fig)
    with (out/'region_mapping.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['atlas_id','region','score','included_in_cortex'])
        writer.writerows((i,n,float(roi[i-1]),i<=210) for i,n in enumerate(names,1))
    (out/'mapping_summary.json').write_text(json.dumps({
        'data_path':str(DATA_PATH), 'template':str(TEMPLATE),
        'cortical_regions':len(used), 'cortical_min':float(roi[used-1].min()),
        'cortical_max':float(roi[used-1].max()), 'vmin':-vmax,'vmax':vmax,
        'white_center':0, 'surface':'HCP S1200 pial MSMAll 32k fs_LR',
        'face_coloring':'mean of vertex scores; missing faces light gray'
    }, indent=2), encoding='utf-8')
    print(f'绘图完成: {out.resolve()}')


if __name__ == '__main__':
    main()
