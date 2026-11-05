"""Plot FDR-significant posthoc t maps using local Brainnetome surfaces.
此代码用于画（三组 ANOVA 比较的事后检验结果）基于脑区比较-t值全是负向或 t值全为正向

"""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from nilearn import plotting, datasets

OUT=Path(__file__).resolve().parent
ATLAS=Path('/Users/qingchen/Documents/Data/template/BrainnetomeAtlas')
SURF=ATLAS/'BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k'
vol=nib.load(ATLAS/'BN_Atlas_246_2mm.nii.gz')
labels=vol.get_fdata().astype(int)
background=datasets.load_mni152_template()
# User palette: exact white is anchored at zero.
negative=['#00478D','#1E5F9F','#4B82B3','#7DA2C6','#B8CEDD','#FAFDFF','#FFFFFF']
positive=['#FFFFFF','#FAFDFF','#E5AAB5','#D78393','#C54A61','#B50A27']
# 每次只修改这一行：输入一个事后检验 CSV。
CSV_PATH=Path('/Volumes/QC/INT/INT_BN246_HC135BP_MDD135BP_DZIII/INT_covariate/Step2_subtype/Step2_BNA246INT_Ttest/三组比较结果/posthoc_HC_minus_subtype2.csv')
# CSV 列名统一使用 ROI、tvalue、fdr_pvalue。
OUT.mkdir(parents=True, exist_ok=True)
contrast=CSV_PATH.stem.removeprefix('posthoc_')
title=contrast.replace('_minus_', ' − ').replace('subtype1', 'Subtype 1').replace('subtype2', 'Subtype 2')
df=pd.read_csv(CSV_PATH)
rois=df.ROI.tolist()
fig=plt.figure(figsize=(7,8),facecolor='white')
d=df.set_index('ROI')
vals=d.tvalue.to_numpy();sig=d.fdr_pvalue.to_numpy()<.05
values=np.where(sig,vals,np.nan)
shown=values[np.isfinite(values)]
if len(shown)==0:
    plt.close(fig)
    raise SystemExit(f'{contrast}: 无 FDR 显著脑区，无需绘图')
if np.all(shown>=0):
    vmin,vmax=0,float(shown.max());colors=positive
elif np.all(shown<=0):
    vmin,vmax=float(shown.min()),0;colors=negative
else:
    vmax=float(np.abs(shown).max());vmin=-vmax
    colors=list(zip(np.linspace(0,.5,len(negative)),negative))+list(zip(np.linspace(.5,1,len(positive))[1:],positive[1:]))
cmap=LinearSegmentedColormap.from_list('user_palette',colors,N=257)
cmap.set_bad('#E5E5E5')
norm=Normalize(vmin,vmax)
for h,hemi in [('L','left'),('R','right')]:
    g=nib.load(SURF/f'fsaverage.{h}.BN_Atlas.32k_fs_LR.label.gii')
    lookup={x.key:x.label[2:].split('/')[-1] for x in g.labeltable.labels if x.key>0}
    lab=g.darrays[0].data
    surfvals=np.full(lab.shape,np.nan)
    for key,name in lookup.items():
        if d.loc[name,'fdr_pvalue']<.05:
            surfvals[lab==key]=d.loc[name,'tvalue']
    mesh=nib.load(SURF/f'fsaverage.{h}.inflated.32k_fs_LR.surf.gii')
    faces=mesh.darrays[1].data
    facevalues=np.mean(surfvals[faces],axis=1)
    facecolors=cmap(norm(facevalues))
    facecolors[~np.isfinite(facevalues)]=matplotlib.colors.to_rgba('#E5E5E5')
    for row,view in enumerate(['lateral','medial']):
        ax=fig.add_axes([.025 if h=='L' else .505,.60-row*.245,.47,.28],projection='3d')
        plotting.plot_surf(str(SURF/f'fsaverage.{h}.inflated.32k_fs_LR.surf.gii'),hemi=hemi,view=view,colorbar=False,axes=ax,figure=fig,alpha=1)
        ax.collections[0].set_facecolor(facecolors)
        ax.collections[0].set_edgecolor('none')
mapped=np.zeros(labels.shape,dtype=np.float32)
for i,value in enumerate(values,1):
    if i>210 and np.isfinite(value):mapped[labels==i]=value
img=nib.Nifti1Image(mapped,vol.affine)
volume_cmap=cmap.copy()
volume_cmap.set_bad((0,0,0,0))
for j,z in enumerate([-14,8]):
    ax=fig.add_axes([.13+j*.4,.145,.34,.22])
    plotting.plot_stat_map(img,bg_img=background,display_mode='z',cut_coords=[z],threshold=1e-8,cmap=volume_cmap,vmin=vmin,vmax=vmax,symmetric_cbar=False,colorbar=False,black_bg=False,annotate=False,draw_cross=False,axes=ax,figure=fig)
cax=fig.add_axes([.20,.09,.60,.018])
cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
cb.set_ticks(np.linspace(vmin,vmax,5))
cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
cb.set_label('t value',fontsize=11)
fig.text(.5,.95,title,ha='center',fontsize=19)
fig.text(.5,.915,f'{sig.sum()} / 246 ROIs significant (FDR < 0.05)',ha='center',fontsize=11)
fig.text(.5,.025,'Positive t: first group > second group; light gray: no displayed value.',ha='center',fontsize=8)
for ext in ['png','pdf']:
    fig.savefig(OUT/f'brain_{contrast}.{ext}',dpi=300,facecolor='white')
plt.close(fig)
print(contrast,vmin,vmax,flush=True)
