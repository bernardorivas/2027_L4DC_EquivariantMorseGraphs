"""Render the O(3) joint experiment from saved native graphs and checkpoints."""

from .training import read
from .analysis import COLORS, bin_indices
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from PIL import Image


def save(fig, stem):
    stem.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",metadata={"CreationDate":None})
    fig.savefig(stem.with_suffix(".png"),bbox_inches="tight",dpi=180)


def foot(fig, page, text="Learned image hulls and occupancy are numerical. Quotient annotations are separate from original-space indices."):
    fig.text(.04,.018,text,fontsize=8,color="#555555")
    fig.text(.965,.018,str(page),ha="right",fontsize=8,color="#555555")


def record(folder):
    return read(folder/"native-results.json")


def region_colors(basin, active, path, image, membership, info, view):
    included=np.ones_like(active,dtype=bool) if view=="phase" else image
    colors=np.ones((*basin.shape,4))
    colors[included&~active,:3]=.87
    for node in info["sink_nodes"].values():
        base=np.array(to_rgba(COLORS[node%len(COLORS)]))
        base[:3]=.6+.4*base[:3]
        colors[included&active&(basin==node)]=base
    saddle=info["saddle_node"]
    base=np.array(to_rgba(COLORS[saddle%len(COLORS)]));base[:3]=.6+.4*base[:3]
    colors[included&active&path]=base
    for node in np.unique(membership[membership>=0]):
        colors[included&(membership==node)]=to_rgba(COLORS[int(node)%len(COLORS)])
    return colors


def cell_axis(ax, folder, title, view="image"):
    rec=record(folder)
    info=read(folder/"basins.json")
    with np.load(folder/"native-phase.npz",allow_pickle=False) as phase:
        boxes,active,membership=phase["boxes"],phase["active"],phase["morse_node"]
        if info.get("available",True):
            with np.load(folder/"basins.npz",allow_pickle=False) as saved:
                rgba=region_colors(saved["safe_sink_node"],active,saved["path_mask"],saved["image_mask"],membership,info,view)
        else:
            rgba=np.ones((len(boxes),4));rgba[~active,:3]=.87
    n=rec["grid_shape"][0]
    lower,upper=np.array(rec["domain_lower"]),np.array(rec["domain_upper"])
    order,_=bin_indices((boxes[:,:2]+boxes[:,2:])/2,n,lower,upper)
    colors=np.ones((n*n,4));colors[order]=rgba
    ax.imshow(colors.reshape(n,n,4).transpose(1,0,2),origin="lower",extent=[lower[0],upper[0],lower[1],upper[1]],
              interpolation="nearest",aspect="auto")
    text_positions=[]
    for row in rec["nodes"]:
        selected=boxes[row["cell_ids"]]
        polygons=[[(b[0],b[1]),(b[2],b[1]),(b[2],b[3]),(b[0],b[3])] for b in selected]
        color=COLORS[row["node"]%len(COLORS)]
        ax.add_collection(PolyCollection(polygons,facecolors=color,edgecolors="none"))
        center=np.mean((selected[:,:2]+selected[:,2:])/2,axis=0)
        dx=.07*(upper[0]-lower[0])*(1 if center[0]<=0 else -1)
        dy=.08*(upper[1]-lower[1])
        position=center+[dx,dy]
        for _ in range(5):
            if not any(abs((position-p)[0])/(upper[0]-lower[0])<.06 and
                       abs((position-p)[1])/(upper[1]-lower[1])<.07 for p in text_positions):break
            position[1]+=.08*(upper[1]-lower[1])
        text_positions.append(position.copy())
        ax.annotate(str(row["node"]),center,xytext=position,ha="center",fontsize=8,
                    arrowprops=dict(arrowstyle="-",lw=.5,color=color),color=color)
    ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),xlabel="$u_1$",ylabel="$u_2$",title=title)
    ax.set_box_aspect(.65)


def image_axis(ax, path, title):
    ax.imshow(np.asarray(Image.open(path)))
    ax.axis("off")
    ax.set_title(title,fontsize=10)


def preimage_axis(ax, folder, title, view="phase"):
    info=read(folder/"basins.json")
    if not info.get("available",True):
        ax.text(.5,.5,"Basin labeling unavailable for this fit.\nSee basins.json and the actual native graph.",
                ha="center",va="center",transform=ax.transAxes,fontsize=9)
        ax.set_title(title);ax.axis("off")
        return
    with np.load(folder/"physical-basins.npz",allow_pickle=False) as saved:
        a,r=saved["physical_a"],saved["physical_r"]
        colors=region_colors(saved["basin_node"],saved["active"],saved["path"],saved["image"],saved["morse_node"],info,view)
    ax.imshow(colors.transpose(1,0,2),origin="lower",extent=[a[0],a[-1],r[0],r[-1]],aspect="auto")
    ax.axvline(0,color="#777777",lw=.5,ls=":")
    ax.set(xlabel="Physical scalar a",ylabel="Physical radius r",title=title,xlim=(-1.5,1.5),ylim=(0,1.5))
    ax.set_box_aspect(.65)


def basin_page(pdf, run, seed, depth, page, learned, source_size):
    fig,axes=plt.subplots(2,2,figsize=(11.7,8.3))
    for row,view in enumerate(("phase","image")):
        suffix="phase domain" if view=="phase" else "one-step image cover"
        cell_axis(axes[row,0],learned,f"BoxMapData: {suffix}",view)
        preimage_axis(axes[row,1],learned,"Physical inverse images under E",view)
        for ax in axes[row]:
            ax.set_box_aspect(.5)
    sampling=f", {source_size} x {source_size} physical sources"
    fig.suptitle(f"Example 2: regions of attraction, seed {seed}, $2^{{{depth}}}\\times2^{{{depth}}}$ cells{sampling}",fontsize=13)
    fig.text(.05,.075,"Pale sink colors: singleton basins. Other pale color: O-to-C paths. Dark colors: native Morse cells.",fontsize=9)
    fig.text(.05,.05,"White: unresolved, including paths to unsupported cells. Gray: outside sampled support. Colors follow node IDs.",fontsize=9)
    fig.tight_layout(rect=(.015,.12,.985,.94),h_pad=3.)
    foot(fig,page,"Native finite-graph reachability; learned image hulls, support, and physical preimages remain numerical.")
    pdf.savefig(fig);save(fig,run/"figures"/f"seed{seed}-basins");plt.close(fig)
