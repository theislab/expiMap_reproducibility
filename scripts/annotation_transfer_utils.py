import collections
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsTransformer

import matplotlib.pyplot as plt
import matplotlib.path
from matplotlib.patches import PathPatch, Patch
import colorsys
import matplotlib.gridspec

import scanpy as sc

from scarches.dataset.trvae.data_handling import remove_sparsity

# All code taken from https://github.com/NUPulmonary/scarches-covid-reference/blob/master/sankey.py and https://github.com/theislab/scarches/blob/51f9ef4ce816bdc0522cb49c6a0eb5d976a59f22/scarches/annotation.py#L9 

def weighted_knn(train_adata, valid_adata, label_key, n_neighbors=50, threshold=0.5,
                 pred_unknown=True, mode='package'):
    """Annotates ``valid_adata`` cells with a trained weighted KNN classifier on ``train_adata``.
        Parameters
        ----------
        train_adata: :class:`~anndata.AnnData`
            Annotated dataset to be used to train KNN classifier with ``label_key`` as the target variable.
        valid_adata: :class:`~anndata.AnnData`
            Annotated dataset to be used to validate KNN classifier.
        label_key: str
            Name of the column to be used as target variable (e.g. cell_type) in ``train_adata`` and ``valid_adata``.
        n_neighbors: int
            Number of nearest neighbors in KNN classifier.
        threshold: float
            Threshold of uncertainty used to annotating cells as "Unknown". cells with uncertainties upper than this
             value will be annotated as "Unknown".
        pred_unknown: bool
            ``True`` by default. Whether to annotate any cell as "unknown" or not. If `False`, will not use
            ``threshold`` and annotate each cell with the label which is the most common in its
            ``n_neighbors`` nearest cells.
        mode: str
            Has to be one of "paper" or "package". If mode is set to "package", uncertainties will be 1 - P(pred_label),
            otherwise it will be 1 - P(true_label).
    """
    print(f'Weighted KNN with n_neighbors = {n_neighbors} and threshold = {threshold} ... ', end='')
    k_neighbors_transformer = KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance',
                                                    algorithm='brute', metric='euclidean',
                                                    n_jobs=-1)
    k_neighbors_transformer.fit(train_adata.X)

    y_train_labels = train_adata.obs[label_key].values
    y_valid_labels = valid_adata.obs[label_key].values

    top_k_distances, top_k_indices = k_neighbors_transformer.kneighbors(X=valid_adata.X)

    stds = np.std(top_k_distances, axis=1)
    stds = (2. / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(top_k_distances_tilda, axis=1, keepdims=True)

    uncertainties = []
    pred_labels = []
    for i in range(len(weights)):
        unique_labels = np.unique(y_train_labels[top_k_indices[i]])
        best_label, best_prob = None, 0.0
        for candidate_label in unique_labels:
            candidate_prob = weights[i, y_train_labels[top_k_indices[i]] == candidate_label].sum()
            if best_prob < candidate_prob:
                best_prob = candidate_prob
                best_label = candidate_label
        
        if pred_unknown:
            if best_prob >= threshold:
                pred_label = best_label
            else:
                pred_label = 'Unknown'
        else:
            pred_label = best_label

        if mode == 'package':
            uncertainties.append(max(1 - best_prob, 0))

        elif mode == 'paper':
            if pred_label == y_valid_labels[i]:
                uncertainties.append(max(1 - best_prob, 0))
            else:
                true_prob = weights[i, y_train_labels[top_k_indices[i]] == y_valid_labels[i]].sum()
                if true_prob > 0.5:
                    pass
                uncertainties.append(max(1 - true_prob, 0))
        else:
            raise Exception("Invalid Mode!")

        pred_labels.append(pred_label)

    pred_labels = np.array(pred_labels).reshape(-1,)
    uncertainties = np.array(uncertainties).reshape(-1,)
    
    labels_eval = pred_labels == y_valid_labels
    labels_eval = labels_eval.astype(object)
    
    n_correct = len(labels_eval[labels_eval == True])
    n_incorrect = len(labels_eval[labels_eval == False]) - len(labels_eval[pred_labels == 'Unknown'])
    n_unknown = len(labels_eval[pred_labels == 'Unknown'])
    
    labels_eval[labels_eval == True] = f'Correct'
    labels_eval[labels_eval == False] = f'InCorrect'
    labels_eval[pred_labels == 'Unknown'] = f'Unknown'
    
    valid_adata.obs['uncertainty'] = uncertainties
    valid_adata.obs[f'pred_{label_key}'] = pred_labels
    valid_adata.obs['evaluation'] = labels_eval
    
    
    print('finished!')
    print(f"Number of correctly classified samples: {n_correct}")
    print(f"Number of misclassified samples: {n_incorrect}")
    print(f"Number of samples classified as unknown: {n_unknown}")

def get_distinct_colors(n):
    """
    https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python/answer/Karthik-Kumar-Viswanathan
    """
    hue_partition = 1 / (n + 1)
    colors = [colorsys.hsv_to_rgb(hue_partition * value, 1.0, 1.0)
              for value in range(0, n)]
    return colors[::2] + colors[1::2]


def text_width(fig, ax, text, fontsize):
    text = ax.text(-100, 0, text, fontsize=fontsize)
    text_bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    text_bb = text_bb.transformed(fig.dpi_scale_trans.inverted())
    width = text_bb.width
    text.remove()
    return width


class Sankey:
    def __init__(self, x, y,
                 plot_width=8,
                 plot_height=8,
                 gap=0.12,
                 alpha=0.3,
                 fontsize='small',
                 left_order=None,
                 mapping=None,
                 tag=None,
                 title=None,
                 title_left=None,
                 title_right=None,
                 ax=None
    ):
        self.X = x
        self.Y = y
        if ax:
            self.plot_width = ax.get_position().width * ax.figure.get_size_inches()[0]
            self.plot_height = ax.get_position().height * ax.figure.get_size_inches()[1]
        else:
            self.plot_width = plot_width
            self.plot_height = plot_height
        self.gap = gap
        self.alpha = alpha
        self.fontsize = fontsize
        self.tag = tag
        self.map = mapping is not None
        self.mapping = mapping
        self.mapping_colors = {
            'increase': '#1f721c',
            'decrease': '#ddc90f',
            'mistake': '#dd1616',
            'correct': '#dddddd',
            'novel': '#59a8d6',
        }
        self.title = title
        self.title_left = title_left
        self.title_right = title_right

        self.need_title = any(map(lambda x: x is not None, (title, title_left, title_right)))
        if self.need_title:
            self.plot_height -= 0.5

        self.init_figure(ax)

        self.flows = collections.Counter(zip(x, y))
        self.init_nodes(left_order)

        self.init_widths()
        # inches per 1 item in x and y
        self.resolution = (plot_height - gap * (len(self.left_nodes) - 1)) / len(x)
        self.colors = {
            name: colour
            for name, colour
            in zip(self.left_nodes.keys(),
                   get_distinct_colors(len(self.left_nodes)))
        }

        self.init_offsets()

    def init_figure(self, ax):
        if ax is None:
            self.fig = plt.figure()
            self.ax = plt.Axes(self.fig, [0, 0, 1, 1])
            self.fig.add_axes(self.ax)
        self.fig = ax.figure
        self.ax = ax

    def init_nodes(self, left_order):
        left_nodes = {}
        right_nodes = {}
        left_offset = 0
        for (left, right), flow in self.flows.items():
            if left in left_nodes:
                left_nodes[left] += flow
            else:
                left_nodes[left] = flow
            if right in right_nodes:
                node = right_nodes[right]
                node[0] += flow
                if flow > node[2]:
                    node[1] = left
                    node[2] = flow
            else:
                right_nodes[right] = [flow, left, flow]

        self.left_nodes = collections.OrderedDict()
        self.left_nodes_idx = {}
        if left_order is None:
            key = lambda pair: -pair[1]
        else:
            left_order = list(left_order)
            key = lambda pair: left_order.index(pair[0])

        for name, flow in sorted(left_nodes.items(), key=key):
            self.left_nodes[name] = flow
            self.left_nodes_idx[name] = len(self.left_nodes_idx)

        left_names = list(self.left_nodes.keys())
        self.right_nodes = collections.OrderedDict()
        self.right_nodes_idx = {}
        for name, node in sorted(
            right_nodes.items(),
            key=lambda pair: (left_names.index(pair[1][1]), -pair[1][2])
        ):
            self.right_nodes[name] = node[0]
            self.right_nodes_idx[name] = len(self.right_nodes_idx)

    def init_widths(self):
        self.left_width = max(
            (text_width(self.fig,
                        self.ax,
                        node,
                        self.fontsize) for node in self.left_nodes)
        )
        if self.title_left:
            self.left_width = max(
                self.left_width,
                text_width(self.fig, self.ax, self.title_left, self.fontsize) / 2
            )
        self.right_width = max(
            (text_width(self.fig,
                        self.ax,
                        node,
                        self.fontsize) for node in self.right_nodes)
        )
        if self.title_right:
            self.right_width = max(
                self.right_width,
                text_width(self.fig, self.ax, self.title_right, self.fontsize) / 2
            )

        self.right_stop = self.plot_width - self.left_width - self.right_width
        self.middle1_stop = self.right_stop * 9 / 20
        self.middle2_stop = self.right_stop * 11 / 20

    def init_offsets(self):
        self.offsets_l = {}
        self.offsets_r = {}

        offset = 0
        for name, flow in self.left_nodes.items():
            self.offsets_l[name] = offset
            offset += flow * self.resolution + self.gap

        offset = 0
        for name, flow in self.right_nodes.items():
            self.offsets_r[name] = offset
            offset += flow * self.resolution + self.gap

    def draw_flow(self, left, right, flow, node_offsets_l, node_offsets_r):
        P = matplotlib.path.Path

        flow *= self.resolution
        left_y = self.offsets_l[left] + node_offsets_l[left]
        right_y = self.offsets_r[right] + node_offsets_r[right]
        if self.need_title:
            left_y += 0.5
            right_y += 0.5
        node_offsets_l[left] += flow
        node_offsets_r[right] += flow
        color = self.colors[left]
        if self.mapping is not None:
            color = self.mapping_colors[self.mapping.category(left, right)]

        path_data = [
            (P.MOVETO, (0, -left_y)),
            (P.LINETO, (0, -left_y - flow)),
            (P.CURVE4, (self.middle1_stop, -left_y - flow)),
            (P.CURVE4, (self.middle2_stop, -right_y - flow)),
            (P.CURVE4, (self.right_stop, -right_y - flow)),
            (P.LINETO, (self.right_stop, -right_y)),
            (P.CURVE4, (self.middle2_stop, -right_y)),
            (P.CURVE4, (self.middle1_stop, -left_y)),
            (P.CURVE4, (0, -left_y)),
            (P.CLOSEPOLY, (0, -left_y)),
        ]
        codes, verts = zip(*path_data)
        path = P(verts, codes)
        patch = PathPatch(
            path,
            facecolor=color,
            alpha=0.9 if flow < .02 else self.alpha,
            edgecolor='none',
        )
        self.ax.add_patch(patch)

    def draw_label(self, label, is_left):
        nodes = self.left_nodes if is_left else self.right_nodes
        offsets = self.offsets_l if is_left else self.offsets_r
        y = offsets[label] + nodes[label] * self.resolution / 2
        if self.need_title:
            y += 0.5

        self.ax.text(
            -.1 if is_left else self.right_stop + .1,
            -y,
            label,
            horizontalalignment='right' if is_left else 'left',
            verticalalignment='center',
            fontsize=self.fontsize,
        )

    def draw_titles(self):
        if self.title:
            self.ax.text(
                self.right_stop / 2,
                -0.25,
                self.title,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=self.fontsize,
                fontweight="bold"
            )
        if self.title_left:
            self.ax.text(
                -.1,
                -0.25,
                self.title_left,
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=self.fontsize
            )
        if self.title_right:
            self.ax.text(
                self.right_stop + .1,
                -0.25,
                self.title_right,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=self.fontsize
            )

    def draw(self):
        node_offsets_l = collections.Counter()
        node_offsets_r = collections.Counter()

        for (left, right), flow in sorted(
            self.flows.items(),
            key=lambda pair: (self.left_nodes_idx[pair[0][0]],
                              self.right_nodes_idx[pair[0][1]])
        ):
            self.draw_flow(left, right, flow, node_offsets_l, node_offsets_r)

        for name in self.left_nodes:
            self.draw_label(name, True)
        for name in self.right_nodes:
            self.draw_label(name, False)
        self.draw_titles()

        self.ax.axis('equal')
        self.ax.set_xlim(-self.left_width - self.gap,
                         self.right_stop + self.gap + self.right_width)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        for k in self.ax.spines.keys():
            self.ax.spines[k].set_visible(False)
        # plt.axis('off')
        # self.fig.set_figheight(self.plot_height)
        # self.fig.set_figwidth(self.plot_width)
        if self.tag:
            text_ax = self.fig.add_axes((0.02, 0.95, 0.05, 0.05), frame_on=False)
            text_ax.set_axis_off()
            plt.text(0, 0, self.tag, fontsize=30, transform=text_ax.transAxes)
        #plt.tight_layout()


def sankey(x, y, **kwargs):
    diag = Sankey(x, y, **kwargs)
    diag.draw()
    return diag.fig

def plot_true_vs_pred(adata_latent_q,ct_col):
    side=12
    # width_ratios=np.array([1,0.4,0.5, 1])
    width_ratios=np.array([1,0.5, 1])
    fig, axes = plt.subplots(ncols=len(width_ratios), 
                             figsize=((width_ratios*side).sum(), side), gridspec_kw={
    "width_ratios":  width_ratios,
    "wspace": 0
    })
    # Used before to ensure that left umap legend did not overlap sankey plot
    #axes[1].set_visible(False)
    #axes[1].set(frame_on=False)
    sc.pl.umap(adata_latent_q, 
               color=ct_col, 
               frameon=False, 
               size=15, 
               #legend_loc="on data",
               title="Query samples with manual cell type annotation",
               ax=axes[0],
               show=False,
               legend_fontweight="normal",
               legend_fontsize=12)
    axes[0].get_legend().remove()
    cluster_colors = pd.Series(adata_latent_q.uns[ct_col+"_colors"])
    cluster_colors.index = adata_latent_q.obs[ct_col].cat.categories
    sankey(adata_latent_q.obs[ct_col], 
                  adata_latent_q.obs['pred_'+ct_col],
                  title="Mapping",
                  title_left="Annotated",
                  title_right="Predicted",
                  ax=axes[1]);
    axes[1].set(frame_on=False)
    cluster_colors["Unknown"] = "#333333"
    cluster_colors = cluster_colors.sort_index()
    pred_cluster_colors = cluster_colors.loc[cluster_colors.index.isin(
        adata_latent_q.obs['pred_'+ct_col].unique())]
    sc.pl.umap(adata_latent_q, 
               color="pred_"+ct_col, 
               frameon=False, 
               size=15, 
               #legend_loc="on data",
               title="Query samples with predicted annotation",
               ax=axes[2],
               palette=list(pred_cluster_colors),
               show=False,
               legend_fontweight="normal",
               legend_fontsize=12)
    axes[2].get_legend().remove()
    handles = []
    for i in cluster_colors.index:
        handles.append(Patch(color=cluster_colors[i], label=i))
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=cluster_colors.size // 2 + 1)
    fig.tight_layout()
    p = axes[1].get_position()
    p.y0 += 0.05
    axes[1].set_position(p)
