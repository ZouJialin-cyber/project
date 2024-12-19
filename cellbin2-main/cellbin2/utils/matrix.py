import os.path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

np.set_printoptions(precision=3)
import pandas as pd
import base64
import seaborn as sns
from cellbin2.modules.metadata import TechType

class cbMatrix(object):
    """ 单个矩阵管理，是 cellbin 生成的矩阵，保存的时候，会转成gef的方式供后续读写 """

    def __init__(self, file_path: str,matrix_type=TechType.Transcriptomics):
        """
        :param file_path: cellbin gef 的文件路径
        """
        self.file_path = file_path
        self._sn = None
        self._matrix_type = os.path.basename(self.file_path).split(".")[1]
        ### class 类内部部访问数据
        self._stereo_exp = None  ### raw stereo_exp
        self._cluster_exp = None
        self.matrix_type=matrix_type

    @property
    def raw_data(self):
        if self._stereo_exp == None:
            self._stereo_exp = self.read(self.file_path)
            self._stereo_exp.cells["MID_counts"] = np.sum(self.raw_data.exp_matrix, axis=1)
        return self._stereo_exp

    @property
    def cluster_data(self):


        if self._cluster_exp == None:
            import copy
            self._cluster_exp = copy.deepcopy(self.raw_data)
            if self.matrix_type == TechType.Transcriptomics:

                self._cluster_exp = self._cluster_exp.tl.filter_cells(
                    min_counts=100,
                    min_genes=10,
                    max_genes=2500,
                    pct_counts_mt=5,
                )
            self._cluster_exp.tl.raw_checkpoint()
            self._cluster_exp.tl.normalize_total()
            self._cluster_exp.tl.log1p()
            if self.matrix_type == TechType.Transcriptomics:
                self._cluster_exp.tl.highly_variable_genes(
                    min_mean=0.0125,
                    max_mean=3,
                    min_disp=0.5,
                    n_top_genes=5000,
                    res_key='highly_variable_genes'
                )
                self._cluster_exp.plt.highly_variable_genes(res_key='highly_variable_genes')
                self._cluster_exp.tl.scale()
                self._cluster_exp.tl.pca(
                    use_highly_genes=True,
                    n_pcs=30,
                    res_key='pca'
                )
            elif self.matrix_type == TechType.Protein:
                # self._cluster_exp.tl.scale()
                self._cluster_exp.tl.pca(
                    n_pcs=30,
                    res_key='pca'
                )
            self._cluster_exp.tl.neighbors(

                pca_res_key='pca',
                n_pcs=30,
                res_key='neighbors'
            )
            # compute spatial neighbors
            self._cluster_exp.tl.spatial_neighbors(
                neighbors_res_key='neighbors',
                res_key='spatial_neighbors'
            )
            self._cluster_exp.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap')
            self._cluster_exp.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')
            # self._cluster_exp.tl.find_marker_genes(
            #     cluster_res_key='leiden',
            #     method='t_test',
            #     use_highly_genes=False,
            #     use_raw=True
            # )
        return self._cluster_exp

    @property
    def sn(self):
        if self._sn == None:
            self._sn = self.raw_data.sn
        return self._sn

    @property
    def cell_diameter(self):
        if "cell_diameter" not in self.raw_data.cells.obs.columns:
            if "area" not in self.raw_data.cells.obs.columns:
                raise Exception("No area result in .gef")
            else:
                self.raw_data.cells["cell_diameter"] = (2 * np.sqrt(self.raw_data.cells["area"].to_numpy() / np.pi)
                                                        * self.raw_data.resolution / 1000)
        return self.raw_data.cells["cell_diameter"]

    @property
    def cell_n_gene(self):
        if "n_gene" not in self.raw_data.cells.obs.columns:
            self.raw_data.cells["n_gene"] = self.raw_data.exp_matrix.getnnz(axis=1)
        return self.raw_data.cells["n_gene"]

    @property
    def cell_MID_counts(self):
        if "MID_counts" not in self.raw_data.cells.obs.columns:
            self.raw_data.cells["MID_counts"] = np.sum(self.raw_data.exp_matrix, axis=1)
        return self.raw_data.cells["MID_counts"]

    def reset(self):
        """
        释放内存
        """
        del self._stereo_exp  ### raw stereo_exp
        del self._cluster_exp

    @property
    def shape(self):
        return self.raw_data.shape

    def read(self, gef_path):
        if gef_path.endswith(".gem") or gef_path.endswith(".txt"):
            from stereo.io import read_gem
            print("This is CellBin .gem file;")
            data = read_gem(gef_path, bin_type='cell_bins', sep="\t", is_sparse=True)
            print(data)
            return data
        elif gef_path.endswith(".gef"):
            from stereo.io import read_gef, read_gef_info
            print("the gef file is: ", read_gef_info(self.file_path))
            data = read_gef(gef_path, bin_type="cell_bins")
            ## print(data)
            return data
        elif gef_path.endswith(".h5ad"):
            try:
                from stereo.io import read_stereo_h5ad
                data = read_stereo_h5ad(gef_path, bin_type="cell_bins")
                print(data)
                return data
            except:
                from stereo.io import read_ann_h5ad
                data = read_ann_h5ad(gef_path, spatial_key="spatial", bin_type="cell_bins")
                print("This is CellBin anndata file")
                return data

    def get_cellcount(self):
        return self.raw_data.n_cells

    def get_cellarea(self):
        return str(self.raw_data.cells["area"].mean().astype(np.int16)), str(
            self.raw_data.cells["area"].median().astype(np.int16))

    def get_genetype(self):
        return str(self.cell_n_gene.to_numpy().mean().astype(np.int16)), str(
            np.median(self.cell_n_gene.to_numpy()).astype(np.int16))

    def get_MID(self):
        return (str(self.cell_MID_counts.to_numpy().mean().astype(np.int16)),
                str(np.median(self.cell_MID_counts.to_numpy()).astype(np.int16)))

    def get_total_MID(self):
        return self.raw_data._exp_matrix.sum()

    def get_faction_cell_gene(self, threshod=200):
        return np.sum(self.cell_n_gene.to_numpy() > threshod) / self.get_cellcount()

    def _plt_vio(self, data, title="", xlabel="", color="Blues", save_path=r"./"):
        # fig = plt.figure()
        fig, ax = plt.subplots()
        q1 = data.quantile(0.25)
        q2 = data.quantile(0.50)  # 中位数
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        data_no_outliers = data[data <= q3 + 2 * iqr]
        sns.violinplot(y=data_no_outliers, color=color, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        fig.savefig(save_path)
        plt.close()
        return [str(round(data.min(), 3)), str(round(q1, 3)),
                str(round(q2, 3)), str(round(q3, 3)), str(round(data.max(), 3))]

    def plot_statistic_vio(self, save_path):

        MID_file = os.path.join(save_path, f"MIDCount.png")
        celldiameter_file = os.path.join(save_path, f"CellArea.png")
        genetype_file = os.path.join(save_path, f"GeneType.png")

        celldiameter_data = self._plt_vio(self.cell_diameter,
                                          title="Univariate distribution of Cell Diameter(\xb5m)",
                                          xlabel="Cell Diameter",
                                          save_path=celldiameter_file, color=mcolors.TABLEAU_COLORS["tab:blue"])

        ngene_data = self._plt_vio(self.cell_n_gene,
                                   title="Univariate distribution of Gene Type",
                                   xlabel="Gene Type",
                                   save_path=genetype_file, color=mcolors.TABLEAU_COLORS["tab:orange"])
        MID_data = self._plt_vio(self.cell_MID_counts,
                                 title="Univariate distribution of MID counts",
                                 xlabel="MID counts",
                                 save_path=MID_file, color=mcolors.TABLEAU_COLORS["tab:green"])

        return {"celldiameter": celldiameter_data, "genetype": ngene_data, "MID": MID_data}

    def get_cluster_data(self, reset=True):
        _temp_df = pd.DataFrame()
        _temp_df["x"], _temp_df["y"] = self.cluster_data.position[:, 0], self.cluster_data.position[:, 1]
        _temp_df["umap_0"], _temp_df["umap_1"] = self.cluster_data.cells_matrix["umap"][0], \
                                                 self.cluster_data.cells_matrix["umap"][1]
        _temp_df["leiden"] = self.cluster_data.cells.obs["leiden"].tolist()
        # if save_path is not None:
        #     self.write_h5ad(save_path)
        if reset is True:
            self.reset()
        return _temp_df

    @property
    def celldensity(self, radiu=200):  ## 200 pixel = 100 um
        if "cell_density" not in self.raw_data.cells.obs.columns:
            from scipy.spatial import cKDTree
            tree = cKDTree(self.raw_data.position)
            nearby_points_count = np.array([len(tree.query_ball_point(point, r=radiu)) - 1 for point in self.raw_data.position])
        return nearby_points_count


    def plot_spatail_figure(self, colormin=None, colormax=None, key="celldensity", save_path='./'):
        _temp_df = pd.DataFrame()
        _temp_df["x"], _temp_df['y'] = self.raw_data.position[:, 0], self.raw_data.position[:, 1]
        if key == "celldensity":
            _temp_df["value"] = self.celldensity
            title = "Cells Density"
        elif key == "CellBin_MID_counts":
            _temp_df["value"] = self.cell_MID_counts.to_numpy()
            title = "CellBin Single Cell MID Counts"
        elif key == "Adjusted_Gene_type":
            _temp_df["value"] = self.cell_MID_counts.to_numpy()
            title = "Adjusted Seg Single Cell MID Counts"
        if colormin is not None and colormax is not None:
            scatter = plt.scatter(_temp_df["x"], _temp_df["y"], c=_temp_df["value"], cmap='viridis', s=5,
                                  vmin=colormin, vmax=colormax)
        else:
            scatter = plt.scatter(_temp_df["x"], _temp_df["y"], c=_temp_df["value"], cmap='viridis', s=5)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        color_min, color_max = scatter.get_clim()
        name = "celldensity" if "density" in key else "MID_counts"
        plt.savefig(os.path.join(save_path, name + '.png'))
        plt.close()
        return color_min, color_max

    def write_h5ad(self, save_path: str):
        from stereo.io import write_h5ad
        outkey_record = {'cluster': ['leiden']}
        write_h5ad(
            self.cluster_data,
            use_raw=False,
            use_result=True,
            key_record=outkey_record,
            output=os.path.join(save_path, f"{self.sn}_cluster.h5ad"),
        )

    def write_gef(self, save_path: str):
        from stereo.io import write_mid_gef
        write_mid_gef(self.raw_data, save_path)
        pass


class BinMatrix(object):
    def __init__(self, file_path: str, bin_read=100):
        self._file_path = file_path
        self._stereo_exp = None  ### raw stereo_exp
        self._bin_read = bin_read

    def reset(self):
        self._stereo_exp = None

    @property
    def stereo_exp(self):
        if self._stereo_exp == None:
            self._stereo_exp = self.read(self._bin_read)
        return self._stereo_exp

    @property
    def _width_height(self):
        from stereo.io import read_gef, read_gef_info
        infor = read_gef_info(self._file_path)
        return (infor["width"], infor["height"])

    def reset(self, bin_read):
        self._stereo_exp = None
        self._bin_read = bin_read

    def read(self, bin_size=100):
        if self._file_path.endswith(".gef"):
            from stereo.io import read_gef, read_gef_info
            data = read_gef(self._file_path, bin_size=bin_size)
        return data

    def get_total_MID(self):
        return self.stereo_exp._exp_matrix.sum()

    @property
    def MID_counts(self):
        return np.array(np.sum(self.stereo_exp.exp_matrix, axis=1))

    def create_heatmap(self, plot_data):
        import io
        heatmap_array= np.zeros(
            (self._width_height[1] // self._bin_read + 1, self._width_height[0] // self._bin_read + 1), dtype=np.uint16)
        df = pd.DataFrame()
        df["x"], df["y"] = self.stereo_exp.position[:, 0] // self._bin_read, self.stereo_exp.position[:,
                                                                             1] // self._bin_read
        df["plot_data"] = plot_data
        heatmap_array[df['y'].astype('uint32'), df['x'].astype('uint32')] = df['plot_data'].to_numpy()
        from matplotlib import colors, cm, ticker
        cmap = colors.LinearSegmentedColormap.from_list("stomics_cmap",
                                                        [(0, '#0c3383'), (0.11, '#005ea3'), (0.22, '#0a88ba'),
                                                         (0.33, '#00c199'), (0.44, '#f2d338'), (0.55, '#f6b132'),
                                                         (0.66, '#f28f38'), (0.77, '#f48f38'), (0.88, '#d91e1e'),
                                                         (1.0, '#d91e1e')])
        cmap.set_under('k', alpha=0)
        plt.imsave('./temp.png', heatmap_array, vmin=1, cmap=cmap)

        with open('./temp.png', "rb") as f:
            img_b = f.read()
            b = io.BytesIO(img_b)  ## heatmap img
        os.remove('./temp.png')
        fig, ax = plt.subplots(figsize=(1, 1))
        norm = colors.Normalize(vmin=0, vmax=df['plot_data'].max())
        im = cm.ScalarMappable(norm=norm, cmap=cmap)

        def human_format(num, pos):
            num = float('{:.3g}'.format(num))
            magnitude = 0
            while abs(num) >= 1000:
                magnitude += 1
                num /= 1000.0
            return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

        cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(human_format))
        ax.remove()
        plt.savefig('./temp.png', bbox_inches='tight')
        with open('./temp.png', "rb") as f:
            img_c = f.read()
            c = io.BytesIO(img_c)  # color bar
        os.remove('./temp.png')

        return ('data:image/png;base64,{}'.format(base64.b64encode(b.getvalue()).decode()),
                'data:image/png;base64,{}'.format(base64.b64encode(c.getvalue()).decode()))


class MultiMatrix(object):
    """ 联合管理多个矩阵：可能需要 """

    def __init__(self, cellbin_path, adjusted_path, tissuegef_path, raw_path, matrix_type: TechType):

        self.cellbin = None
        self.tissuebin = None
        self.rawbin = None
        self.adjustedbin = None
        self.matrix_type = matrix_type

        if not cellbin_path == "":
            self.cellbin = cbMatrix(cellbin_path,matrix_type=self.matrix_type)
        if not tissuegef_path == "":
            self.tissuebin = BinMatrix(tissuegef_path, bin_read=10)
        if not raw_path == "":
            self.rawbin = BinMatrix(raw_path, bin_read=10)
        if not adjusted_path == "":
            self.adjustedbin = cbMatrix(adjusted_path,matrix_type=self.matrix_type)

    def read(self, file_path: str):
        pass

    def write(self, file_path: str):
        pass

    def print_info(self, ):
        pass


def main():
    path = "Z:\MySyncFiles"
    geffile = os.path.join(path, 'D04167E2.cellbin.txt')
    cb = cbMatrix(geffile)
    print(cb.shape)


if __name__ == '__main__':
    main()
