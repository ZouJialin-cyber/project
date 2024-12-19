import json
import os
from enum import Enum
from typing import Union, List, Dict
import requests
from cellbin2.utils import clog
from tqdm import tqdm

from cellbin2.utils.ipr import sPlaceHolder


class DNNModuleName(Enum):
    cellseg = 1
    tissueseg = 2
    clarity = 3
    points_detect = 4
    chip_detect = 5


class WeightDownloader(object):
    def __init__(self, save_dir: str, url_file: str = None):
        if url_file:
            self._url_file = url_file
        else:
            curr_path = os.path.dirname(os.path.realpath(__file__))
            self._url_file = os.path.join(curr_path, r'../config/weights_url.json')
        with open(self._url_file, 'r') as fd:
            self._WEIGHTS = json.load(fd)
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        clog.info('Weights files will be stored in {}'.format(save_dir))

    @property
    def weights_list(self, ):
        w = {}
        for k in self._WEIGHTS.keys():
            w.update(self._WEIGHTS[k])
        return w

    def _download(self, weight_name: str, model_url: str):
        weight = os.path.join(self._save_dir, weight_name)
        if not os.path.exists(weight):
            try:
                r = requests.get(model_url, stream=True)
                total = int(r.headers.get('content-length', 0))
                with open(os.path.join(self._save_dir, weight_name), 'wb') as fd, tqdm(
                        desc='Downloading {}'.format(weight_name), total=total,
                        unit='B', unit_scale=True) as bar:
                    for data in r.iter_content(chunk_size=1024):
                        siz = fd.write(data)
                        bar.update(siz)
            except Exception as e:
                clog.error('FAILED! (Download {} from remote {})'.format(weight_name, model_url))
                print(e)
                return 1
        else:
            clog.info('{} already exists'.format(weight_name))

        return 0

    def download_weight_by_names(self, weight_names: List[str]):
        all_weights = self.weights_list
        for weight_name in weight_names:
            # weight = os.path.join(save_dir, weight_name)
            if weight_name not in all_weights:
                clog.error(f"{weight_name} not in auto download lists")
                continue
            model_url = all_weights[weight_name]
            flag = self._download(weight_name, model_url)
            if flag != 0:
                return flag

        return 0

    def download_weights(self, module_name: DNNModuleName, weight_name: str):
        weight = os.path.join(self._save_dir, weight_name)
        if weight_name in self._WEIGHTS[module_name.name].keys():
            model_url = self._WEIGHTS[module_name.name][weight_name]
            if not os.path.exists(weight):
                flag = self._download(weight_name, model_url)
                if flag != 0: return flag
        else:
            clog.warning('{} not in [{}] module'.format(weight_name, module_name.name))

        return 0

    def download_module_weight(self, module_name: Union[DNNModuleName, list]):
        weights_to_download = []
        if isinstance(module_name, DNNModuleName):
            weights_to_download = [module_name]
        elif isinstance(module_name, list):
            weights_to_download = module_name
        elif module_name is None:
            # 如果不给这个参数，默认就全部下载
            weights_to_download = DNNModuleName.__members__.keys()
        for module_name in weights_to_download:
            for model_name in self._WEIGHTS[module_name.name].keys():
                self.download_weights(module_name, model_name)
        return 0


if __name__ == '__main__':
    save_dir = r'E:\03.users\liuhuanlin\01.data\cellbin2\weights'
    names = [DNNModuleName.cellseg]
    wd = WeightDownloader(save_dir)

    # 通过模块下载
    wd.download_module_weight(names)

    # 通过模块/模型名字下载
    wd.download_weights(names[0], 'cellseg_bcdu_SHDI_221008_tf.onnx')
    wd.download_weights(names[0], 'points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx')

    # 通过模型名字列表下载
    wd.download_weight_by_names(['chip_detect_yolov5obb_SSDNA_20241001_pytorch.onnx',
                                 'tissueseg_yolo_SH_20230131_th.onnx'])


    """
    """


