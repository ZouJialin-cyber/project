from pydantic import BaseModel, Field
from typing import List, Any, Tuple, Union
import numpy as np

from cellbin2.utils.common import TechType
from cellbin2.image import CBImage, cbimread
from cellbin2.utils.common import bPlaceHolder, fPlaceHolder, iPlaceHolder, sPlaceHolder


class ChipFeature(object):
    def __init__(self):
        self.tech_type: TechType = TechType.ssDNA
        self._chip_box: ChipBoxInfo = ChipBoxInfo()
        self._template: TemplateInfo = TemplateInfo()
        # self._index_points: np.ndarray = np.array([])
        self._point00: Tuple[int, int] = (0, 0)  # xy,相对于芯片而不是矩阵的位置坐标
        self._mat: Union[str, np.ndarray, CBImage] = ''
        # self.ref: List[List, List] = [[]]
        self._anchor_point: Tuple[int, int] = (0, 0)  # xy, 用于配准前置的辅助锚点

    @property
    def chip_box(self, ):
        return self._chip_box

    def set_chip_box(self, chip_box):
        self._chip_box = chip_box

    def set_point00(self, points):
        if isinstance(points, tuple) and len(points) == 2:
            self._point00 = points

    def set_anchor_point(self, points):
        if isinstance(points, tuple) and len(points) == 2:
            self._anchor_point = points

    @property
    def anchor_point(self, ):
        return self._anchor_point

    @property
    def point00(self, ):
        return self._point00

    @property
    def mat(self, ):
        return self._mat

    def set_mat(self, mat):
        if not isinstance(mat, CBImage):
            self._mat = cbimread(mat)
        else:
            self._mat = mat

    @property
    def template(self, ):
        return self._template

    def set_template(self, template):
        if not isinstance(template, TemplateInfo):
            self._template.template_points = template
        else:
            self._template = template


class TrackPointsInfo(BaseModel):
    track_points: dict = Field(dict(), description="识别到的track点坐标信息")
    good_fov_count: int = Field(iPlaceHolder, description="track点数目在某个数目之上的FOV数目")
    score: float = Field(fPlaceHolder, description="得分")
    fov_location: Any = Field(description='全部FOV坐标')


class TemplateInfo(BaseModel):
    template_points: np.ndarray = Field(np.array([]), description="推导出的所有模板点")
    template_recall: float = Field(fPlaceHolder, description="识别到的track点可以匹配回模板的占比")
    template_valid_area: float = Field(fPlaceHolder, description="track点分布的面积占比")
    trackcross_qc_pass_flag: int = Field(iPlaceHolder, description="track点数目及分布是否达到要求")
    trackline_channel: int = Field(iPlaceHolder, description="检测track line的通道")
    rotation: float = Field(fPlaceHolder, description="track line的水平角")
    scale_x: float = Field(fPlaceHolder, description="水平尺度")
    scale_y: float = Field(fPlaceHolder, description="竖直尺度")

    class Config:
        arbitrary_types_allowed = True


class CalibrationInfo(BaseModel):
    score: float = Field(fPlaceHolder, description='')
    offset: List[float] = Field([fPlaceHolder, fPlaceHolder], description='')
    scale: float = Field(fPlaceHolder, description='')
    rotate: float = Field(fPlaceHolder, description='')
    pass_flag: int = Field(iPlaceHolder, description='')


class ChipBoxInfo(BaseModel):
    LeftTop: List[float] = Field([fPlaceHolder, fPlaceHolder], description='左上角XY')
    LeftBottom: List[float] = Field([fPlaceHolder, fPlaceHolder], description='左下角XY')
    RightTop: List[float] = Field([fPlaceHolder, fPlaceHolder], description='右上角XY')
    RightBottom: List[float] = Field([fPlaceHolder, fPlaceHolder], description='右下角XY')
    ScaleX: float = Field(fPlaceHolder, description='相对固定图的X尺度')
    ScaleY: float = Field(fPlaceHolder, description='相对固定图的Y尺度')
    ChipSize: Tuple[float, float] = Field((fPlaceHolder, fPlaceHolder), description='芯片宽高')
    Rotation: float = Field(fPlaceHolder, description='芯片放置角度')
    IsAvailable: bool = Field(bPlaceHolder, description='是否建议下游流程使用该参数')

    @property
    def chip_box(self) -> np.ndarray:
        """以左上 左下 右下 右上排布

        Returns:

        """
        points = np.array([self.LeftTop,
                           self.LeftBottom,
                           self.RightBottom,
                           self.RightTop])
        return points

    def set_chip_box(
            self, cb: np.ndarray
    ) -> None:
        self.LeftTop, self.LeftBottom, self.RightBottom, self.RightTop = \
            [list(i) for i in cb]


class CellSegInfo(BaseModel):
    mask: Any = Field(None, description='')
    fast_mask: Any = Field(None, description='')


class TissueSegOutputInfo(BaseModel):
    tissue_mask: Any = Field(None, description='输出的组织分割mask')
    threshold_list: Any = Field(None, description='返回的阈值列表')
