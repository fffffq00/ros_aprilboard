# import sys
# from pathlib import Path
# __package__ = Path(__file__).parent.name
# sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from dataclasses import dataclass
from .tag_family import TAG_FAMILY_DICT

def build_board_map_grid(rows, cols, start_id, tag_size, spacing=None):
    """构造一个行列排列的 tag_board_map: id -> (start_x, start_y) 
    单位为米。约定：原点放在左下角，行索引从下往上（r=0 对应最下行），
    id 分配按行优先（从左到右），然后向上行进。
    """
    if spacing is None:
        spacing = tag_size / 2
    tag_board_map = {}
    for r in range(rows):
        y = r * (tag_size + spacing)
        for c in range(cols):
            tid = start_id + r * cols + c
            x = c * (tag_size + spacing)
            tag_board_map[tid] = (x, y)
    return tag_board_map

@dataclass
class AprilBoard:
    rows: int
    cols: int
    markerLength: float  # in meters
    spaceLength: float   # in meters
    start_id: int
    family_name: str = "t36h11"

    def __post_init__(self):
        self.end_id = self.start_id + self.rows * self.cols  #左闭右开
        self.tag_family = TAG_FAMILY_DICT[self.family_name]
        self.board_map = build_board_map_grid(
            self.rows, self.cols, self.start_id, self.markerLength, self.spaceLength
        )

        
    def matchImagePoints(self, marker_corners, marker_ids):
        imgpts_all = []  # image points
        objpts_all = []  # object points in board coordinate

        for id, corners in zip(marker_ids, marker_corners):
            if id in self.board_map:
                sx_board, sy_board = self.board_map[id]  #中心坐标，单位米
                # corners order assumed [bl, br, tr, tl] #左下 右下 左上 右上
                corners_obj = np.array([
                    [sx_board, sy_board, 0.0],
                    [sx_board + self.markerLength, sy_board, 0.0],
                    [sx_board + self.markerLength, sy_board + self.markerLength, 0.0],
                    [sx_board, sy_board + self.markerLength, 0.0],
                ], dtype=np.float64)
                imgpts_all.append(corners)
                objpts_all.append(corners_obj)

        imgpts = np.concatenate(imgpts_all).astype(np.float64)
        objpts = np.concatenate(objpts_all).astype(np.float64)
        return objpts, imgpts

