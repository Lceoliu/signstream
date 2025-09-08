__all__ = [
    "COCO_EDGES",
    "BODY_PARTS_INTERVALS",
    "CENTER_INDICES",
    "DEFAULT_PROCESS_INFO",
    "normalize_by_global_bbox",
    "interpolate_low_confidence_linear",
    "compute_velocity",
    "build_skeleton_edges",
    "derive_edge_features",
    "split_body_parts",
    "process_all",
    "generate_video",
]
import torch
from typing import Literal, Optional, Tuple, List, Dict
from dataclasses import dataclass
import numpy as np

COCO_EDGES = {
    'body': [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
    ],
    'face': [
        [23, 24],
        [24, 25],
        [25, 26],
        [26, 27],
        [27, 28],
        [28, 29],
        [29, 30],
        [30, 31],
        [31, 32],
        [32, 33],
        [33, 34],
        [34, 35],
        [35, 36],
        [36, 37],
        [37, 38],
        [38, 39],
        [40, 41],
        [41, 42],
        [42, 43],
        [43, 44],
        [45, 46],
        [46, 47],
        [47, 48],
        [48, 49],
        [50, 51],
        [51, 52],
        [52, 53],
        [54, 55],
        [55, 56],
        [56, 57],
        [57, 58],
        [59, 60],
        [60, 61],
        [61, 62],
        [62, 63],
        [63, 64],
        [64, 59],
        [65, 66],
        [66, 67],
        [67, 68],
        [68, 69],
        [69, 70],
        [70, 65],
        [71, 72],
        [72, 73],
        [73, 74],
        [74, 75],
        [75, 76],
        [76, 77],
        [77, 78],
        [78, 79],
        [79, 80],
        [80, 81],
        [81, 71],
        [82, 83],
        [83, 84],
        [84, 85],
        [85, 86],
        [86, 87],
        [87, 88],
        [88, 89],
        [89, 90],
        [90, 82],
    ],
    'right_hand': [
        [112, 113],
        [113, 114],
        [114, 115],
        [115, 116],
        [112, 117],
        [117, 118],
        [118, 119],
        [119, 120],
        [112, 121],
        [121, 122],
        [122, 123],
        [123, 124],
        [112, 125],
        [125, 126],
        [126, 127],
        [127, 128],
        [112, 129],
        [129, 130],
        [130, 131],
        [131, 132],
    ],
    'left_hand': [
        [91, 92],
        [92, 93],
        [93, 94],
        [94, 95],
        [91, 96],
        [96, 97],
        [97, 98],
        [98, 99],
        [91, 100],
        [100, 101],
        [101, 102],
        [102, 103],
        [91, 104],
        [104, 105],
        [105, 106],
        [106, 107],
        [91, 108],
        [108, 109],
        [109, 110],
        [110, 111],
    ],
}

BODY_PARTS_INTERVALS = {
    'face': (23, 91),
    'left_hand': (91, 112),
    'right_hand': (112, 133),
    'body': (0, 17),
    'full_body': (0, 133),
}

CENTER_INDICES = {
    'body': 0,
    'face': 31,
    'left_hand': 91,
    'right_hand': 112,
}


DEFAULT_PROCESS_INFO = {
    "confidence_threshold": 0.25,
    "bbox_range": (8.0, 8.0),
    "keep_aspect": False,
    "eps": 1e-9,
}


def normalize_by_global_bbox(
    pose: torch.Tensor,  # [T, K, 3]
    bbox_range: Tuple[float, float] = (1.0, 1.0),
    conf_threshold: Optional[float] = None,  # 若给定，只用高置信点估计 bbox
    keep_aspect: bool = False,  # True=等比+留边
    eps: float = 1e-9,
):
    """
    返回:
      pose_out:  归一化后的 [T, K, 3]（只改 x,y；conf 保留）
      info: dict 包含 x_min/x_max/y_min/y_max/scale/offset/pad 等调试参数
    """
    assert pose.ndim == 3 and pose.shape[-1] == 3
    device, dtype = pose.device, pose.dtype
    T, K, _ = pose.shape
    xy = pose[..., :2]
    conf = pose[..., 2]

    if conf_threshold is None:
        valid = torch.ones(T, K, dtype=torch.bool, device=device)
    else:
        valid = conf >= conf_threshold

    # 用 where + (+/-inf) 做掩码 min/max
    x = xy[..., 0]
    y = xy[..., 1]
    big = torch.tensor(float('inf'), device=device, dtype=dtype)
    nbig = torch.tensor(float('-inf'), device=device, dtype=dtype)

    x_min = torch.where(valid, x, big).amin(dim=(0, 1)).item()
    x_max = torch.where(valid, x, nbig).amax(dim=(0, 1)).item()
    y_min = torch.where(valid, y, big).amin(dim=(0, 1)).item()
    y_max = torch.where(valid, y, nbig).amax(dim=(0, 1)).item()

    # 若全体无有效点，退化到用所有点；再不行就设个单位盒
    if not torch.isfinite(
        torch.tensor([x_min, x_max, y_min, y_max], device=device)
    ).all():
        x_min = x.min().item()
        x_max = x.max().item()
        y_min = y.min().item()
        y_max = y.max().item()
    if abs(x_max - x_min) < eps:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < eps:
        y_max = y_min + 1.0

    rx, ry = float(bbox_range[0]), float(bbox_range[1])
    pose_out = pose.clone()
    x_new = x - x_min
    y_new = y - y_min

    if not keep_aspect:
        sx = rx / max(eps, (x_max - x_min))
        sy = ry / max(eps, (y_max - y_min))
        pose_out[..., 0] = x_new * sx
        pose_out[..., 1] = y_new * sy
        info = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "scale": (sx, sy),
            "offset": (x_min, y_min),
            "keep_aspect": False,
        }
    else:
        # 等比缩放 + 居中留边
        sx = (x_max - x_min) / max(eps, rx)
        sy = (y_max - y_min) / max(eps, ry)
        s = max(sx, sy)  # 选择让内容能放进去的比率
        s = max(s, eps)
        # 输出内容尺寸
        out_w = (x_max - x_min) / s
        out_h = (y_max - y_min) / s
        pad_x = (rx - out_w) * 0.5
        pad_y = (ry - out_h) * 0.5
        pose_out[..., 0] = x_new / s + pad_x
        pose_out[..., 1] = y_new / s + pad_y
        info = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "scale": (1.0 / s, 1.0 / s),
            "offset": (x_min, y_min),
            "pad": (pad_x, pad_y),
            "keep_aspect": True,
        }
    return pose_out, info


def interpolate_low_confidence_linear(
    pose: torch.Tensor,  # [T, K, 3]
    confidence_threshold: float = 0.1,
    max_search: int = 10,
    set_conf_zero: bool = True,
):
    """
    返回:
      pose_filled: [T, K, 3] 补点后的结果（x,y 被替换；conf 视 set_conf_zero 可能被清零）
      fill_mask:   [T, K]    True 表示该点是插值/填充出来的
    """
    assert pose.ndim == 3 and pose.shape[-1] == 3
    T, K, _ = pose.shape
    device, dtype = pose.device, pose.dtype

    out = pose.clone()
    xy = out[..., :2]
    conf = out[..., 2]
    valid = conf >= confidence_threshold
    fill_mask = torch.zeros(T, K, dtype=torch.bool, device=device)

    # 逐关键点处理（清晰直观，O(T*K)）
    for k in range(K):
        vk = valid[:, k]  # [T]
        xk = xy[:, k, 0]
        yk = xy[:, k, 1]
        ck = conf[:, k]

        # 预先拿到所有有效帧索引
        idx_valid = torch.nonzero(vk, as_tuple=False).flatten()
        if idx_valid.numel() == 0:
            # 全无有效：全置 0
            xk.zero_()
            yk.zero_()
            if set_conf_zero:
                ck.zero_()
            fill_mask[:, k] = True
            continue

        for t in range(T):
            if vk[t]:
                continue  # 本帧有效，跳过
            # 向左找
            t_left = None
            for dt in range(1, max_search + 1):
                tl = t - dt
                if tl < 0:
                    break
                if vk[tl]:
                    t_left = tl
                    break
            # 向右找
            t_right = None
            for dt in range(1, max_search + 1):
                tr = t + dt
                if tr >= T:
                    break
                if vk[tr]:
                    t_right = tr
                    break

            if (t_left is not None) and (t_right is not None):
                # 线性插值
                w = (t - t_left) / float(t_right - t_left)
                xk[t] = (1 - w) * xk[t_left] + w * xk[t_right]
                yk[t] = (1 - w) * yk[t_left] + w * yk[t_right]
                fill_mask[t, k] = True
                if set_conf_zero:
                    ck[t] = 0.0
            elif (t_left is not None) or (t_right is not None):
                # 最近邻
                src = t_left if t_left is not None else t_right
                xk[t] = xk[src]
                yk[t] = yk[src]
                fill_mask[t, k] = True
                if set_conf_zero:
                    ck[t] = 0.0
            else:
                # 完全无邻居：置零
                xk[t] = 0.0
                yk[t] = 0.0
                fill_mask[t, k] = True
                if set_conf_zero:
                    ck[t] = 0.0

    return out, fill_mask


def compute_velocity(
    pose: torch.Tensor,  # [T, K, 3]
    dt: float = 1.0,  # 帧间隔（单位自定，e.g., s）
):
    """
    返回:
      vel: [T, K, 2]，最后一帧 = 倒数第二帧的速度
    """
    assert pose.ndim == 3 and pose.shape[-1] == 3
    xy = pose[..., :2]  # [T, K, 2]
    T, K, _ = xy.shape
    vel = torch.zeros(T, K, 2, device=pose.device, dtype=pose.dtype)
    vel[:-1] = (xy[1:] - xy[:-1]) / dt
    if T >= 2:
        vel[-1] = vel[-2]
    return vel


def build_adj_from_E2(E2, K, bidir=True, norm="sym"):
    """
    E2: np.ndarray [E, 2]，相对索引 (i, j)
    K : 该 part 的节点数
    返回 A[K,K] (float32)
    norm: "sym" => D^{-1/2} A D^{-1/2}, "row" => D^{-1} A, None => 原始0/1
    """
    A = np.zeros((K, K), dtype=np.float32)
    for i, j in E2:
        A[i, j] = 1.0
        if bidir:
            A[j, i] = 1.0
    if norm is None:
        return A
    deg = A.sum(axis=1, keepdims=True) + 1e-8
    if norm == "row":
        A = A / deg
    elif norm == "sym":
        d_sqrt_inv = 1.0 / np.sqrt(deg)
        A = (A * d_sqrt_inv) * d_sqrt_inv.transpose(1, 0)
    return A.astype(np.float32)


def build_skeleton_edges(
    edges: List[Tuple[int, int]],  # 例如 [(5,6),(5,7),(7,9),(6,8),(8,10)]
    num_keypoints: Optional[int] = None,
    symmetric: bool = False,  # True 则构建无向邻接（i<->j）
):
    """
    返回:
      E2:   LongTensor [E, 2]，每行 (i, j)
      A:    邻接矩阵 [K, K]（若给定 num_keypoints，否则 None）
      Inc:  入射矩阵 [E, K]，每行 e 对应 +1@i, -1@j（若给定 num_keypoints，否则 None）
    """
    E2 = torch.tensor(edges, dtype=torch.long)
    K = num_keypoints if num_keypoints is not None else int(E2.max().item() + 1)

    A = None
    Inc = None
    if num_keypoints is not None:
        A = torch.zeros(K, K, dtype=torch.float32)
        for i, j in edges:
            A[i, j] = 1.0
            if symmetric:
                A[j, i] = 1.0
        Inc = torch.zeros(len(edges), K, dtype=torch.float32)
        for e, (i, j) in enumerate(edges):
            Inc[e, i] = +1.0
            Inc[e, j] = -1.0
    return E2, A, Inc


def derive_edge_features(
    pose: torch.Tensor,  # [T, K, 3]
    E2: torch.Tensor,  # [E, 2]
    conf_threshold: float = 0.1,
    unit_vector: bool = False,  # True 则输出单位方向向量
    eps: float = 1e-9,
):
    """
    返回:
      edge_vec:   [T, E, 2]   (p_j - p_i)
      edge_len:   [T, E]      ||vec||
      edge_valid: [T, E]      两端点 conf>=阈值
    """
    assert pose.ndim == 3 and pose.shape[-1] == 3
    T, K, _ = pose.shape
    E = E2.shape[0]
    xy = pose[..., :2]  # [T, K, 2]
    conf = pose[..., 2]  # [T, K]

    i = E2[:, 0].view(1, E, 1).expand(T, E, 1)  # [T,E,1]
    j = E2[:, 1].view(1, E, 1).expand(T, E, 1)

    pi = torch.gather(xy, dim=1, index=i.expand(T, E, 2))  # [T,E,2]
    pj = torch.gather(xy, dim=1, index=j.expand(T, E, 2))  # [T,E,2]
    edge_vec = pj - pi  # [T,E,2]
    edge_len = torch.linalg.norm(edge_vec, dim=-1)  # [T,E]

    ci = torch.gather(conf, dim=1, index=i.squeeze(-1))  # [T,E]
    cj = torch.gather(conf, dim=1, index=j.squeeze(-1))  # [T,E]
    edge_valid = (ci >= conf_threshold) & (cj >= conf_threshold)

    if unit_vector:
        edge_vec = edge_vec / torch.clamp(edge_len.unsqueeze(-1), min=eps)

    return edge_vec, edge_len, edge_valid


def split_body_parts(
    pose: torch.Tensor,  # [T, K, 3]
    parts: List[str] = ['body', 'face', 'left_hand', 'right_hand', 'full_body'],
    partial_bbox: bool = False,
    partial_bbox_info: Optional[Dict[str, Dict]] = None,
) -> Dict[str, torch.Tensor]:
    """
    根据 BODY_PARTS_INTERVALS 切分身体各部分关键点
    返回:
      dict: key=part name, value=pose tensor [T, K_part, 3]
    """
    result = {}
    for part in parts:
        if part not in BODY_PARTS_INTERVALS:
            raise ValueError(f"Unknown body part: {part}")
        start, end = BODY_PARTS_INTERVALS[part]
        partial = pose[:, start:end, :].clone()
        if partial_bbox:
            partial, _ = normalize_by_global_bbox(
                partial,
                bbox_range=(
                    partial_bbox_info.get(part, {}).get("bbox_range", (8, 8))
                    if partial_bbox_info
                    else (8, 8)
                ),
                conf_threshold=(
                    partial_bbox_info.get(part, {}).get("conf_threshold", 0.25)
                    if partial_bbox_info
                    else 0.25
                ),
                keep_aspect=(
                    partial_bbox_info.get(part, {}).get("keep_aspect", True)
                    if partial_bbox_info
                    else True
                ),
                eps=(
                    partial_bbox_info.get(part, {}).get("eps", 1e-9)
                    if partial_bbox_info
                    else 1e-9
                ),
            )
        result[part] = partial
    return result


def process_all(
    pose: torch.Tensor,
    fps: int = 30,
    confidence_threshold: float = 0.25,
    bbox_range: Tuple[float, float] = (8.0, 8.0),
    keep_aspect: bool = False,
    eps: float = 1e-9,
    max_search: int = 10,
):
    """Process all body parts from the pose tensor.

    Args:
        pose (torch.Tensor): The input pose tensor of shape [T, K, 3].

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the processed pose tensors for each body part.
    """
    T, K, _ = pose.shape
    if K == 134:
        H, W = pose[0, 0, 0], pose[0, 0, 1]
        pose = pose[:, 1:, :]  # 去掉第一个点（frame size）
        video_size = (H, W)
    elif K == 133:
        video_size = (512, 512)

    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose).float()
    device, dtype = pose.device, pose.dtype
    pose_norm, _ = normalize_by_global_bbox(
        pose,
        bbox_range=bbox_range,
        conf_threshold=confidence_threshold,
        keep_aspect=keep_aspect,
        eps=eps,
    )
    pose_interp, fill_mask = interpolate_low_confidence_linear(
        pose_norm,
        confidence_threshold=confidence_threshold,
        max_search=max_search,
        set_conf_zero=True,
    )
    edges_all = (
        COCO_EDGES['body']
        + COCO_EDGES['face']
        + COCO_EDGES['left_hand']
        + COCO_EDGES['right_hand']
    )
    E2, _, _ = build_skeleton_edges(
        edges_all,
        num_keypoints=pose.shape[1],
        symmetric=True,
    )
    velocity = compute_velocity(pose_interp, dt=1.0 / fps)
    parts = split_body_parts(
        pose_interp,
        parts=['body', 'face', 'left_hand', 'right_hand', 'full_body'],
        partial_bbox=True,
        partial_bbox_info={
            "body": {
                "bbox_range": bbox_range,
                "conf_threshold": confidence_threshold,
                "keep_aspect": keep_aspect,
                "eps": eps,
            },
            "face": {
                "bbox_range": bbox_range,
                "conf_threshold": confidence_threshold,
                "keep_aspect": keep_aspect,
                "eps": eps,
            },
            "left_hand": {
                "bbox_range": bbox_range,
                "conf_threshold": confidence_threshold,
                "keep_aspect": keep_aspect,
                "eps": eps,
            },
            "right_hand": {
                "bbox_range": bbox_range,
                "conf_threshold": confidence_threshold,
                "keep_aspect": keep_aspect,
                "eps": eps,
            },
            "full_body": {
                "bbox_range": bbox_range,
                "conf_threshold": confidence_threshold,
                "keep_aspect": keep_aspect,
                "eps": eps,
            },
        },
    )

    def get_format_part(
        part: Literal['body', 'face', 'left_hand', 'right_hand', 'full_body'],
        parts,
        velocity,
    ):
        """Get the formatted part tensor.

        formatted part likes:
        {
            'part': {
                'pose': [T, K_part, 3],
                'velocity': [T, K_part, 2],
                'E2': [E_part, 2],      # 相对索引
            }
        }
        """
        part_E2 = (
            torch.tensor(
                COCO_EDGES[part],
                dtype=torch.long,
                device=device,
            )
            if part != 'full_body'
            else torch.tensor(
                (
                    COCO_EDGES['body']
                    + COCO_EDGES['face']
                    + COCO_EDGES['left_hand']
                    + COCO_EDGES['right_hand']
                ),
                dtype=torch.long,
                device=device,
            )
        )
        for e in range(part_E2.shape[0]):
            part_E2[e, 0] -= BODY_PARTS_INTERVALS[part][0]
            part_E2[e, 1] -= BODY_PARTS_INTERVALS[part][0]
        A = build_adj_from_E2(
            part_E2.cpu().numpy(),
            K=BODY_PARTS_INTERVALS[part][1] - BODY_PARTS_INTERVALS[part][0],
        )
        part_dict = {
            'pose': parts[part],  # [T, K_part, 3]
            'velocity': velocity[
                :, BODY_PARTS_INTERVALS[part][0] : BODY_PARTS_INTERVALS[part][1], :
            ],  # [T, K_part, 2]
            'E2': part_E2,  # [E_part, 2]
        }
        return part_dict

    formatted_parts = {}
    for part in ['body', 'face', 'left_hand', 'right_hand', 'full_body']:
        formatted_parts[part] = get_format_part(part, parts, velocity)
    
    # Clean up intermediate tensors to prevent memory accumulation
    del velocity
    del parts
    del pose_norm
    del pose_interp
    del fill_mask
    
    return formatted_parts


def generate_video(
    pose: torch.Tensor,
    video_size: tuple,
    save_path: str = "visualization.mp4",
    velocity: Optional[torch.Tensor] = None,
    edge_vec: Optional[torch.Tensor] = None,
    edge_len: Optional[torch.Tensor] = None,
    E2: Optional[torch.Tensor] = None,  # [E,2]，每行 (i,j)
    edge_valid: Optional[torch.Tensor] = None,  # [T,E]，可选
):
    """
    仅作为 demo，后续可移到 visualize.py
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from tqdm import tqdm

    # 统一转为 numpy，避免 torch/np 混算
    def to_np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    pose = to_np(pose)
    velocity = to_np(velocity)
    edge_vec = to_np(edge_vec)
    edge_len = to_np(edge_len)
    E2 = to_np(E2)
    edge_valid = to_np(edge_valid)

    T, K, _ = pose.shape
    # Create a blank canvas
    H, W = video_size
    H = int(H)
    W = int(W)
    canvas_size = (512, 512)  # (height, width)
    ratio = (canvas_size[0] / H, canvas_size[1] / W)
    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        save_path, fourcc, 30, (canvas_size[1], canvas_size[0])
    )

    for t in tqdm(range(T), desc="Generating video frames"):
        frame = canvas.copy()
        # 先画点
        for j in range(K):
            x, y, conf = pose[t, j]
            x = int(x * ratio[1])
            y = int(y * ratio[0])
            if conf > 0.1:  # Only draw if confidence is high enough
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    5,
                    (0, 0, 255),
                    -1,
                )
                if velocity is not None:
                    vx, vy = velocity[t, j]
                    vx = vx * ratio[1]
                    vy = vy * ratio[0]
                    # log 缩放
                    scale = 1.5
                    vx = np.log1p(abs(vx)) * (1 if vx >= 0 else -1)
                    vy = np.log1p(abs(vy)) * (1 if vy >= 0 else -1)
                    # 可按需缩放速度向量，避免箭头过长
                    cv2.arrowedLine(
                        frame,
                        (int(x), int(y)),
                        (int(x + scale * vx), int(y + scale * vy)),
                        (255, 0, 0),
                        2,
                        tipLength=0.3,
                    )

        # 再画骨架边
        if E2 is not None:
            E = E2.shape[0]
            for e in range(E):
                i, j = int(E2[e, 0]), int(E2[e, 1])
                xi, yi, ci = pose[t, i]
                xj, yj, cj = pose[t, j]
                xi = xi * ratio[1]
                yi = yi * ratio[0]
                xj = xj * ratio[1]
                yj = yj * ratio[0]
                # 有效性：优先使用传入的 edge_valid，否则根据点置信度判定
                e_valid = True
                if edge_valid is not None:
                    e_valid = bool(edge_valid[t, e])
                else:
                    e_valid = (ci > 0.1) and (cj > 0.1)

                color = (0, 200, 0) if e_valid else (200, 200, 200)
                thickness = 2 if e_valid else 1
                cv2.line(
                    frame,
                    (int(xi), int(yi)),
                    (int(xj), int(yj)),
                    color,
                    thickness,
                )

        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved to {save_path}")


if __name__ == "__main__":
    import numpy as np

    test_data_path = r"D:\SKD\SLR\slr_new\test_datas\S000146_P0008_T00.npy"
    pose = np.load(test_data_path)
    H, W = pose[0, 0, 0], pose[0, 0, 1]
    pose = pose[:, 1:, :]  # 去掉第一个点（frame size）
    video_size = (H, W)
    # generate_video(pose, video_size)
    pose = torch.from_numpy(pose).float()

    pose_norm, info = normalize_by_global_bbox(
        pose, bbox_range=(8, 8), conf_threshold=0.25, keep_aspect=True
    )
    video_size = (8, 8)
    # generate_video(pose_norm.numpy(), video_size, save_path="visualization_norm.mp4")
    pose_interp, fill_mask = interpolate_low_confidence_linear(
        pose_norm, confidence_threshold=0.25, max_search=10, set_conf_zero=True
    )
    # generate_video(
    #     pose_interp.numpy(),
    #     video_size,
    #     save_path="visualization_interp_velocity.mp4",
    #     velocity=compute_velocity(pose_interp),
    # )
    edges_all = (
        COCO_EDGES['body']
        + COCO_EDGES['face']
        + COCO_EDGES['left_hand']
        + COCO_EDGES['right_hand']
    )
    E2, A, Inc = build_skeleton_edges(
        edges_all,
        num_keypoints=pose.shape[1],
        symmetric=True,
    )
    edge_vec, edge_len, edge_valid = derive_edge_features(
        pose_interp, E2, conf_threshold=0.25, unit_vector=True
    )
    # generate_video(
    #     pose_interp.numpy(),
    #     video_size,
    #     save_path="visualization_interp_edgevec.mp4",
    #     velocity=compute_velocity(pose_interp, dt=1.0 / 30.0),
    #     edge_vec=edge_vec,
    #     edge_len=edge_len,
    #     E2=E2,
    #     edge_valid=edge_valid,
    # )
    partial_poses = split_body_parts(pose_interp, partial_bbox=True)
    show_part = 'left_hand'  # 'body'/'face'/'left_hand'/'right_hand'/'full_body'
    assert show_part in partial_poses
    part_E2 = torch.tensor(
        COCO_EDGES[show_part],
        dtype=torch.long,
    )
    for e in range(part_E2.shape[0]):
        part_E2[e, 0] -= BODY_PARTS_INTERVALS[show_part][0]
        part_E2[e, 1] -= BODY_PARTS_INTERVALS[show_part][0]
    generate_video(
        partial_poses[show_part].numpy(),
        video_size,
        save_path="visualization_interp_face.mp4",
        velocity=compute_velocity(partial_poses[show_part], dt=1.0 / 30.0),
        edge_vec=edge_vec[
            :, BODY_PARTS_INTERVALS[show_part][0] : BODY_PARTS_INTERVALS[show_part][1]
        ],
        edge_len=edge_len[
            :, BODY_PARTS_INTERVALS[show_part][0] : BODY_PARTS_INTERVALS[show_part][1]
        ],
        E2=part_E2,
        edge_valid=edge_valid,
    )
    print("Done")
