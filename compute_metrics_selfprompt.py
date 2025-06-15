import numpy as np
import os
from os.path import join
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import argparse
from scipy.spatial.distance import directed_hausdorff
from skimage.transform import resize

from SurfaceDice import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
    compute_dice_coefficient
)

# ---------------- 参数设置 ----------------
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seg_dir', required=True, type=str, help='Path to predicted npz files.')
parser.add_argument('-g', '--gt_dir', required=True, type=str, help='Path to GT npz files.')
parser.add_argument('-csv_dir', required=True, type=str, help='Output CSV directory.')
parser.add_argument('--label', type=int, default=1, help='Target label in GT to evaluate (e.g., 1 for disc)')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--nsd', action='store_true', help='Whether to compute NSD (surface dice)')
args = parser.parse_args()

seg_dir = args.seg_dir
gt_dir = args.gt_dir
csv_path = os.path.join(args.csv_dir, 'metrics.csv')
target_label = args.label
num_workers = args.num_workers
compute_NSD = args.nsd

# ---------------- 基础指标函数 ----------------
def compute_iou(gt, seg):
    intersection = np.logical_and(gt, seg).sum()
    union = np.logical_or(gt, seg).sum()
    return intersection / union if union > 0 else 0.0

def compute_hd(gt, seg):
    gt_pts = np.argwhere(gt)
    seg_pts = np.argwhere(seg)
    if gt_pts.size == 0 or seg_pts.size == 0:
        return np.inf
    hd1 = directed_hausdorff(gt_pts, seg_pts)[0]
    hd2 = directed_hausdorff(seg_pts, gt_pts)[0]
    return max(hd1, hd2)

# ---------------- 主评估函数 ----------------
def compute_metrics(npz_name):
    result = {
        'dsc': -1.0,
        'iou': -1.0,
        'hd': -1.0,
        'nsd': -1.0 if compute_NSD else None
    }

    try:
        seg_npz = np.load(join(seg_dir, npz_name), allow_pickle=True)
        gt_npz  = np.load(join(gt_dir,  npz_name), allow_pickle=True)

        gts = gt_npz['gts']
        segs = seg_npz['pred_mask']

        # 二值化预测掩码
        segs = (segs > 0.5).astype(np.uint8)

        # 自动对齐预测掩码大小
        if gts.shape != segs.shape:
            # print(f"[WARN] Shape mismatch in {npz_name}: gts {gts.shape}, pred {segs.shape} → resizing")
            segs = resize(segs, gts.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

        # 只取 GT 中的 target_label 区域
        gt_mask = (gts == target_label).astype(np.uint8)
        pred_mask = segs  # pred 是 0/1 前景预测

        ndim = gt_mask.ndim
        if 'spacing' in gt_npz:
            spacing = list(gt_npz['spacing'])
        else:
            spacing = [1.0] * ndim
        spacing += [1.0] * (3 - len(spacing))
        spacing = spacing[:3]

        # 添加通道维度
        if ndim == 2:
            gt_mask = gt_mask[..., np.newaxis]
            pred_mask = pred_mask[..., np.newaxis]

        dsc = compute_dice_coefficient(gt_mask, pred_mask)
        iou = compute_iou(gt_mask, pred_mask)
        hd  = compute_hd(gt_mask, pred_mask)

        result['dsc'] = dsc
        result['iou'] = iou
        result['hd']  = hd

        if compute_NSD and dsc > 0.2:
            surface_dist = compute_surface_distances(gt_mask, pred_mask, spacing_mm=spacing)
            nsd = compute_surface_dice_at_tolerance(surface_dist, tolerance_mm=2.0)
            result['nsd'] = nsd

    except Exception as e:
        print(f"[ERROR] {npz_name}: {e}")

    return npz_name, result

# ---------------- 主执行逻辑 ----------------
if __name__ == '__main__':
    os.makedirs(args.csv_dir, exist_ok=True)
    npz_files = [f for f in os.listdir(gt_dir) if f.endswith('.npz')]

    seg_metrics = {'case': [], 'dsc': [], 'iou': [], 'hd': []}
    if compute_NSD:
        seg_metrics['nsd'] = []

    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_files)) as pbar:
            for npz_name, metrics in pool.imap_unordered(compute_metrics, npz_files):
                seg_metrics['case'].append(npz_name)
                seg_metrics['dsc'].append(round(metrics['dsc'], 4))
                seg_metrics['iou'].append(round(metrics['iou'], 4))
                seg_metrics['hd'].append(round(metrics['hd'], 4))
                if compute_NSD:
                    seg_metrics['nsd'].append(round(metrics['nsd'], 4))
                pbar.update()

    df = pd.DataFrame(seg_metrics).sort_values(by='case')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Evaluation finished. Results saved to: {csv_path}")
