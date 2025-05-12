import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.tools import write_csv
import numpy as np
from sklearn.cluster import KMeans


def uniform_sampling(masks, N=5):
    n_points = []
    for mask in masks:
        if not isinstance(mask, np.ndarray):
            mask = mask.cpu().numpy()

        indices = np.argwhere(mask == 1) # [y, x]
        sampled_indices = np.random.choice(len(indices), N, replace=True)
        sampled_points = np.flip(indices[sampled_indices], axis=1)
        n_points.append(sampled_points.tolist())

    return n_points


def get_multi_distance_points(input_point, mask, points_nubmer):
    new_points = np.zeros((points_nubmer + 1, 2))
    new_points[0] = [input_point[1], input_point[0]]
    for i in range(points_nubmer):
        new_points[i + 1] = get_next_distance_point(new_points[:i + 1, :], mask)

    new_points = swap_xy(new_points)
    return new_points


def get_next_distance_point(input_points, mask):
    max_distance_point = [0, 0]
    max_distance = 0
    input_points = np.array(input_points)

    indices = np.argwhere(mask == True)
    for x, y in indices:
        # print(x,y,input_points)
        distance = np.sum(np.sqrt((x - input_points[:, 0]) ** 2 + (y - input_points[:, 1]) ** 2))
        if max_distance < distance:
            max_distance_point = [x, y]
            max_distance = distance
    return max_distance_point


def swap_xy(points):
    new_points = np.zeros((len(points),2))
    new_points[:,0] = points[:,1]
    new_points[:,1] = points[:,0]
    return new_points


def k_means_sampling(mask, k):
    points = np.argwhere(mask == 1) # [y, x]
    points = np.flip(points, axis=1)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    points = kmeans.cluster_centers_
    return points


def get_point_prompt_max_dist(masks, num_points):
    n_points = []
    for mask in masks:
        mask_np = mask.cpu().numpy()

        indices = np.argwhere(mask_np > 0)
        random_index = np.random.choice(len(indices), 1)[0]

        first_point = [indices[random_index][1], indices[random_index][0]]
        new_points = get_multi_distance_points(first_point, mask_np, num_points - 1)
        n_points.append(new_points)

    return n_points


def get_point_prompt_kmeans(masks, num_points):
    n_points = []
    for mask in masks:
        mask_np = mask.cpu().numpy()
        points = k_means_sampling(mask_np, num_points)
        n_points.append(points.astype(int))
    return n_points


def get_point_prompts(gt_masks, num_points):
    prompts = []
    for mask in gt_masks:
        po_points = uniform_sampling(mask, num_points)
        na_points = uniform_sampling((~mask.to(bool)).to(float), num_points)
        po_point_coords = torch.tensor(po_points, device=mask.device)
        na_point_coords = torch.tensor(na_points, device=mask.device)
        point_coords = torch.cat((po_point_coords, na_point_coords), dim=1)
        po_point_labels = torch.ones(po_point_coords.shape[:2], dtype=torch.int, device=po_point_coords.device)
        na_point_labels = torch.zeros(na_point_coords.shape[:2], dtype=torch.int, device=na_point_coords.device)
        point_labels = torch.cat((po_point_labels, na_point_labels), dim=1)
        in_points = (point_coords, point_labels)
        prompts.append(in_points)
    return prompts



class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts


def validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "iters": iters}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg


def unspervised_validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    init_prompt = cfg.prompt
    cfg.prompt = "box"
    iou_box, f1_box = validate(fabric, cfg, model, val_dataloader, name, iters)
    cfg.prompt = "point"
    iou_point, f1_point = validate(fabric, cfg, model, val_dataloader, name, iters)
    # cfg.prompt = "coarse"
    # validate(fabric, cfg, model, val_dataloader, name, iters)
    cfg.prompt = init_prompt
    return iou_box, f1_box, iou_point, f1_point


def contrast_validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0, loss: float = 0.):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, _, _ = model(images, prompts)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    csv_dict = {"Name": name, "Prompt": cfg.prompt, "Mean IoU": f"{ious.avg:.4f}", "Mean F1": f"{f1_scores.avg:.4f}", "iters": iters, "loss": loss}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"metrics-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg
