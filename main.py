import os
import yaml
import time
import torch
import lightning as L
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from configs.config import cfg
from loss import DiceLoss, FocalLoss, ContraLoss
from datasets import call_load_dataset

from model import Model
from lora import LoRA_Sam
from utils.utils import AverageMeter, calc_iou, validate, get_prompts
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances


def train_sam(
    cfg: Box,                           # Configuration parameters
    fabric: L.Fabric,                   # Lightning Fabric instance for distributed training
    model: Model,                       # Main model
    anchor_model: Model,                # Anchor model (for semi-supervised learning)
    optimizer: _FabricOptimizer,        # Optimizer
    scheduler: _FabricOptimizer,        # Learning rate scheduler
    train_dataloader: DataLoader,       # Training data loader
    val_dataloader: DataLoader,         # Validation data loader
    num_iters: int,                     # Total number of iterations
):
    """The SAM training loop with semi-supervised learning and contrastive loss."""
    # Initialize metric trackers
    batch_time = AverageMeter()         # Track batch processing time
    data_time = AverageMeter()          # Track data loading time
    focal_losses = AverageMeter()       # Track focal loss
    dice_losses = AverageMeter()        # Track dice loss
    iou_losses = AverageMeter()         # Track IoU loss
    anchor_losses = AverageMeter()      # Track anchor loss
    contra_losses = AverageMeter()      # Track contrastive loss
    total_losses = AverageMeter()       # Track total loss
    
    # Initialize loss functions
    focal_loss = FocalLoss()            # For focal loss calculation
    dice_loss = DiceLoss()             # For dice loss calculation
    contra_loss = ContraLoss()          # For contrastive loss calculation
    
    end = time.time()
    max_iou = 0.                        # Track best IoU for model saving
    num_epochs = cfg.num_iters // num_iters + 1

    # Training loop
    for epoch in range(1, num_epochs):
        for iter, data in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            
            # Unpack the data
            images_weak, images_strong, bboxes, gt_masks = data
            batch_size = images_weak.size(0)
            
            # Count instances and reduce if exceeds maximum limit
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                print(num_insts)
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)

            # Get prompts (box or point)
            prompts = get_prompts(cfg, bboxes, gt_masks)

            # Generate pseudo-labels using anchor model (no gradient computation)
            with torch.no_grad():
                anchor_image_embeds, anchor_masks, anchor_iou_predictions, anchor_res_masks = anchor_model(images_weak, prompts)

            # Process weak and strong augmented images with main model
            # Teacher branch
            soft_image_embeds, soft_masks, soft_iou_predictions, soft_res_masks = model(images_weak, prompts)    
            # Student branch
            pred_image_embeds, pred_masks, iou_predictions, pred_res_masks = model(images_strong, prompts)   

            # Initialize loss components
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            loss_anchor = torch.tensor(0., device=fabric.device)
            loss_contra = torch.tensor(0., device=fabric.device)

            # Calculate losses for each instance
            for i, (pred_mask, soft_mask, anchor_mask, iou_prediction) in enumerate(zip(
                    pred_masks, soft_masks, anchor_masks, iou_predictions)):
                # Convert anchor mask to binary
                anchor_mask = (anchor_mask > 0.).float()
                
                # Calculate contrastive loss between teacher and anchor embeddings
                loss_contra += contra_loss(soft_image_embeds[i], anchor_image_embeds[i], 
                                         soft_res_masks[i].clone().detach(), 
                                         anchor_res_masks[i].clone().detach())
                
                # Calculate consistency loss with anchor model
                loss_anchor += (0.5 * dice_loss(pred_mask, anchor_mask) + 
                              0.5 * dice_loss(soft_mask, anchor_mask))

                # Convert soft mask to binary
                soft_mask = (soft_mask > 0.).float()
                
                # Calculate segmentation losses
                loss_focal += focal_loss(pred_mask, soft_mask, num_masks)
                loss_dice += dice_loss(pred_mask, soft_mask, num_masks)
                
                # Calculate IoU prediction loss
                batch_iou = calc_iou(pred_mask, soft_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            # Compute total loss and backpropagate
            loss_total = 20. * loss_focal + loss_dice + loss_iou + loss_anchor + loss_contra
            fabric.backward(loss_total)

            # Optimization step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # Update timing metrics
            batch_time.update(time.time() - end)
            end = time.time()

            # Update loss metrics
            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            anchor_losses.update(loss_anchor.item(), batch_size)
            contra_losses.update(loss_contra.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            # Print training information
            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                        f' | Dataset: [{cfg.dataset} - {cfg.prompt}]'
                        f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                        f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                        f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                        f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                        f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                        f' | Anchor Loss [{anchor_losses.val:.4f} ({anchor_losses.avg:.4f})]'
                        f' | Contrast Loss [{contra_losses.val:.4f} ({contra_losses.avg:.4f})]'
                        f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            # Log losses
            loss_logger = {
                "Focal Loss": focal_losses.avg, 
                "Dice Loss": dice_losses.avg,
                "IoU Loss": iou_losses.avg, 
                "Anchor Loss": anchor_losses.avg,
                "Contrast Loss": contra_losses.avg, 
                "Total Loss": total_losses.avg
            }
            fabric.log_dict(loss_logger)
            torch.cuda.empty_cache()

        # Periodic validation and model saving
        if epoch % cfg.eval_interval == 0:
            # Validate the model
            iou, f1_score = validate(fabric, cfg, model, val_dataloader, cfg.name, epoch * num_iters)
            # Save model if current IoU is better
            if iou > max_iou:
                state = {"model": model, "optimizer": optimizer}
                fabric.save(os.path.join(cfg.out_dir, "save", 
                          f"{cfg.dataset}-{cfg.prompt}-last-ckpt.pth"), state)
                max_iou = iou


def configure_opt(cfg: Box, model: Model):
    """Configure optimizer and learning rate scheduler with warmup and decay"""
    
    def lr_lambda(step):
        # Warmup phase
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        # Full learning rate phase
        elif step < cfg.opt.steps[0]:
            return 1.0
        # First decay phase
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        # Second decay phase
        else:
            return 1 / (cfg.opt.decay_factor**2)

    # Initialize Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.opt.learning_rate,
        weight_decay=cfg.opt.weight_decay
    )
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def multi_main(cfg):
    prompts = ["box", "point"]
    for prompt in prompts:
        cfg.prompt = prompt
        torch.cuda.empty_cache()
        main(cfg)



def main(cfg: Box, ckpt: str = None) -> None:
    """
    Main training function that sets up environment and initiates training.
    
    Args:
        cfg: Configuration object containing training parameters
        ckpt: Optional checkpoint path to resume training
    """
    # Setup GPU environment
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    # Initialize Lightning Fabric for distributed training
    fabric = L.Fabric(
        accelerator="auto",
        devices=num_devices,
        strategy="auto",
        loggers=[TensorBoardLogger(cfg.out_dir, name=f"{cfg.dataset}-{cfg.prompt}")]
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    # Save configuration file
    if fabric.global_rank == 0:
        cfg_dict = cfg.to_dict()
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        cfg_dict_path = os.path.join(cfg.out_dir, "configs", f"{cfg.dataset}-{cfg.prompt}.yaml")
        with open(cfg_dict_path, "w") as file:
            yaml.dump(cfg_dict, file)
        
        # Create directory for model checkpoints
        os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), 
                  csv_head=cfg.csv_keys)

    # Initialize model and apply LoRA
    with fabric.device:
        model = Model(cfg)
        model.setup()
        anchor_model = copy_model(model)
        LoRA_Sam(model.model, 4)  # Apply LoRA with rank 4

    # Load datasets
    load_datasets = call_load_dataset(cfg)
    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model.model)

    # Print dataset sizes
    fabric.print(f"Train Data: {len(train_data) * cfg.batch_size}; "
                f"Val Data: {len(val_data) * cfg.val_batchsize}")
    num_iters = len(train_data) * cfg.batch_size

    # Load checkpoint if provided
    if ckpt is not None:
        full_checkpoint = fabric.load(ckpt)
        model.load_state_dict(full_checkpoint["model"])

    # Setup data loaders and model for distributed training
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    model, optimizer = fabric.setup(model, optimizer)

    # Initial validation and start training
    validate(fabric, cfg, anchor_model, val_data, name=cfg.name, iters=0)
    train_sam(cfg, fabric, model, anchor_model, optimizer, scheduler, 
             train_data, val_data, num_iters)

    # Cleanup
    del model, anchor_model, train_data, val_data


if __name__ == "__main__":
    # Initialize training environment
    torch.cuda.empty_cache()                                    # Clear GPU memory
    torch.set_float32_matmul_precision('high')                 # Set precision for matrix operations
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids           # Set visible GPU devices

    main(cfg)                                                  # Run main training loop
    torch.cuda.empty_cache()                                   # Final memory cleanup


"""
This code implements:
1. A semi-supervised learning approach for training SAM
2. Teacher-student architecture
3. Multiple loss functions (Focal, Dice, IoU, Anchor, Contrastive)
4. Distributed training support
5. Complete training, validation, and model saving pipeline
6. Support for different prompt types (box and point)
7. LoRA for model fine-tuning
"""