# Embed-MedSAM: Embedded Framework for Clinical Medical Image Segment Anything in Resource-Limited Healthcare Regions (Under Review)
The hyperparameters in the training code should be adjusted and optimized according to your specific target dataset. To verify the functionality of the code, we will provide a preliminary distilled base model in the ckpts folder after the publication of the paper.

## Getting Started
```bash
pip install -e .
```

## Training and inference commands (Take the REFUGE dataset as an example)
### Training (Prompt-guided Mask Decoder Training)
```bash
python train_self_prompt_v1.py   -data_root /root/autodl-tmp/REFUGE/Training-400-npy   -pretrained_checkpoint ckpts/pretrained_model.pth   -work_dir work_dir/disc_cup_train_self_box   -num_epochs 100   -batch_size 8   -num_workers 8   -device cuda:0
```

### Inference
```bash
python3 inference_self_prompt_low.py -i ../autodl-tmp/REFUGE/Test-400-imgs -o preds_selfprompt -c work_dir/disc_cup_train_self_box/medsam_lite_4.pth
```

## Data Structure for REFUGE dataset
    files in imgs, gts and embeddings share the same file name
    train_npy
        ├─gts
        └─imgs
