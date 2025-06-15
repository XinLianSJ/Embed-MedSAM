import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import join, basename
import torch.nn.functional as F
import argparse
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from repvit import RepViT
from repvit_cfgs import repvit_m1_0_cfgs
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# ---- 模型结构定义 ----
class MedSAM_Lite(torch.nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    # def forward(self, image, boxes):
    #     image_embedding = self.image_encoder(image)
    #     sparse_embeddings, dense_embeddings = self.prompt_encoder(
    #         points=None, boxes=boxes, masks=None
    #     )
    #     low_res_masks, iou_predictions = self.mask_decoder(
    #         image_embeddings=image_embedding,
    #         image_pe=self.prompt_encoder.get_dense_pe(),
    #         sparse_prompt_embeddings=sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=False,
    #     )
    #     return low_res_masks, iou_predictions
    def forward(self, image=None, boxes=None, image_embedding=None):
        if image_embedding is None:
            assert image is not None, "Either image or image_embedding must be provided."
            image_embedding = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=boxes, masks=None
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks, iou_predictions, image_embedding
    

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        masks = masks[:, :, :new_size[0], :new_size[1]]
        masks = F.interpolate(masks, size=(original_size[0], original_size[1]), mode="bilinear", align_corners=False)
        return masks

# ---- box 从第一次分割生成 ----
def generate_box_prompt_from_mask(logits):
    boxes = []
    for idx, mask in enumerate(logits):
        bin_mask = torch.sigmoid(mask.squeeze(0)) > 0.5
        y_indices, x_indices = torch.where(bin_mask)

        if len(x_indices) == 0 or len(y_indices) == 0:
            box = torch.tensor([0, 0, 255, 255], device=mask.device)
            # print(f"[{idx}] ⚠️ Empty mask → Using fallback box: {box.tolist()}")
        else:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            box = torch.stack([x_min, y_min, x_max, y_max])
            # print(f"[{idx}] ✅ Box from mask: {box.tolist()}")
        
        boxes.append(box)
    return torch.stack(boxes).unsqueeze(1)

def model_forward_self_prompt(model, image_tensor):
    with torch.no_grad():
        # logits_0, _ = model(image_tensor, boxes=None)
        # box_prompt = generate_box_prompt_from_mask(logits_0)
        # logits_1, iou_pred = model(image_tensor, boxes=box_prompt)
        logits, _, image_embedding = model(image=image_tensor, boxes=None)
        box_prompt = generate_box_prompt_from_mask(logits)
        logits, iou_pred, _ = model(image=None, boxes=box_prompt, image_embedding=image_embedding)
    return box_prompt, logits, iou_pred

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MedSAM_TwoStepWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        logits_0, _, image_embedding = self.model(image=image, boxes=None)
        box_prompt = generate_box_prompt_from_mask(logits_0)
        # print("box_prompt")
        # print(box_prompt)
        logits_1, _, _ = self.model(image=None, boxes=box_prompt, image_embedding=image_embedding)
        return logits_1


def compute_total_self_prompt_flops(model, device):
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    wrapper_model = MedSAM_TwoStepWrapper(model).to(device)
    flops = FlopCountAnalysis(wrapper_model, (dummy_input,))
    print(f"✅ Self-Prompt Inference FLOPs (1x encoder): {flops.total() / 1e9:.2f} GFLOPs")
    return flops.total()



# ---- 主函数入口 ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='输入npy文件路径 (imgs/*.npy)')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='保存预测掩码路径')
    parser.add_argument('-c', '--ckpt', type=str, required=True, help='MedSAM模型权重路径')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- 构建模型 ----
    model = MedSAM_Lite(
        image_encoder=RepViT(cfgs=repvit_m1_0_cfgs, img_size=256),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(256, 256),
            mask_in_chans=16,
        )
    )
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device).eval()

    print("🧠 Model Structure:")
    print(model)
    _ = compute_total_self_prompt_flops(model, device)
    
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    dummy_box = torch.tensor([[[[0, 0, 255, 255]]]], device=device)  # dummy box for model input
    
    with torch.no_grad():
        flops = FlopCountAnalysis(model, (dummy_input, dummy_box))
        # print(f"⚙️ FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
        print(parameter_count_table(model))
    
    # ---- 处理图像 ----
    npy_files = sorted(glob(join(args.input_dir, '*.npy')))
    for npy_path in tqdm(npy_files, desc='Running Inference'):
        img_name = basename(npy_path)
        img_np = np.load(npy_path)  # shape (256, 256, 3), normalized to [0,1]

        assert img_np.shape == (256, 256, 3)
        assert np.max(img_np) <= 1.0 and np.min(img_np) >= 0.0

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1,3,256,256)

        box_prompt, logits, iou_pred = model_forward_self_prompt(model, img_tensor)

        pred_mask = torch.sigmoid(logits).squeeze().cpu().numpy()  # (256,256)
        print("box_prompt")
        print(box_prompt)
        np.save(join(args.output_dir, img_name), pred_mask.astype(np.float32))




if __name__ == "__main__":
    main()
