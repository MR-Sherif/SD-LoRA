import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import clip
from peft import PeftModel, PeftConfig

# Import TokenCLIPVisual to handle 336x336 positional embedding interpolation
from datasets.utils import Datum, DatasetBase, TokenCLIPVisual
from utils import clip_classifier, cls_acc
from datasets.imagenet import ImageNet

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/imagenet.yaml')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Target OOD dataset: imagenet_a, imagenet_r, imagenet_sketch, imagenet_v2')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to the directory containing OOD datasets')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Parent directory of the ImageNet dataset')
    
    # Model Paths
    parser.add_argument('--cache_dir', type=str, default="./caches/b16/imagenet",
                        help='Directory containing keys/values and best_F model')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to the saved LoRA adapter folder')
    
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=str, default='0')
    
    # Eval Params
    parser.add_argument('--alpha', type=float, default=1.7)
    parser.add_argument('--beta', type=float, default=1.0)

    return parser.parse_args()

def load_imagenet_mapping():
    if not os.path.exists('imagenet_class_index.json'):
        print("Warning: imagenet_class_index.json not found in current directory.")
        print("Required for mapping OOD classes (ImageNet-A/R) to ImageNet-1k indices.")
        return {}
        
    with open('imagenet_class_index.json', 'r') as f:
        class_index = json.load(f)
    # Mapping: "n01440764": 0
    wnid_to_idx = {wnid: int(idx) for idx, (wnid, _) in class_index.items()}
    return wnid_to_idx

def get_ood_loader(dataset_name, root, preprocess, ref_class_to_idx, batch_size):
    name_map = { 'imagenet_a': 'imagenet-a', 'imagenet_r': 'imagenet-r', 
                 'imagenet_sketch': 'imagenet-sketch', 'imagenet_v2': 'imagenet-v2' }
    folder_name = name_map.get(dataset_name, dataset_name)
    data_dir = os.path.join(root, folder_name)

    # Handle ImageNet-V2 specifics
    if dataset_name == 'imagenet_v2':
        if os.path.exists(os.path.join(data_dir, 'imagenetv2-matched-frequency-format-val')):
            data_dir = os.path.join(data_dir, 'imagenetv2-matched-frequency-format-val')
        elif os.path.exists(os.path.join(data_dir, 'val')):
            data_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Loading raw data from: {data_dir}")
    raw_dataset = datasets.ImageFolder(data_dir, transform=preprocess)
    
    # --- Mask Generation Logic ---
    ood_classes = raw_dataset.classes
    local_to_global = {}
    valid_indices = [] 
    
    print("Mapping OOD classes to ImageNet-1k indices...")
    is_integer_folders = ood_classes[0].isdigit()

    for local_idx, folder_name in enumerate(ood_classes):
        global_idx = -1
        if is_integer_folders:
            # For ImageNet-V2/Sketch where folders might be '0', '1'...
            try: global_idx = int(folder_name)
            except: pass
        else:
            # For ImageNet-A/R where folders are WNIDs 'n01234567'
            if folder_name in ref_class_to_idx:
                global_idx = ref_class_to_idx[folder_name]

        if global_idx != -1 and 0 <= global_idx < 1000:
            local_to_global[local_idx] = global_idx
            valid_indices.append(global_idx)

    # Convert to list of tuples (image, global_label)
    test_data = []
    for path, local_label in raw_dataset.samples:
        if local_label in local_to_global:
            real_label = local_to_global[local_label]
            test_data.append((path, real_label))
            
    print(f"Mapped {len(test_data)} images. Classes found: {len(valid_indices)}")
    
    # Create Mask
    mask = torch.zeros(1000, dtype=torch.bool)
    if len(valid_indices) > 0:
        mask[valid_indices] = True
    else:
        print("Warning: No class mapping found. Assuming identity mapping.")
        mask[:] = True
        
    class OODDataset(torch.utils.data.Dataset):
        def __init__(self, data_list, transform):
            self.data_list = data_list
            self.transform = transform
            self.loader = datasets.folder.default_loader

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, index):
            path, target = self.data_list[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target

    dataset = OODDataset(test_data, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return loader, mask

def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, 'r', encoding='utf-8') as config_file:
        cfg = yaml.safe_load(config_file)
    cfg['root_path'] = args.root_path
    if args.lora_path is None:
        args.lora_path = os.path.join(args.cache_dir, f"best_lora_{args.shots}shots")
    
    print(f"\nModel: ImageNet ({args.shots} shots) | Target: {args.dataset}")
    print(f"LoRA Path: {args.lora_path}")
    print(f"Cache Dir: {args.cache_dir}")
    print("Resolution: 336x336 (using TokenCLIPVisual interpolation)")

    # 1. Load CLIP Backbone
    print("\n[1/5] Loading CLIP...")
    clip_model, _ = clip.load(cfg['backbone'][0], device=device, jit=False)
    clip_model = clip_model.float()
    
    # --- Wrap Visual Encoder ---
    clip_model.visual = TokenCLIPVisual(clip_model.visual)
    
    # 2. Generate Classifier Weights (using BASE CLIP)
    print("Generating text features (Classifier)...")
    ref_dataset = ImageNet(root=cfg['root_path'], num_shots=args.shots)
    text_features = clip_classifier(ref_dataset.classnames, ref_dataset.template, clip_model)
    
    # FIXED: Do NOT transpose here. clip_classifier returns [Dim, 1000].
    clip_weights = text_features 

    # 3. Load LoRA Adapter
    print("\n[2/5] Loading LoRA Weights...")
    clip_model = PeftModel.from_pretrained(clip_model, args.lora_path)
    clip_model.eval()
    clip_model.to(device)
    print("LoRA loaded successfully.")

    # 4. Load Cache Keys/Values
    print("\n[3/5] Loading Cache...")
    keys_path = os.path.join(args.cache_dir, f"keys_{args.shots}shots.pt")
    vals_path = os.path.join(args.cache_dir, f"values_{args.shots}shots.pt")
    
    cache_keys = torch.load(keys_path, map_location=device)   # [Dim, Keys]
    cache_values = torch.load(vals_path, map_location=device) # [Keys, C]
    
    # 5. Load Trained Linear Adapter
    print("\n[4/5] Loading Trained Adapter...")
    adapter_path = os.path.join(args.cache_dir, f"best_F_{args.shots}shots.pt")
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(device)
    
    adapter_state = torch.load(adapter_path, map_location=device)
    adapter.load_state_dict(adapter_state)
    adapter.eval()

    # 6. Prepare Data
    print(f"\n[5/5] Preparing OOD Data ({args.dataset})...")
    
    # Define Transform (Resolution 336x336)
    val_transform = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    wnid_map = load_imagenet_mapping()
    loader, logit_mask = get_ood_loader(args.dataset, args.data_root, val_transform, wnid_map, args.batch_size)
    logit_mask = logit_mask.to(device)

    # 7. Evaluation Loop
    print("\nStarting Inference...")
    acc_accum = 0.0
    total_samples = 0
    dtype = torch.float32

    with torch.no_grad():
        for images, target in tqdm(loader):
            images, target = images.to(device), target.to(device)
            B = images.shape[0]

            # A. Encode Image
            features = clip_model.base_model.encode_image(images) 
            
            if features.dim() == 3:
                features = features[:, 0, :]
            
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.to(dtype)

            # B. Zero-shot Logits
            # [Batch, Dim] @ [Dim, Class] -> [Batch, Class]
            clip_logits = 100. * features @ clip_weights.to(dtype)

            # C. Adapter (Cache) Logits
            affinity = adapter(features)
            cache_logits = ((-1) * (args.beta - args.beta * affinity)).exp() @ cache_values.to(dtype)

            # D. Final Logits
            tip_logits = clip_logits + cache_logits * args.alpha

            # E. Masking
            tip_logits[:, ~logit_mask] = -float('inf')

            # F. Accuracy
            acc = cls_acc(tip_logits, target)
            acc_accum += acc / 100 * B
            total_samples += B

    final_acc = (acc_accum / total_samples) * 100
    print(f"\nOptions: Alpha={args.alpha}, Beta={args.beta}")
    print(f"**** {args.dataset} Accuracy: {final_acc:.2f}% ****\n")

if __name__ == '__main__':
    main()
