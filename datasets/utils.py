import os
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
from torch_geometric.data import Batch
import random
import os.path as osp
from collections import defaultdict
import json
import torch.nn as nn
from torch_geometric.data import HeteroData
import math

# Assuming this function is in a file named graph_builder.py
from graph_builder import build_fully_connected_hetero_graph



# --- 1. CLIP Wrapper to force Token Output ---
class TokenCLIPVisual(nn.Module):
    def __init__(self, original_visual):
        super().__init__()
        self.original = original_visual
        # Expose attributes for CLIP parent class
        self.conv1 = original_visual.conv1 
        self.class_embedding = original_visual.class_embedding
        self.positional_embedding = original_visual.positional_embedding
        self.ln_pre = original_visual.ln_pre
        self.transformer = original_visual.transformer
        self.ln_post = original_visual.ln_post
        self.proj = original_visual.proj
        self.input_resolution = original_visual.input_resolution
        self.output_dim = original_visual.output_dim
        
    def forward(self, x):
        model = self.original
        
        # 1. Convolution to get patches
        # x shape: [Batch, 3, H, W]
        x = model.conv1(x)  # shape = [Batch, Width, Grid_H, Grid_W]
        B, C, H, W = x.shape
        
        x = x.reshape(B, C, -1)  # shape = [Batch, Width, Grid_H * Grid_W]
        x = x.permute(0, 2, 1)   # shape = [Batch, Grid_Size, Width]

        # 2. Add CLS token
        # Shape: [Batch, 1, Width]
        cls_token = model.class_embedding.to(x.dtype) + torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1) # [Batch, 1 + Grid_Size, Width]

        # 3. Dynamic Positional Embedding Interpolation
        # Standard CLIP pos_embed is [1 + 14*14, Width] (for 224x224)
        pos_embed = model.positional_embedding.to(x.dtype)
        
        # If our grid size matches the pre-trained size (14x14), use as is
        if pos_embed.shape[0] == x.shape[1]:
            x = x + pos_embed
        else:
            # We need to interpolate pos_embed to match x's size
            # Separate CLS pos_embed from Patch pos_embeds
            cls_pos_embed = pos_embed[0:1] # [1, Width]
            patch_pos_embed = pos_embed[1:] # [196, Width]
            
            # Determine original grid size (usually 14x14)
            orig_size = int(math.sqrt(patch_pos_embed.shape[0]))
            
            # Reshape to [1, Width, Orig_H, Orig_W] for interpolation
            patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, C).permute(0, 3, 1, 2)
            
            # Interpolate to new grid size (H, W comes from conv1 output)
            patch_pos_embed = F.interpolate(patch_pos_embed, size=(H, W), mode='bicubic', align_corners=False)
            
            # Reshape back to [1, New_Grid_Size, Width]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, C)
            
            # Recombine
            new_pos_embed = torch.cat([cls_pos_embed.unsqueeze(0), patch_pos_embed], dim=1)
            x = x + new_pos_embed

        x = model.ln_pre(x)

        # 4. Transformer Layers
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 5. Final Norm & Projection
        x = model.ln_post(x)
        if model.proj is not None:
            x = x @ model.proj

        return x
    
# --- 2. Pre-compute Grid Topology ---
def create_grid_edge_index(grid_size=14):
    """
    Creates edges for a grid where every node is connected to its 8 neighbors.
    Node 0 is CLS. Nodes 1..196 are the 14x14 grid.
    """
    src_nodes = []
    dst_nodes = []
    
    # Offsets for 8 neighbors (King's moves)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    for y in range(grid_size):
        for x in range(grid_size):
            # Map grid (y,x) to node index (1 to 196)
            current_node = 1 + y * grid_size + x
            
            # A. Connect to Spatial Neighbors
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_size and 0 <= nx < grid_size:
                    neighbor_node = 1 + ny * grid_size + nx
                    src_nodes.append(current_node)
                    dst_nodes.append(neighbor_node)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index




    

# --- I/O and Helper Functions ---

def read_json(fpath: str) -> dict:
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj: dict, fpath: str):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def read_image(path: str) -> Image.Image:
    """Read image from path using PIL.Image, retrying on IOError."""
    if not os.path.exists(path):
        raise IOError(f'No file exists at {path}')
    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(f'Cannot read image from {path}, retrying...')

def listdir_nohidden(path: str, sort: bool = False) -> list[str]:
    """List non-hidden items in a directory."""
    items = [f for f in os.listdir(path) if not f.startswith('.')]
    if sort:
        items.sort()
    return items

# --- Data Structures ---

class Datum:
    """
    Data instance which defines the basic attributes for an image, including
    classname and domain for compatibility with DatasetBase.
    """
    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

class DatasetBase:
    """
    A unified dataset class for managing train/val/test splits, class names,
    and other dataset-level information.
    """
    dataset_dir = ''
    domains = []

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x
        self._train_u = train_u
        self._val = val
        self._test = test

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames
    
    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=True):
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')
        output = []
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)
            output.append(dataset)
        if len(output) == 1:
            return output[0]
        return output

    def split_dataset_by_label(self, data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)
        return output

# --- Image Processing ---

# def fixed_grid_crops(img: Image.Image, crop_size: int = 224, nodes: int = 17) -> list[Image.Image]:
#     """
#     Generate a deterministic set of crops from a 336x336 image.
#     The default behavior produces 18 crops in total.
#     """
#     resize = T.Resize(size=(336, 336), interpolation=T.InterpolationMode.BICUBIC)
#     resize224 = T.Resize(size=(224, 224), interpolation=T.InterpolationMode.BICUBIC)
#     img = resize(img)
#     w, h = img.size
#     assert w == 336 and h == 336, f"Expected image of size 336x336, but got {w}x{h}"
    
#     crops = []
    
#     crops.append(resize224(img))
    
#     corners = [(0, 0), (w - crop_size, 0), (0, h - crop_size), (w - crop_size, h - crop_size)]
#     for x, y in corners:
#         crops.append(TF.crop(img, top=y, left=x, height=crop_size, width=crop_size))

#     crops.append(T.CenterCrop(crop_size)(img))
    
#     stride = crop_size // 2
#     shifts = [(stride, 0), (0, stride), (2 * stride, stride), (stride, 2 * stride)]
#     for x, y in shifts:
#         shifted_crop = TF.crop(img, top=y, left=x, height=stride, width=stride)
#         crops.append(resize224(shifted_crop))

#     quadrant_size = 168
#     quadrants = [(0, 0), (w - quadrant_size, 0), (0, h - quadrant_size), (w - quadrant_size, h - quadrant_size)]
#     for x, y in quadrants:
#         q_crop = TF.crop(img, top=y, left=x, height=quadrant_size, width=quadrant_size)
#         crops.append(resize224(q_crop))

#     horizontal_halves = [(0, 0), (0, h // 2)]
#     for x, y in horizontal_halves:
#         h_half_crop = TF.crop(img, top=y, left=x, height=168, width=336)
#         crops.append(resize224(h_half_crop))

#     vertical_halves = [(0, 0), (w // 2, 0)]
#     for x, y in vertical_halves:
#         v_half_crop = TF.crop(img, top=y, left=x, height=336, width=168)
#         crops.append(resize224(v_half_crop))
        
#     return crops


def fixed_grid_crops(img: Image.Image, crop_size: int = 224, nodes=17) -> list[Image.Image]:
    """
    Generates a multi-scale grid of patches from a single image.

    The process creates:
    - 1 full-image view
    - 9 patches from a 3x3 grid
    - 4 patches from a 2x2 grid
    - 2 patches from a vertical split
    - 2 patches from a horizontal split
    
    Total patches = 1 + 9 + 4 + 2 + 2 = 18.
    """
    crops = []
    
    # Standard resizer for crops that aren't already the target size
    resize_to_crop = T.Resize(size=(crop_size, crop_size), interpolation=T.InterpolationMode.BICUBIC)

    # --- 1. The original full image, resized ---
    # This is the global view.
    crops.append(resize_to_crop(img))

    # --- 2. Generate 3x3 grid (9 patches) ---
    # Resize the image to a size perfectly divisible by 3 (e.g., 672x672)
    grid_3x3_size = crop_size * 3
    img_3x3 = T.Resize(size=(grid_3x3_size, grid_3x3_size), interpolation=T.InterpolationMode.BICUBIC)(img)
    patch_size_3x3 = grid_3x3_size // 3  # This will be `crop_size`

    for i in range(3):
        for j in range(3):
            top = i * patch_size_3x3
            left = j * patch_size_3x3
            patch = TF.crop(img_3x3, top, left, patch_size_3x3, patch_size_3x3)
            crops.append(patch)

    # --- 3. Generate 2x2 grid (4 patches) ---
    # Resize the image to a size perfectly divisible by 2 (e.g., 448x448)
    grid_2x2_size = crop_size * 2
    img_2x2 = T.Resize(size=(grid_2x2_size, grid_2x2_size), interpolation=T.InterpolationMode.BICUBIC)(img)
    patch_size_2x2 = grid_2x2_size // 2 # This will be `crop_size`

    for i in range(2):
        for j in range(2):
            top = i * patch_size_2x2
            left = j * patch_size_2x2
            patch = TF.crop(img_2x2, top, left, patch_size_2x2, patch_size_2x2)
            crops.append(patch)
            
    # --- 4. Generate vertical halves (2 patches) ---
    # Use the 2x2 resized image (448x448) for this split
    w, h = img_2x2.size
    
    # Crop left half (448x224) and resize to 224x224
    left_half = TF.crop(img_2x2, top=0, left=0, height=h, width=w // 2)
    crops.append(resize_to_crop(left_half))
    
    # Crop right half (448x224) and resize to 224x224
    right_half = TF.crop(img_2x2, top=0, left=w // 2, height=h, width=w // 2)
    crops.append(resize_to_crop(right_half))

    # --- 5. Generate horizontal halves (2 patches) ---
    # Use the same 2x2 resized image (448x448)
    
    # Crop top half (224x448) and resize to 224x224
    top_half = TF.crop(img_2x2, top=0, left=0, height=h // 2, width=w)
    crops.append(resize_to_crop(top_half))
    
    # Crop bottom half (224x448) and resize to 224x224
    bottom_half = TF.crop(img_2x2, top=h // 2, left=0, height=h // 2, width=w)
    crops.append(resize_to_crop(bottom_half))

    # Sanity check to ensure we have the correct number of patches
    assert len(crops) == 18, f"Expected 18 patches, but generated {len(crops)}"
    
    return crops

# --- Standard DataLoader (for val/test) ---

class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, transform=None, is_train=False):
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = read_image(item.impath)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, item.label

def build_data_loader(data_source, batch_size, is_train=False, tfm=None, shuffle=False,
                      num_workers=4, persistent_workers=False):
    data_loader = torch.utils.data.DataLoader(
        DatasetWrapper(data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(persistent_workers and num_workers > 0)
    )
    return data_loader

# --- Graph-based DataLoader (for training) ---

# --- 3. Updated Dataset (No more crops) ---
class ImagePatchDataset(TorchDataset):
    def __init__(self, data_source, transform=None, patch_processor=None):
        self.data_source = data_source
        self.transform = transform
        # patch_processor is kept for compatibility but not used for cropping anymore

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        
        # Read and transform the full image
        try:
            image = Image.open(item.impath).convert('RGB')
        except IOError:
            print(f"Error reading {item.impath}")
            # Return a dummy black image in case of error to prevent crash
            image = Image.new('RGB', (224, 224))
            
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        return {
            'image': image_tensor,
            'label': torch.tensor(item.label, dtype=torch.long),
            'image_id': item.impath
        }


# --- 4. Updated Collate Function ---
class GraphCollate:
    def __init__(self, vit_feature_extractor, text_features, device):
        self.vit_model = vit_feature_extractor.to(device).eval()
        self.text_features = text_features.to(device)
        self.device = device
        self.edge_cache = {} # Cache edges for different resolutions

    def get_edges(self, grid_size):
        # Cache the edge index so we don't rebuild it if resolution stays same
        if grid_size not in self.edge_cache:
            self.edge_cache[grid_size] = create_grid_edge_index(grid_size).to(self.device)
        return self.edge_cache[grid_size]

    def __call__(self, batch: list[dict]):
        images = torch.stack([item['image'] for item in batch]).to(self.device)
        labels = torch.stack([item['label'] for item in batch]).to(self.device)
        
        # 1. Run ViT (Now supports dynamic size)
        with torch.no_grad():
            batch_vit_tokens = self.vit_model(images) 

        graph_list = []
        B, N, D = batch_vit_tokens.shape 
        
        # Calculate grid size from number of patches (N-1)
        # e.g., if N=197 -> 196 patches -> sqrt(196)=14. 
        # e.g., if N=442 -> 441 patches -> sqrt(441)=21.
        grid_size = int(math.sqrt(N - 1))
        
        visual_edge_index = self.get_edges(grid_size)
        
        num_text_nodes = self.text_features.shape[0]
        cls_idx = torch.zeros(num_text_nodes, dtype=torch.long, device=self.device)
        text_idx = torch.arange(num_text_nodes, dtype=torch.long, device=self.device)
        v2t_edge_index = torch.stack([cls_idx, text_idx], dim=0)
        t2v_edge_index = torch.stack([text_idx, cls_idx], dim=0)

        for i in range(B):
            vit_f = batch_vit_tokens[i]
            data = HeteroData()
            data['vit'].x = vit_f
            data['text'].x = self.text_features
            data['vit', 'intra_patch', 'vit'].edge_index = visual_edge_index
            data['vit', 'visual_to_text', 'text'].edge_index = v2t_edge_index
            data['text', 'text_to_visual', 'vit'].edge_index = t2v_edge_index
            graph_list.append(data)
        
        batched_graphs = Batch.from_data_list(graph_list)
        return batched_graphs, images, labels
    
    
    
    
def build_graph_data_loader(
    data_source, batch_size, shuffle, transform, 
    vit_model, text_features, processor, device, num_workers=0
):
    dataset = ImagePatchDataset(data_source, transform=transform, patch_processor=processor)
    
    collate_fn = GraphCollate(
        vit_feature_extractor=vit_model,
        text_features=text_features,
        device=device
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    return data_loader



# @torch.no_grad()
# def select_shots_herding(data_source, model, transform, num_shots, device='cuda', batch_size=64):
#     """
#     Selects the N best samples per class based on distance to the class mean (Herding).
#     """
#     print(f"\n[Herding] Selecting best {num_shots} shots from {len(data_source)} total samples...")
#     model.eval()
#     model.to(device)
    
#     # 1. Create a temporary loader for the FULL dataset using Validation Transforms
#     # We use validation transforms because we want the "true" canonical view of the object, 
#     # not a randomly cropped/flipped version.
#     temp_loader = torch.utils.data.DataLoader(
#         DatasetWrapper(data_source, transform=transform, is_train=False),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         drop_last=False
#     )
    
#     all_features = []
#     all_labels = []
    
#     # 2. Extract features for every image in the dataset
#     print("[Herding] Extracting features...")
#     for images, labels in temp_loader:
#         images = images.to(device)
        
#         # Pass through TokenCLIPVisual wrapper
#         # Output shape: [Batch, Tokens, Dim]
#         with torch.cuda.amp.autocast():
#             output = model(images)
            
#         # We take the CLS token (index 0) as the global representation
#         global_features = output[:, 0, :]
        
#         # Normalize features (Critical for Cosine Similarity / Euclidean distance in hypersphere)
#         global_features = global_features / global_features.norm(dim=-1, keepdim=True)
        
#         all_features.append(global_features.cpu())
#         all_labels.append(labels)
        
#     all_features = torch.cat(all_features, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
    
#     # 3. Select best shots per class
#     selected_indices = []
#     unique_classes = torch.unique(all_labels).sort().values
    
#     print("[Herding] Calculating class means and selecting closest samples...")
#     for cls_idx in unique_classes:
#         # Get indices of all samples belonging to this class
#         # (This corresponds to the index in 'data_source')
#         cls_mask = (all_labels == cls_idx)
#         global_indices = torch.nonzero(cls_mask).squeeze(1)
        
#         # Get their features
#         cls_feats = all_features[cls_mask]
        
#         # Calculate Class Prototype (Mean)
#         prototype = cls_feats.mean(dim=0, keepdim=True)
#         prototype = prototype / prototype.norm(dim=-1, keepdim=True)
        
#         # Calculate Cosine Similarity to the prototype
#         # Higher similarity = Closer to mean
#         similarities = torch.mm(cls_feats, prototype.t()).squeeze()
        
#         # Handle edge case where class has fewer samples than num_shots
#         k = min(num_shots, len(global_indices))
        
#         # Get indices of the top K highest similarity scores
#         _, best_rel_indices = torch.topk(similarities, k)
        
#         # Map back to the original dataset indices
#         best_global_indices = global_indices[best_rel_indices]
#         selected_indices.extend(best_global_indices.tolist())
    
#     # 4. Filter the data source
#     herded_data_source = [data_source[i] for i in selected_indices]
    
#     print(f"[Herding] Selection complete. Reduced from {len(data_source)} to {len(herded_data_source)} samples.")
#     return herded_data_source



@torch.no_grad()
def select_shots_herding(data_source, model, transform, num_shots, device='cuda', batch_size=64,
                         num_workers=4, pin_memory=False):
    """
    Selects the N best samples per class.
    - If num_shots == 1: Selects the sample closest to the global class mean.
    - If num_shots > 1: Uses Spherical K-Means to select N diverse prototypes.
    """
    print(f"\n[Herding] Selecting best {num_shots} shots from {len(data_source)} total samples...")
    model.eval()
    model.to(device)
    
    # 1. Create loader
    temp_loader = torch.utils.data.DataLoader(
        DatasetWrapper(data_source, transform=transform, is_train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )
    
    all_features = []
    all_labels = []
    
    # 2. Extract features
    print("[Herding] Extracting features...")
    for images, labels in temp_loader:
        images = images.to(device)
        with torch.cuda.amp.autocast():
            output = model(images)
            
        # CLS token + Normalize
        global_features = output[:, 0, :]
        global_features = F.normalize(global_features, dim=-1)
        
        all_features.append(global_features.cpu())
        all_labels.append(labels)
        
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    selected_indices = []
    unique_classes = torch.unique(all_labels).sort().values
    
    print(f"[Herding] Running selection (Mode: {'K-Means' if num_shots > 2 else 'Mean-Centroid'})...")
    
    for cls_idx in unique_classes:
        # Get indices/features for this class
        cls_mask = (all_labels == cls_idx)
        global_indices = torch.nonzero(cls_mask).squeeze(1)
        cls_feats = all_features[cls_mask].to(device) # Move to GPU for K-Means
        
        num_samples = len(global_indices)
        k = min(num_shots, num_samples)
        
        if k == num_samples:
            # Take all if not enough samples
            selected_indices.extend(global_indices.tolist())
            continue

        # --- STRATEGY SWITCH ---
        
        if k <= 2:
            # Strategy: Pick k samples closest to the Global Class Mean
            prototype = cls_feats.mean(dim=0, keepdim=True)
            prototype = F.normalize(prototype, dim=-1)
            
            # Cosine similarity
            sims = torch.mm(cls_feats, prototype.t()).squeeze()
            
            # FIX: Use topk instead of argmax to handle k=2
            _, best_rel_indices = torch.topk(sims, k)
            
            # Add all selected indices
            for rel_idx in best_rel_indices:
                selected_indices.append(global_indices[rel_idx].item())
            
        else:
            # >1 Shot: Spherical K-Means for Diversity
            # We want k clusters.
            
            # 1. Initialize Centroids (Randomly pick k samples)
            # We use a fixed seed per class for reproducibility within the run
            g_cpu = torch.Generator()
            g_cpu.manual_seed(int(cls_idx)) 
            init_perm = torch.randperm(num_samples, generator=g_cpu)[:k]
            centroids = cls_feats[init_perm].clone()
            
            # 2. Run K-Means (Simple PyTorch implementation)
            # Max iterations 20 is usually plenty for feature convergence
            for _ in range(20):
                # Assign points to closest centroid (Cosine Similarity)
                # sims: [N_samples, K_clusters]
                sims = torch.mm(cls_feats, centroids.t()) 
                labels = torch.argmax(sims, dim=1)
                
                # Update centroids
                new_centroids = []
                for i in range(k):
                    mask = (labels == i)
                    if mask.sum() > 0:
                        cluster_mean = cls_feats[mask].mean(dim=0)
                        new_centroids.append(cluster_mean)
                    else:
                        # Handle empty cluster (rare): keep old centroid
                        new_centroids.append(centroids[i])
                
                centroids = torch.stack(new_centroids)
                centroids = F.normalize(centroids, dim=-1) # Spherical constraint
            
            # 3. Select Representative for each Cluster
            # For each centroid, find the actual sample closest to it
            sims = torch.mm(cls_feats, centroids.t()) # [N, K]
            
            for i in range(k):
                # We only consider points assigned to this cluster to ensure partition
                # (Optional: can also just pick globally closest, but this is safer)
                mask = (labels == i)
                
                if mask.sum() > 0:
                    # Find max sim within this cluster
                    cluster_sims = sims[:, i].clone()
                    cluster_sims[~mask] = -1.0 # Ignore other points
                    best_rel_idx = torch.argmax(cluster_sims).item()
                else:
                    # Fallback if cluster empty: just pick globally closest
                    best_rel_idx = torch.argmax(sims[:, i]).item()
                
                selected_indices.append(global_indices[best_rel_idx].item())

    # Filter and Return
    # Sort indices to maintain some order (optional)
    selected_indices.sort()
    herded_data_source = [data_source[i] for i in selected_indices]
    
    print(f"[Herding] Selection complete. Reduced from {len(data_source)} to {len(herded_data_source)} samples.")
    return herded_data_source
