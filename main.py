import os
import random
import argparse 
import json
from pathlib import Path
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb

# PEFT Imports for LoRA
from peft import LoraConfig, get_peft_model

from datasets import build_dataset
from datasets.utils import build_data_loader, build_graph_data_loader, TokenCLIPVisual
import clip
from utils import *
from rgtmodel import RGTImageFeatureExtractor

wandb_log = True

def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    # Ensure cache_values is in the correct dtype from the start
    cache_values = cache_values.to(val_features.dtype)

    print("\n-------- Searching hyperparameters on the val set. --------")
    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = wandb.config.init_beta, wandb.config.init_alpha
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter     
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_loader, test_loader, clip_weights, clip_model, image_model, train_loader_F):
    """
    An enhanced version that combines Zero-Shot, Cache, and Graph-based logits.
    Now includes LoRA fine-tuning for the CLIP model AND Visual-Prototype JEPA predictive loss.
    """
    device = next(clip_model.parameters()).device
    dtype = next(clip_model.parameters()).dtype
    image_model.to(device)

    # ---------------------------------------------------------
    # 1. APPLY LORA TO CLIP
    # ---------------------------------------------------------
    peft_config = LoraConfig(
        r=wandb.config.lora_r,
        lora_alpha=wandb.config.lora_alpha,
        target_modules=["c_fc", "c_proj"], 
        lora_dropout=wandb.config.lora_dropout,
        bias="none",
        modules_to_save=None,
    )
    
    # Wrap the model. Base weights are frozen, LoRA weights are trainable.
    clip_model = get_peft_model(clip_model, peft_config)
    print("\nLoRA Trainable Parameters:")
    clip_model.print_trainable_parameters()

    # ---------------------------------------------------------
    # 2. SETUP ADAPTER
    # ---------------------------------------------------------
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(device).to(dtype)
    adapter.weight = nn.Parameter(cache_keys.t())

    # ---------------------------------------------------------
    # 3. SETUP OPTIMIZER
    # ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(image_model.parameters()) + list(clip_model.parameters()),
        lr=wandb.config.lr,
        eps=1e-4,
        weight_decay=wandb.config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    # Hyperparameters
    beta, alpha = wandb.config.init_beta, wandb.config.init_alpha
    gamma = wandb.config.init_gamma 
    lambda_con = wandb.config.lambda_con
    focal_gamma = wandb.config.focal_loss_gamma
    
    # Weight for Visual-Prototype JEPA Loss
    # Ideally tune this, but 1.0 is a good start for predictive tasks
    lambda_jepa = 1.0 
    
    best_acc, best_epoch = 0.0, 0
    lora_save_path = os.path.join(cfg['cache_dir'], f"best_lora_{cfg['shots']}shots")
    patience = cfg.get('patience')
    patience_counter = 0

    # ---------------------------------------------------------
    # 4. PRE-CALCULATE VISUAL PROTOTYPES (For JEPA Target)
    # ---------------------------------------------------------
    # cache_keys: [Dim, Total_Samples]
    # cache_values: [Total_Samples, N_Classes] (One-hot)
    # We aggregate keys by class to get the "Ideal Visual Prototype" per class
    with torch.no_grad():
        # --- FIX: Ensure dtypes match (float32) and move to device for multiplication ---
        keys = cache_keys.to(device).float()
        values = cache_values.to(device).float()
        
        visual_prototypes = keys @ values # [Dim, N_Classes]
        visual_prototypes = visual_prototypes / visual_prototypes.norm(dim=0, keepdim=True)
        
        # Cast back to model dtype
        visual_prototypes = visual_prototypes.to(dtype)
    
    print("\nVisual Prototypes prepared for JEPA Loss.\n")

    for train_idx in range(cfg['train_epoch']):
        adapter.train()
        image_model.train()
        clip_model.train() 
        
        correct_samples, all_samples = 0, 0
        loss_list = []
        print(f'Train Epoch: {train_idx} / {cfg["train_epoch"]}')

        for i, (batched_graphs, images, target) in enumerate(tqdm(train_loader_F)):
            batched_graphs, images, target = batched_graphs.to(device), images.to(device), target.to(device)
            
            # --- 1. Graph Forward Pass (Returns 3 Values now) ---
            # graph_visual_feature: The instance feature [Batch, Dim]
            # prototype_prediction: The predicted prototype [Batch, Dim]
            graph_visual_feature, all_updated_text_features, prototype_prediction = image_model(
                batched_graphs.x_dict,
                batched_graphs.edge_index_dict,
                batched_graphs.batch_dict
            )
            
            # --- 2. CLIP Global Forward Pass ---
            token_sequence = clip_model.encode_image(images)
            
            # Handle 3D Token Output
            if token_sequence.dim() == 3:
                image_features = token_sequence[:, 0, :] # Extract CLS token
            else:
                image_features = token_sequence

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # --- 3. Calculate Logits ---
            # A. Zero-Shot
            original_clip_logits = 100. * image_features.to(dtype) @ clip_weights.to(dtype)

            # B. Tip-Adapter Cache
            affinity = adapter(image_features.to(dtype))
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(dtype)
            
            # C. Graph Logits
            graph_visual_feature = graph_visual_feature / graph_visual_feature.norm(dim=-1, keepdim=True)
            all_updated_text_features = all_updated_text_features / all_updated_text_features.norm(dim=-1, keepdim=True)
            
            B, D, C = graph_visual_feature.shape[0], graph_visual_feature.shape[1], all_updated_text_features.shape[0] // graph_visual_feature.shape[0]
            updated_text_features_per_image = all_updated_text_features.view(B, C, D)
            visual_feature_for_bmm = graph_visual_feature.unsqueeze(1)
            text_features_for_bmm = updated_text_features_per_image.transpose(1, 2)
            graph_logits = 100. * torch.bmm(visual_feature_for_bmm, text_features_for_bmm).squeeze(1)
            
            # Combine
            tip_logits = original_clip_logits + cache_logits * alpha + graph_logits * gamma

            # ==============================================================================
            # TRUE JEPA LOSS (Visual-Prototype Prediction)
            # ==============================================================================
            
            # 1. Get Target (The fixed Visual Prototype for the correct class)
            with torch.no_grad():
                # Select correct prototype columns and transpose to [Batch, Dim]
                target_prototypes = visual_prototypes[:, target].t() 
                # target_prototypes = target_prototypes.detach() # Stop Gradient

            # 2. Normalize Prediction
            prediction_norm = prototype_prediction / prototype_prediction.norm(dim=-1, keepdim=True)
            
            # 3. Calculate Loss: 1 - CosineSimilarity(Prediction, Fixed_Prototype)
            cos_sim_jepa = (prediction_norm * target_prototypes).sum(dim=-1)
            loss_jepa = 1.0 - cos_sim_jepa.mean()
            # ==============================================================================

            # --- Standard Losses ---
            loss = F.cross_entropy(tip_logits, target)
            
            # Focal Loss
            log_pt = F.log_softmax(graph_logits, dim=1)
            pt = torch.exp(log_pt)
            pt_correct = pt[torch.arange(B, device=device), target]
            log_pt_correct = log_pt[torch.arange(B, device=device), target]
            loss_con = -torch.pow(1 - pt_correct, focal_gamma) * log_pt_correct
            loss_con = loss_con.mean() 
            
            # TOTAL LOSS
            total_loss = loss + lambda_con * loss_con + lambda_jepa * loss_jepa
            
            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            
            loss_list.append(total_loss.item())
            
            if(wandb_log):
                wandb.log({
                    "Train accuracy": correct_samples / all_samples,
                    "Total loss": sum(loss_list)/len(loss_list),
                    "Loss (Classification)": loss.item(),
                    "Loss (Focal Contrastive)": loss_con.item(),
                    "Loss (JEPA Prototype)": loss_jepa.item(),
                    "Learning rate": scheduler.get_last_lr()[0]
                })

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f'LR: {current_lr:.6f}, Acc: {correct_samples / all_samples:.4f} ({correct_samples}/{all_samples}), Loss: {sum(loss_list)/len(loss_list):.4f}')

        # --- Evaluation Phase ---
        adapter.eval()
        image_model.eval()
        clip_model.eval()
        
        test_features_list = []
        test_labels_list = []
        with torch.no_grad():
            for images, target in tqdm(test_loader, desc="Encoding Test Set"):
                images = images.to(device)
                target = target.to(device)
                
                # Use AutoCast for speed
                # with torch.cuda.amp.autocast():
                features = clip_model.encode_image(images)

                if features.dim() == 3:
                    features = features[:, 0, :]

                features = features / features.norm(dim=-1, keepdim=True)
                
                test_features_list.append(features.float())
                test_labels_list.append(target)
        
        test_features_lora = torch.cat(test_features_list, dim=0)
        test_labels_lora = torch.cat(test_labels_list, dim=0)

        with torch.no_grad():
            affinity = adapter(test_features_lora.to(dtype))
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(dtype)
            
            clip_logits = 100. * test_features_lora.to(dtype) @ clip_weights.to(dtype)
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, test_labels_lora)
        
        if wandb_log:
            wandb.log({"Test accuracy": acc, "Epoch": train_idx})

        print(f"**** Tip-Adapter-F's test accuracy: {acc:.2f}. ****\n")
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            patience_counter = 0
            if cfg.get('save_graph_checkpoint', False):
                torch.save(image_model.state_dict(), os.path.join(cfg['cache_dir'], f"best_graph_{cfg['shots']}shots.pt"))
            torch.save(adapter.state_dict(), os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"))
            clip_model.save_pretrained(lora_save_path)
        elif patience is not None:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {train_idx} after {patience} epochs without improvement.")
                break

    # --- Restore Best Models ---
    print(f"**** Loading best models from epoch {best_epoch} ****")
    adapter.load_state_dict(torch.load(os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"), map_location=device))
    clip_model.load_adapter(lora_save_path, adapter_name="default")
    
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")
    
    if wandb_log:
        wandb.summary["best_test_accuracy"] = best_acc
    
    # Re-encode Val and Test sets with the BEST model for final search and eval
    print("Re-encoding Validation set with best model for Hyperparameter Search...")
    val_features_list = []
    val_labels_list = []
    clip_model.eval()
    with torch.no_grad():
        for images, target in tqdm(val_loader, desc="Encoding Val Set"):
            images = images.to(device)
            target = target.to(device)
            # with torch.cuda.amp.autocast():
            features = clip_model.encode_image(images)
            if features.dim() == 3:
                features = features[:, 0, :]
            features = features / features.norm(dim=-1, keepdim=True)
            val_features_list.append(features.float())
            val_labels_list.append(target)
    val_features_best = torch.cat(val_features_list, dim=0)
    val_labels_best = torch.cat(val_labels_list, dim=0)
    
    print("Re-encoding Test set with best model...")
    test_features_list = []
    test_labels_list = []
    with torch.no_grad():
        for images, target in tqdm(test_loader, desc="Encoding Test Set"):
            images = images.to(device)
            target = target.to(device)
            # with torch.cuda.amp.autocast():
            features = clip_model.encode_image(images)
            if features.dim() == 3:
                features = features[:, 0, :]
            features = features / features.norm(dim=-1, keepdim=True)
            test_features_list.append(features.float())
            test_labels_list.append(target)
    test_features_best = torch.cat(test_features_list, dim=0)
    test_labels_best = torch.cat(test_labels_list, dim=0)

    print("\n-------- Searching hyperparameters on the val set. --------")
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values.to(dtype), val_features_best.to(dtype), val_labels_best, clip_weights.to(dtype), adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
    
    affinity = adapter(test_features_best.to(dtype))
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values.to(dtype)
    
    clip_logits_best = 100. * test_features_best.to(dtype) @ clip_weights.to(dtype)
    
    tip_logits = clip_logits_best + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels_best)
    final_acc = max(best_acc, acc)
    print("**** {} Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(cfg['dataset'], final_acc))
    
    if wandb_log:
        wandb.summary["final_test_accuracy_after_search"] = final_acc


def main():
    
    parser = argparse.ArgumentParser()
    # Config and dataset args
    parser.add_argument('--config', type=Path, required=True, help='Path to dataset config file')
    parser.add_argument('--shots', type=int, required=True, choices=(1, 2, 4, 8, 16), help='Number of few-shot samples')
    parser.add_argument('--root_path', '--data-root', dest='root_path', type=Path, required=True, help='Parent directory of dataset folders')
    parser.add_argument('--backbone', choices=('ViT-B/16', 'ViT-B/32'), default=None, help='Override the config backbone')
    parser.add_argument('--no_preset', action='store_true', help='Use the original script defaults rather than stored settings')
    parser.add_argument('--wandb_mode', choices=('disabled', 'offline', 'online'), default='disabled')
    # Optimizer args
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='Data-loader workers; defaults to dataset config')
    # Model architecture args
    parser.add_argument('--rgt_num_layers', type=int, default=3, help='Number of relation-aware graph layers')
    parser.add_argument('--rgt_num_heads', type=int, default=16, help='Number of attention heads per graph layer')
    parser.add_argument('--transformer_num_layers', type=int, default=3, help='Number of Transformer Encoder layers')
    parser.add_argument('--transformer_nhead', type=int, default=16, help='Number of attention heads in the transformer encoder')
    parser.add_argument('--transformer_ff_multiplier', type=int, default=2, help='Feed-forward network dimension multiplier')
    parser.add_argument('--pooling_ratio', type=float, default=0.25, help='TopKPooling ratio')
    # Regularization args
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    # Training args
    parser.add_argument('--train_epoch', type=int, default=100, help='Total training epochs')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience when enabled by dataset config')
    parser.add_argument('--init_beta', type=float, default=2.0493246903001525, help='Initial beta for Tip-Adapter')
    parser.add_argument('--init_alpha', type=float, default=9.806773071379816, help='Initial alpha for Tip-Adapter')
    parser.add_argument('--init_gamma', type=float, default=1.0, help='Initial gamma for graph logits contribution')
    parser.add_argument('--lambda_con', type=float, default=0.5, help='Weight for the direct contrastive loss')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='Gamma for Focal Loss (controls hardness focus)')

    # <<< NEW: LoRA Hyperparameters >>>
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA Rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA Alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA Dropout')

    initial_args, _ = parser.parse_known_args()
    with initial_args.config.open('r', encoding='utf-8') as config_file:
        cfg = yaml.safe_load(config_file)
    parser.set_defaults(**cfg.get('script_defaults', {}))
    if not initial_args.no_preset and initial_args.backbone != 'ViT-B/32':
        preset_key = cfg.get('preset_id', cfg['dataset'])
        preset_path = Path(__file__).resolve().parent / 'configs' / 'best_hyperparameters.json'
        with preset_path.open('r', encoding='utf-8') as preset_file:
            preset = json.load(preset_file)['datasets'][preset_key][str(initial_args.shots)]['args']
        parser.set_defaults(**preset)
    args = parser.parse_args()

    if not args.root_path.is_dir():
        parser.error(f'Dataset root does not exist: {args.root_path}')
    cfg['root_path'] = str(args.root_path.resolve())
    if args.backbone is not None:
        cfg['backbone'] = [args.backbone]
    backbone_tag = 'b32' if cfg['backbone'][0] == 'ViT-B/32' else 'b16'
    workers = args.num_workers if args.num_workers is not None else cfg.get('num_workers', 4)
    eval_batch_size = cfg.get('eval_batch_size') or args.batch_size
    persistent_workers = cfg.get('persistent_workers', False)

    # Initialize wandb
    wandb.init(
        project=cfg.get('wandb_project', 'SD-LoRA'),
        name=f"{args.shots}_shot_{backbone_tag}",
        config={key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()},
        mode=args.wandb_mode,
    ) 
    config = wandb.config
    
    cfg["shots"] = config.shots
    cache_dir = os.path.join('./caches', backbone_tag, cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['train_epoch'] = config.train_epoch
    cfg['patience'] = args.patience if args.patience is not None else cfg.get('early_stopping_patience')

    print("\nRunning configs.")
    print(cfg, "\n")
    print("Run settings:")
    print(wandb.config)
    
    # Dataset specific num_classes
    if(cfg['dataset'] == "birds"): num_classes = 200
    elif(cfg['dataset'] == "dogs"): num_classes = 120
    elif(cfg['dataset'] == "cars"): num_classes = 196
    elif(cfg['dataset'] == "oxford_pets"): num_classes = 37
    elif(cfg['dataset'] == "flowers"): num_classes = 102
    elif(cfg['dataset'] == "food101"): num_classes = 101
    elif(cfg['dataset'] == "dtd"): num_classes = 47
    elif(cfg['dataset'] == "aircrafts"): num_classes = 100
    elif(cfg['dataset'] == "ucf101"): num_classes = 101
    elif(cfg['dataset'] == "imagenet"): num_classes = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    node_types = ['vit', 'text']
    edge_types = [
        ('vit', 'intra_patch', 'vit'),
        ('vit', 'visual_to_text', 'text'),
        ('text', 'text_to_visual', 'vit'),
    ]

    # --- Load CLIP Models ---
    # Make sure to add jit=False
    clip_model_vit, vit_preprocess = clip.load(cfg['backbone'][0], device=device, jit=False)
    clip_model_vit = clip_model_vit.float()
    
    # WRAP CLIP Visual to return tokens
    clip_model_vit.visual = TokenCLIPVisual(clip_model_vit.visual)
    patch_processor = vit_preprocess

    # --- Prepare Dataset ---
    random.seed(33)
    torch.manual_seed(33)
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], shots=-1)
    
    resolution = 352 if backbone_tag == 'b32' else 336
    
    # Validation/Test Transform (Pass this to build_data_loader)
    val_transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    from datasets.utils import select_shots_herding
    
    # Note: We overwrite dataset.train_x with the selected samples
    dataset._train_x = select_shots_herding(
        data_source=dataset.train_x,
        model=clip_model_vit.visual,
        transform=val_transform,
        num_shots=cfg['shots'],  # Use the actual shots from config here
        device=device,
        batch_size=config.batch_size if cfg.get('herding_uses_train_batch_size') else 64,
        num_workers=workers,
        pin_memory=cfg.get('herding_pin_memory', False)
    )

    print("\nGetting textual features for graph nodes and CLIP's classifier.")
    text_features_for_nodes = clip_classifier(dataset.classnames, dataset.template, clip_model_vit)
    text_features_for_nodes = text_features_for_nodes.t()
    
    clip_weights = text_features_for_nodes.t()

    # --- Initialize the relation-aware graph teacher ---
    input_dims = {
        'vit': clip_model_vit.visual.output_dim,
        'text': clip_model_vit.text_projection.shape[1] 
    }
    hidden_dim = clip_model_vit.visual.output_dim

    image_model = RGTImageFeatureExtractor(
        node_types=node_types,
        edge_types=edge_types,
        input_dims=input_dims,
        hidden_channels=hidden_dim,
        rgt_num_heads=config.rgt_num_heads,
        rgt_num_layers=config.rgt_num_layers,
        dropout_rate=config.dropout_rate,
        transformer_nhead=config.transformer_nhead,
        transformer_num_layers=config.transformer_num_layers,
        transformer_ff_multiplier=config.transformer_ff_multiplier,
        transformer_activation='gelu',
        pooling_ratio=config.pooling_ratio,
        shots=config.shots
    )
    image_model.to(device)

    # --- DataLoaders ---
    # <<< INCREASED RESOLUTION FOR SINGLE PASS TOKENS >>>
    

    val_loader = build_data_loader(data_source=dataset.val, batch_size=eval_batch_size, num_workers=workers, persistent_workers=persistent_workers, is_train=False, tfm=val_transform)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=eval_batch_size, num_workers=workers, persistent_workers=persistent_workers, is_train=False, tfm=val_transform)
    
    # Train Transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=resolution, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=config.batch_size, num_workers=workers, persistent_workers=persistent_workers, is_train=True, tfm=train_transform)
    
    # Note: num_workers=0 is handled inside build_graph_data_loader in utils.py
    train_loader_F = build_graph_data_loader(
        data_source=dataset.train_x,
        batch_size=config.batch_size,
        shuffle=True,
        transform=train_transform,
        vit_model=clip_model_vit.visual, 
        text_features=text_features_for_nodes,
        processor=patch_processor,
        device=device,
        num_workers=0
    )

    # --- Pre-load features and build initial cache model ---
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model_vit, train_loader_cache)

    print("\nLoading visual features and labels from val set.")
    # Keep these for the initial Zero-Shot/Tip-Adapter evaluation before LoRA
    val_features, val_labels = pre_load_features(cfg, "val", clip_model_vit, val_loader)

    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model_vit, test_loader)

    # --- Run Initial Zero-Shot and Tip-Adapter (Pre-LoRA) ---
    run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # --- Run Fine-Tuning and Evaluation ---
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_loader, test_loader, clip_weights, clip_model_vit, image_model, train_loader_F)
    
    print("\nRunning configs.")
    print(cfg, "\n")
    print("Run settings:")
    print(wandb.config)
    
    if wandb_log:
        wandb.finish()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
