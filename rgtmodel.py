from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

class RGTImageFeatureExtractor(nn.Module):
    """Relation-aware graph teacher using PyG's HGTConv for typed-edge attention."""

    def __init__(self, node_types, edge_types, input_dims, hidden_channels,
                 rgt_num_heads, rgt_num_layers, dropout_rate,
                 transformer_nhead, transformer_num_layers,
                 transformer_ff_multiplier, transformer_activation, shots,
                 pooling_ratio: float):
        super().__init__()

        self.node_types = node_types
        self.visual_node_types = [nt for nt in node_types if nt != 'text']
        self.metadata = (self.node_types, edge_types)
        self.hidden_channels = hidden_channels
        self.num_rgt_layers = rgt_num_layers
        self.pooling_ratio = pooling_ratio

        # --- Input Projection ---
        self.input_proj = nn.ModuleDict()
        for node_type in self.node_types:
            self.input_proj[node_type] = nn.Linear(input_dims[node_type], hidden_channels)

        # --- Transformer Encoders ---
        self.transformer_encoders = nn.ModuleDict()
        for node_type in self.node_types:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels, nhead=transformer_nhead,
                dim_feedforward=hidden_channels * transformer_ff_multiplier,
                dropout=dropout_rate, activation=transformer_activation, batch_first=True
            )
            self.transformer_encoders[node_type] = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_num_layers
            )

        # --- Relation-aware graph layers ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_rgt_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, self.metadata, rgt_num_heads))
            norm_dict = nn.ModuleDict()
            for node_type in self.node_types:
                norm_dict[node_type] = nn.LayerNorm(hidden_channels)
            self.norms.append(norm_dict)
            
        self.dropout = nn.Dropout(dropout_rate)
        
        # --- Saliency Scorer ---
        self.saliency_scorer = nn.Linear(hidden_channels, 1)
        
        self.jepa_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels) 
        )

    def forward(self, x_dict, edge_index_dict, batch_dict):
        # --- 1. Projection ---
        projected_x_dict = {}
        for node_type, x_features in x_dict.items():
            projected_x_dict[node_type] = self.input_proj[node_type](x_features)

        # --- 2. Transformer Encoder (Refined for Dynamic Tokens) ---
        transformed_x_dict = {}
        
        # Handle VIT (Visual Tokens)
        vit_x = projected_x_dict['vit']
        vit_batch = batch_dict['vit']
        
        # Calculate Batch Size and Nodes per Graph dynamically
        batch_size = vit_batch.max().item() + 1
        num_nodes_per_graph = vit_x.shape[0] // batch_size
        
        # Reshape to [Batch, Num_Nodes, Dim] for Transformer
        if vit_x.shape[0] == batch_size * num_nodes_per_graph:
            vit_seq = vit_x.view(batch_size, num_nodes_per_graph, -1)
            vit_out = self.transformer_encoders['vit'](vit_seq)
            transformed_x_dict['vit'] = vit_out.view(-1, self.hidden_channels)
        else:
            # Fallback if batch sizes are irregular
            transformed_x_dict['vit'] = vit_x
            
        # Handle Text
        transformed_x_dict['text'] = projected_x_dict['text']

        # --- 3. Relation-aware graph message passing ---
        current_x_dict = transformed_x_dict
        for conv, norm_dict in zip(self.convs, self.norms):
            x_out = conv(current_x_dict, edge_index_dict)
            for node_type in x_out.keys():
                out = self.dropout(norm_dict[node_type](x_out[node_type]).relu())
                current_x_dict[node_type] = current_x_dict[node_type] + out
        
        # --- 4. Residual Saliency Pooling (CLS + TopK Patches) ---
        
        vit_features = current_x_dict['vit']
        
        # Create mask for CLS tokens (indices 0, num_nodes_per_graph, 2*num_nodes_per_graph...)
        # CRITICAL FIX: Use dynamic stride based on actual nodes per graph
        is_cls = torch.zeros(vit_features.size(0), dtype=torch.bool, device=vit_features.device)
        is_cls[::num_nodes_per_graph] = True
        
        # A. Extract Global CLS Features
        cls_features = vit_features[is_cls] # [Batch, Dim]
        
        # B. Extract Patch Features
        patch_features = vit_features[~is_cls] # [Batch * Num_Patches, Dim]
        
        # Calculate dynamic number of patches
        num_patches = num_nodes_per_graph - 1
        
        # C. Calculate Saliency Scores
        scores = self.saliency_scorer(patch_features).squeeze(-1)
        
        # Select TopK patches per image
        k = int(num_patches * self.pooling_ratio)
        if k < 1: k = 1 # Safety check
        
        # Reshape scores to [Batch, Num_Patches] using dynamic patch count
        scores_reshaped = scores.view(batch_size, num_patches)
        patch_features_reshaped = patch_features.view(batch_size, num_patches, -1)
        
        topk_scores, topk_indices = torch.topk(scores_reshaped, k, dim=1)
        
        # Gather the topk features
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_channels)
        selected_patches = torch.gather(patch_features_reshaped, 1, topk_indices_expanded)
        
#         # Average the selected salient patches
#         saliency_feature = selected_patches.mean(dim=1) # [Batch, Dim]
        
#         # Final Visual Feature: Global Context + Refined Local Details
#         graph_visual_feature = cls_features + saliency_feature

        saliency_feature = selected_patches.sum(dim=1) 

        # 2. Normalize (Fixes the volume so it doesn't drown out CLS)
        # This preserves the "direction" of the sum (the information) but fixes the "scale"
        saliency_feature = F.layer_norm(saliency_feature, (self.hidden_channels,))

        # 3. Add (Now both CLS and Patches contribute equally)
        graph_visual_feature = cls_features + saliency_feature
        
        all_updated_text_features = current_x_dict['text']
        
        prototype_prediction = self.jepa_predictor(graph_visual_feature)
        
        return graph_visual_feature, all_updated_text_features, prototype_prediction
