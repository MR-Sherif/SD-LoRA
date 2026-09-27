# graph_builder.py

from __future__ import annotations
import torch
from torch_geometric.data import HeteroData

def build_fully_connected_hetero_graph(
        vit_f: torch.Tensor,
        text_f: torch.Tensor, # <<< Text features
        image_id: str | int
    ) -> HeteroData:
    """
    Builds a single, fully connected heterogeneous graph for an image.

    The graph has two node types: 'vit' and 'text'.
    'vit' represents feature representations of the image patches.
    'text' represents the feature for each class name in the dataset.
    The graph is fully connected within the 'vit' view and between 'vit' and 'text' views.

    Args:
        vit_f (torch.Tensor): ViT patch features. Shape: [num_patches, feature_dim_1]
        text_f (torch.Tensor): Text features for all classes. Shape: [num_classes, feature_dim_3]
        image_id (str | int): An identifier for the image.

    Returns:
        HeteroData: A PyG HeteroData object representing the graph.
    """
    data = HeteroData()
    num_patches = vit_f.shape[0]
    num_classes = text_f.shape[0]

    # --- Assign node features ---
    data['vit'].x = vit_f
    data['text'].x = text_f

    # --- Create edge indices ---

    # Helper function
    def create_dense_edges(num_src, num_dst, with_self_loops=False):
        src, dst = [], []
        for i in range(num_src):
            for j in range(num_dst):
                if not with_self_loops and i == j and num_src == num_dst:
                    continue
                src.append(i)
                dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)

    # 1. Intra-view connections (ViT patches)
    if num_patches > 1:
        edge_index_intra_visual = create_dense_edges(num_patches, num_patches)
        data['vit', 'intra_patch', 'vit'].edge_index = edge_index_intra_visual
    else:
        data['vit', 'intra_patch', 'vit'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # 2. Inter-view connections (Visual <-> Text)
    edge_index_vit_text = create_dense_edges(num_patches, num_classes, with_self_loops=True)
    data['vit', 'visual_to_text', 'text'].edge_index = edge_index_vit_text
    data['text', 'text_to_visual', 'vit'].edge_index = edge_index_vit_text.flip([0])

    data.image_id = image_id
    return data