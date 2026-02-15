"""PV-enhanced QA-GNN model for Component 3."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from models import QAGNN
from utils import get_logger

logger = get_logger(__name__)


class PV_QAGNN(QAGNN):
    """Dual-stream QA-GNN with PV-gated joint attention."""

    def __init__(
        self,
        model_name: str,
        n_gnn_layers: int = 2,
        gnn_hidden_dim: int = 256,
        gnn_out_features: int = 256,
        lm_layer_features: int | None = None,
        gnn_batch_norm: bool = True,
        freeze_base_model: bool = False,
        freeze_up_to_pooler: bool = True,
        gnn_dropout: float = 0.3,
        classifier_dropout: float = 0.2,
        lm_layer_dropout: float = 0.4,
        use_roberta: bool = False,
        mask_sup_alpha_init: float = 4.0,
        mask_sup_beta_init: float = -2.0,
        mask_ref_alpha_init: float = 4.0,
        mask_ref_beta_init: float = -2.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            n_gnn_layers=n_gnn_layers,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_out_features=gnn_out_features,
            lm_layer_features=lm_layer_features,
            gnn_batch_norm=gnn_batch_norm,
            freeze_base_model=freeze_base_model,
            freeze_up_to_pooler=freeze_up_to_pooler,
            gnn_dropout=gnn_dropout,
            classifier_dropout=classifier_dropout,
            lm_layer_dropout=lm_layer_dropout,
            use_roberta=use_roberta,
        )
        # Parent QAGNN graph stacks are not used in PV_QAGNN dual-stream forward.
        # Removing them reduces unnecessary parameter/memory overhead on 8GB GPUs.
        if hasattr(self, "gnn_layers"):
            del self.gnn_layers
        if hasattr(self, "gnn_batch_norm_layers"):
            del self.gnn_batch_norm_layers

        self.mask_sup_alpha = nn.Parameter(torch.tensor(float(mask_sup_alpha_init), dtype=torch.float32))
        self.mask_sup_beta = nn.Parameter(torch.tensor(float(mask_sup_beta_init), dtype=torch.float32))
        self.mask_ref_alpha = nn.Parameter(torch.tensor(float(mask_ref_alpha_init), dtype=torch.float32))
        self.mask_ref_beta = nn.Parameter(torch.tensor(float(mask_ref_beta_init), dtype=torch.float32))
        self.gnn_out_features = int(gnn_out_features)
        self.latest_x_sup = None
        self.latest_x_ref = None
        self.latest_edge_index = None

        in_dim = self.bert.config.hidden_size
        self.gnn_layers_sup = self._build_stream_layers(
            n_gnn_layers=n_gnn_layers,
            in_dim=in_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_features,
            dropout=gnn_dropout,
        )
        self.gnn_layers_ref = self._build_stream_layers(
            n_gnn_layers=n_gnn_layers,
            in_dim=in_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_features,
            dropout=gnn_dropout,
        )

        if self.gnn_batch_norm:
            self.gnn_batch_norm_layers_sup = nn.ModuleList(
                [nn.BatchNorm1d(gnn_hidden_dim) for _ in range(n_gnn_layers - 1)]
            )
            self.gnn_batch_norm_layers_ref = nn.ModuleList(
                [nn.BatchNorm1d(gnn_hidden_dim) for _ in range(n_gnn_layers - 1)]
            )

        claim_dim = self.lm_layer.out_features if self.with_lm_layer else self.bert.config.hidden_size
        self.classifier = nn.Linear((2 * gnn_out_features) + claim_dim, 1)

    @staticmethod
    def _build_stream_layers(
        n_gnn_layers: int, in_dim: int, hidden_dim: int, out_dim: int, dropout: float
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        layers.append(GATConv(in_dim, hidden_dim, heads=1, concat=True, dropout=dropout, edge_dim=1))
        for _ in range(n_gnn_layers - 2):
            layers.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=True, dropout=dropout, edge_dim=1))
        layers.append(GATConv(hidden_dim, out_dim, heads=1, concat=True, dropout=dropout, edge_dim=1))
        return layers

    def support_gate(self, p_ent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((self.mask_sup_alpha * p_ent) + self.mask_sup_beta)

    def refute_gate(self, p_con: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((self.mask_ref_alpha * p_con) + self.mask_ref_beta)

    @staticmethod
    def _extract_pv_scores(edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_attr.dim() != 2:
            raise ValueError("Expected edge_attr to have shape [num_edges, edge_dim].")
        if edge_attr.size(1) < 5:
            raise ValueError(
                f"edge_attr must include trailing PV metadata [p_ent,p_con,p_neu,rel,pool_id]. "
                f"Got edge_dim={int(edge_attr.size(1))}."
            )
        return edge_attr[:, -5], edge_attr[:, -4]

    def _compute_joint_edge_weights(
        self, claim_embeddings: torch.Tensor, batch_graph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_attr = getattr(batch_graph, "edge_attr", None)
        if edge_attr is None:
            raise ValueError("PV_QAGNN requires `edge_attr` containing PV scores.")

        claim_embeddings_expanded = claim_embeddings[batch_graph.batch]
        node_relevance = F.cosine_similarity(claim_embeddings_expanded, batch_graph.x, dim=-1)
        source_idx, target_idx = batch_graph.edge_index
        edge_relevance = 0.5 * (node_relevance[source_idx] + node_relevance[target_idx])
        num_edges = int(edge_relevance.size(0))

        if int(edge_attr.size(0)) != num_edges:
            raise ValueError(
                "edge_attr/edge_index row mismatch in PV_QAGNN: "
                f"edge_attr_rows={int(edge_attr.size(0))}, "
                f"edge_index_edges={num_edges}, "
                f"edge_index_shape={tuple(batch_graph.edge_index.shape)}, "
                f"edge_attr_shape={tuple(edge_attr.shape)}."
            )

        p_ent, p_con = self._extract_pv_scores(edge_attr)
        gate_sup = self.support_gate(p_ent)
        gate_ref = self.refute_gate(p_con)

        weight_sup = (edge_relevance * gate_sup).unsqueeze(-1)
        weight_ref = (edge_relevance * gate_ref).unsqueeze(-1)
        return weight_sup, weight_ref

    def _run_stream(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        gnn_layers: nn.ModuleList,
        batch_norm_layers: nn.ModuleList | None,
    ) -> torch.Tensor:
        for idx, gnn_layer in enumerate(gnn_layers):
            x = gnn_layer(x, edge_index, edge_weight)
            if batch_norm_layers is not None and idx < len(gnn_layers) - 1:
                x = batch_norm_layers[idx](x)
            x = F.relu(x)
        return x

    def forward(self, claim_tokens, data_graphs):
        claim_outputs = self.bert(**claim_tokens)
        claim_embeddings = claim_outputs.last_hidden_state[:, 0]

        weight_sup, weight_ref = self._compute_joint_edge_weights(
            claim_embeddings=claim_embeddings,
            batch_graph=data_graphs,
        )

        sup_batch_norm = self.gnn_batch_norm_layers_sup if self.gnn_batch_norm else None
        ref_batch_norm = self.gnn_batch_norm_layers_ref if self.gnn_batch_norm else None

        x_sup = self._run_stream(
            x=data_graphs.x,
            edge_index=data_graphs.edge_index,
            edge_weight=weight_sup,
            gnn_layers=self.gnn_layers_sup,
            batch_norm_layers=sup_batch_norm,
        )
        x_ref = self._run_stream(
            x=data_graphs.x,
            edge_index=data_graphs.edge_index,
            edge_weight=weight_ref,
            gnn_layers=self.gnn_layers_ref,
            batch_norm_layers=ref_batch_norm,
        )
        self.latest_x_sup = x_sup
        self.latest_x_ref = x_ref
        self.latest_edge_index = data_graphs.edge_index

        support_pool = global_mean_pool(x_sup, data_graphs.batch)
        refute_pool = global_mean_pool(x_ref, data_graphs.batch)

        if self.with_lm_layer:
            claim_embeddings = self.lm_dropout(claim_embeddings)
            claim_embeddings = self.lm_layer(claim_embeddings)

        combined_features = torch.cat((support_pool, refute_pool, claim_embeddings), dim=1)
        combined_features = self.classsifier_dropout_layer(combined_features)
        logits = self.classifier(combined_features)
        return logits.squeeze(1)
