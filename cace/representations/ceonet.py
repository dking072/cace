import torch
import torch.nn as nn
from typing import Dict, List

from .cace_representation import Cace
from ..modules.tensornet import TensorProductLayer, TensorLinearMixing, TensorFeedForward
from ..modules.tensornet_utils import (
    expand_to, find_distances, find_moment, _scatter_add,
    normalize_tensors, single_tensor_product, irrep_tensors,
)
from ..modules import GaussianRBFCentered, PolynomialCutoff

__all__ = ["CEONet"]

class MessagePassingLayer(nn.Module):
    def __init__(self,
                 nc: int,
                 n_rbf: int,
                 avg_neighbors: int = 3,
                 lomax: int = 2,
                 linmax: int = 2,
                 mix: bool = True,
                 norm_at_beginning: bool = True,
                 stacking: bool = False,
                 irrep_mixing: bool = False,
                 linear_messages: bool = True,
                 ) -> None:
        super().__init__()
        self.lomax = lomax
        self.linmax = linmax
        self.avg_neighbors = avg_neighbors
        self.nc = nc
        self.stacking = stacking
        self.norm_at_beginning = norm_at_beginning
        self.irrep_mixing = irrep_mixing
        self.norm_func = normalize_tensors
        self.linear_layer = TensorLinearMixing
        self.linear_messages = linear_messages

        # hi x hi
        self.tp_hi_hi = TensorProductLayer(nc, max_x_way=self.linmax, max_y_way=self.linmax, max_z_way=self.lomax, stacking=stacking)
        self.hi_hi_mix_hi_left = self.linear_layer(nc, linmax)
        self.hi_hi_mix_hi_right = self.linear_layer(nc, linmax)

        # hi x hj attention
        self.att_hi_hj = TensorProductLayer(nc, max_x_way=self.linmax, max_y_way=self.linmax, max_z_way=0, stacking=True)
        self.att_hi_mix = TensorLinearMixing(nc, linmax)
        self.att_hj_mix = TensorLinearMixing(nc, linmax)

        # hi x r x hj
        self.tp_right = TensorProductLayer(nc, max_x_way=self.lomax, max_y_way=self.linmax, max_z_way=self.lomax, stacking=stacking)
        self.tp_left = TensorProductLayer(nc, max_x_way=self.linmax, max_y_way=self.lomax, max_z_way=self.lomax, stacking=stacking)
        self.hi_r_hj_mix_hi = self.linear_layer(nc, linmax)
        self.hi_r_hj_mix_r_hj = self.linear_layer(nc, lomax)
        self.hi_r_hj_mix_hj = self.linear_layer(nc, linmax)

        if self.linear_messages:
            # hi x r
            self.tp_hi_r = TensorProductLayer(nc, max_x_way=self.linmax, max_y_way=self.lomax, max_z_way=self.lomax, stacking=stacking)
            self.hi_r_mix_hi = self.linear_layer(nc, linmax)

            # r x hj
            self.tp_r_hj = TensorProductLayer(nc, max_x_way=self.lomax, max_y_way=self.linmax, max_z_way=self.lomax, stacking=stacking)
            self.r_hj_mix_hj = self.linear_layer(nc, linmax)

        # combo count for attention & radial
        combo_count = {l: 0 for l in range(self.lomax + 1)}
        tensor_layers = [self.tp_left]
        if self.linear_messages:
            tensor_layers.append(self.tp_hi_r)
            tensor_layers.append(self.tp_r_hj)
        for m in tensor_layers:
            for l in range(self.lomax + 1):
                if self.stacking:
                    combo_count[l] += len([t for t in m.combinations if t[-1] == l])
                else:
                    if len([t for t in m.combinations if t[-1] == l]) > 0:
                        combo_count[l] += 1
        if irrep_mixing:
            combo_count[0] += combo_count[2]
            combo_count[2] += combo_count[2]
        self.combo_count = combo_count

        # Radial mixing (linear, no bias to preserve cutoff)
        self.rbf_mixing_list = nn.ModuleList([
            nn.Linear(n_rbf, combo_count[l] * nc, bias=False) for l in range(lomax + 1)
        ])

        def build_mlp(nout):
            return nn.Sequential(
                nn.LazyLinear(nout, bias=True), nn.SiLU(),
                nn.Linear(nout, nout, bias=True), nn.SiLU(),
                nn.Linear(nout, nout, bias=True)
            )

        self.attention_net_list = nn.ModuleList([
            build_mlp(combo_count[l] * nc) for l in range(lomax + 1)
        ])

        self.mix = mix
        if self.mix:
            self.linear_mixing = TensorFeedForward(nc, lomax)

    def calc_hi_r_hj(self, hi: Dict[int, torch.Tensor], u: Dict[int, torch.Tensor], hj: Dict[int, torch.Tensor]):
        att_feed = []
        h1 = self.att_hi_mix(hi)
        h2 = self.att_hj_mix(hj)
        att_feed.append(self.att_hi_hj(h1, h2)[0])

        h1 = self.hi_r_hj_mix_hi(hi)
        h2 = self.hi_r_hj_mix_hj(hj)
        edge_messages = self.tp_right(u, h2)
        if self.irrep_mixing:
            edge_messages = irrep_tensors(edge_messages)
        edge_messages = self.hi_r_hj_mix_r_hj(edge_messages)
        edge_messages = self.tp_left(h1, edge_messages)

        if self.linear_messages:
            h1 = self.hi_r_mix_hi(hi)
            h1 = self.tp_hi_r(h1, u)
            for l in range(self.lomax + 1):
                edge_messages[l] = torch.hstack([edge_messages[l], h1[l]])

            h1 = self.r_hj_mix_hj(hj)
            h1 = self.tp_r_hj(u, h1)
            for l in range(self.lomax + 1):
                edge_messages[l] = torch.hstack([edge_messages[l], h1[l]])

        att_feed.append(edge_messages[0])
        att_feed = torch.hstack(att_feed)
        return edge_messages, att_feed

    def calc_edge_messages(self, data: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        n_atoms = data["atomic_numbers"].shape[0]
        idx_i = data["edge_index"][0]
        idx_j = data["edge_index"][1]
        rbf_ij = data["rbf_ij"]

        hi = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.linmax + 1):
            hi[l] = data["node_feats"][l][idx_i]

        hj = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.linmax + 1):
            hj[l] = data["node_feats"][l][idx_j]

        u = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l, rbf_mixing in enumerate(self.rbf_mixing_list):
            ones = torch.ones(rbf_ij.shape[0], self.nc, device=rbf_ij.device)
            u[l] = find_moment(data, l).unsqueeze(1) * expand_to(ones, l + 2)

        edge_messages, att_feed = self.calc_hi_r_hj(hi, u, hj)
        if self.irrep_mixing:
            edge_messages = irrep_tensors(edge_messages)

        for l in range(self.lomax + 1):
            attention = self.attention_net_list[l](att_feed)
            rbf_mixed = self.rbf_mixing_list[l](rbf_ij)
            edge_messages[l] = edge_messages[l] * expand_to(attention, l + 2)
            edge_messages[l] = edge_messages[l] * expand_to(rbf_mixed, l + 2)

        return edge_messages

    def calc_node_message_hi_hj(self, data: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        edge_messages = self.calc_edge_messages(data)

        n_atoms = data["atomic_numbers"].shape[0]
        idx_i = data["edge_index"][0]
        node_message_hi_hj = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.lomax + 1):
            node_message_hi_hj[l] = _scatter_add(edge_messages[l], idx_i, dim_size=n_atoms)
            node_message_hi_hj[l] = node_message_hi_hj[l] / self.avg_neighbors
        return node_message_hi_hj

    def self_interaction(self, h0: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        h1 = self.hi_hi_mix_hi_left(h0)
        h2 = self.hi_hi_mix_hi_right(h0)
        node_message_hi_hi = self.tp_hi_hi(h1, h2)
        if self.linear_messages:
            for l in range(self.linmax + 1):
                node_message_hi_hi[l] = torch.hstack([node_message_hi_hi[l], h0[l]])
        return node_message_hi_hi

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.norm_at_beginning:
            data["node_feats"] = self.norm_func(data["node_feats"])

        node_message_hi_hi = self.self_interaction(data["node_feats"])
        node_message_hi_hj = self.calc_node_message_hi_hj(data)

        combined_message = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.lomax + 1):
            if l in node_message_hi_hi:
                combined_message[l] = torch.hstack([node_message_hi_hi[l], node_message_hi_hj[l]])
            else:
                combined_message[l] = node_message_hi_hj[l]
        if self.irrep_mixing:
            combined_message = irrep_tensors(combined_message)

        if not self.mix:
            data["node_feats"] = combined_message
            return data

        data["node_feats"] = self.linear_mixing(combined_message)
        return data


class CEONet(nn.Module):
    """CEONet representation without orbital inputs.

    Uses CACE A-basis features (message passing step 0, via ``node_feats_l``)
    as the initial node features, then runs the CEONet tensorial message
    passing layers on top.
    """

    def __init__(self,
                 nc: int,
                 layers: int = 4,
                 n_rbf: int = 8,
                 lomax: int = 2,
                 cutoff: float = 4.5,
                 stacking: bool = False,
                 irrep_mixing: bool = False,
                 zs: List[int] = list(range(1, 54)),
                 n_atom_basis: int = 4,
                 n_radial_basis: int = 12,
                 cace_max_nu: int = 2,
                 avg_neighbors: int = 3,
                 ) -> None:
        super().__init__()
        self.nc = nc
        self.lomax = lomax
        self.layers = layers
        self.irrep_mixing = irrep_mixing
        self.norm_func = normalize_tensors

        # Radial basis shared between CACE representation and MP rbf_ij
        radial_basis = GaussianRBFCentered(n_rbf=n_rbf, cutoff=cutoff, start=1.0, trainable=True)
        cutoff_fn = PolynomialCutoff(cutoff)
        self.radial = radial_basis
        self.cutoff_fn = cutoff_fn

        # CACE representation (0 message passing steps) provides node_feats_l
        self.representation = Cace(
            zs=zs,
            n_atom_basis=n_atom_basis,
            cutoff=cutoff,
            radial_basis=radial_basis,
            cutoff_fn=cutoff_fn,
            max_l=lomax,
            max_nu=cace_max_nu,
            num_message_passing=0,
            embed_receiver_nodes=True,
            n_radial_basis=n_radial_basis,
            max_l_out=lomax,
        )

        # Mix CACE A-basis features to nc channels
        self.a_mix = TensorFeedForward(nc, lomax)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(
                nc, n_rbf=n_rbf, lomax=lomax, linmax=lomax,
                stacking=stacking, irrep_mixing=irrep_mixing,
                linear_messages=True, avg_neighbors=avg_neighbors,
            )
            for _ in range(layers)
        ])

        # B-basis construction and normalization
        nu3_combos = []
        for l1 in range(1, lomax + 1):
            for l2 in range(1, lomax + 1):
                l3 = l1 + l2
                if l3 <= lomax:
                    nu3_combos.append((l1, l3, l2))
        self.nu3_combos = nu3_combos
        num_b = 1 + lomax + len(nu3_combos)
        self.b_size = nc * num_b * (layers + 1)
        self.b_norm = nn.LayerNorm([self.b_size], bias=True)

    def calc_rbf(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        _, dij, _ = find_distances(data)
        data["rbf_ij"] = self.radial(dij[:, None]) * self.cutoff_fn(dij[:, None])
        return data

    def make_b(self, dct: Dict[int, torch.Tensor]) -> torch.Tensor:
        bfeats = [dct[0]]
        for l in range(1, self.lomax + 1):
            bfeats.append(single_tensor_product(dct[l], dct[l], (l, l, 0)))
        for l1, l2, l3 in self.nu3_combos:
            x = single_tensor_product(dct[l1], dct[l2], (l1, l2, l3))
            bfeats.append(single_tensor_product(x, dct[l3], (l3, l3, 0)))
        return torch.hstack(bfeats)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data = self.calc_rbf(data)

        # Get CACE A-basis features (node_feats_l from message passing step 0)
        cace_out = self.representation(data)
        adct = cace_out["node_feats_l"]
        batch = cace_out["batch"]  # Cace normalises None → zeros tensor

        if self.irrep_mixing:
            adct = irrep_tensors(adct)
        adct = self.norm_func(adct)

        # Mix to nc channels
        mixed = self.a_mix(adct)

        # Accumulate equivariant features and B basis across all steps
        equivariant_feats = {l: [mixed[l]] for l in range(self.lomax + 1)}
        bfeats = [self.make_b(mixed)]
        data["node_feats"] = mixed
        for mp_layer in self.mp_layers:
            data = mp_layer(data)
            for l in range(self.lomax + 1):
                equivariant_feats[l].append(data["node_feats"][l])
            bfeats.append(self.make_b(data["node_feats"]))

        # Concatenate along the channel dimension (dim=1) for each l
        node_feats_l = {l: torch.cat(equivariant_feats[l], dim=1) for l in range(self.lomax + 1)}

        # Normalize concatenated B basis
        data["node_feats"] = self.b_norm(torch.hstack(bfeats))

        return {
            "positions": data["positions"],
            "cell": data["cell"],
            "batch": batch,
            "node_feats": data["node_feats"],
            "node_feats_l": node_feats_l,
        }
