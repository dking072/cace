#!/usr/bin/env python
# coding: utf-8
"""CACE + LES with permanent dipoles + induced charges + induced dipoles.

Variant of ``caceles-uiuQ-dipep.py``. TensorReadout emits per-atom kappas,
dipoles, and alphas; LesWrapper uses dipoles (permanent), kappa (induces
charges via the Ewald field), and alpha (induces dipoles via the field).
No quadrupole.
"""

import os
import glob

import torch
import lightning as L

import cace
from cace.representations import Cace
from cace.modules import BesselRBF, PolynomialCutoff
from cace.modules import TensorReadout
from cace.models.atomistic import NeuralNetworkPotential
from cace.modules.les_wrapper import LesWrapper
from cace.tasks import LightningTrainingTask, GetLoss
from cace.tools import Metrics, torch_geometric
from cace.data.atomic_data import AtomicData
from cace.tasks.load_data import get_dataset_from_xyz

torch.set_float32_matmul_precision('medium')


class DipepData(L.LightningDataModule):
    """LightningDataModule for the pre-split SPICE dipeptides XYZ files."""

    def __init__(self, train_xyz, valid_xyz, test_xyz, cutoff=4.0, batch_size=4,
                 data_key=None, atomic_energies=None):
        super().__init__()
        self.train_xyz = train_xyz
        self.valid_xyz = valid_xyz
        self.test_xyz = test_xyz
        self.cutoff = cutoff
        self.batch_size = batch_size
        self.data_key = data_key or {'energy': 'energy', 'forces': 'force'}
        self.atomic_energies = atomic_energies
        try:
            self.num_workers = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        except KeyError:
            self.num_workers = max(1, (os.cpu_count() or 2) - 1)
        self.prepare_data()

    def _build(self, configs):
        return [
            AtomicData.from_atoms(a, cutoff=self.cutoff, data_key=self.data_key,
                                  atomic_energies=self.atomic_energies)
            for a in configs
        ]

    def prepare_data(self):
        collection = get_dataset_from_xyz(
            train_path=self.train_xyz,
            valid_path=self.valid_xyz,
            test_path=self.test_xyz,
            cutoff=self.cutoff,
            data_key=self.data_key,
            atomic_energies=self.atomic_energies,
        )
        self.train_dataset = self._build(collection.train)
        self.valid_dataset = self._build(collection.valid)
        self.test_dataset = self._build(collection.test)

    def train_dataloader(self):
        return torch_geometric.DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, drop_last=True, num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch_geometric.DataLoader(
            self.valid_dataset, batch_size=self.batch_size,
            shuffle=False, drop_last=False, num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch_geometric.DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, drop_last=False, num_workers=self.num_workers,
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
cutoff = 4.0
batch_size = 4

on_cluster = 'SLURM_JOB_CPUS_PER_NODE' in os.environ
data_dir = ("/global/scratch/users/king1305/data"
            if on_cluster else
            "/home/king1305/Apps/cace-lr-fit/fit-dipeptides/data")

atomic_energies = {
    1: -2.8966763404693143,  # H
    6: -6.951186908157155,   # C
    7: -5.014517770054954,   # N
    8: -4.078939874200257,   # O
}

data = DipepData(
    train_xyz=f"{data_dir}/spice-dipep-dipolar_train.xyz",
    valid_xyz=f"{data_dir}/spice-dipep-dipolar_val.xyz",
    test_xyz=f"{data_dir}/spice-dipep-dipolar_test.xyz",
    cutoff=cutoff,
    batch_size=batch_size,
    data_key={'energy': 'energy', 'forces': 'force'},
    atomic_energies=atomic_energies,
)

logs_name = "caceles_uiqiu_dipep"

# ---------------------------------------------------------------------------
# Representation — l<=2 for the alpha tensor
# ---------------------------------------------------------------------------
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[1, 6, 7, 8],
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=4,
    max_l_out=2,
    max_nu=3,
    num_message_passing=1,
    type_message_passing=['M', 'Ar', 'Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    timeit=False,
)

# ---------------------------------------------------------------------------
# Output modules — dipoles + induced charges (kappa) + induced dipoles (alpha)
# ---------------------------------------------------------------------------
multipoles = TensorReadout(
    max_l=2,
    l0_key='kappas',
    l1_key='dipoles',
    l2_key='alphas',
    l0_output_scale=0.1,
    l1_output_scale=1.0,
    l2_output_scale=1.0,
)

les_e = LesWrapper(
    dipole_key='dipoles',
    kappa_key='kappas',
    alpha_key='alphas',
    energy_key='ewald_potential',
    compute_bec=False,
    make_alpha_positive=True,
    add_scalar_alpha=True,
)

sr_energy = cace.modules.atomwise.Atomwise(
    n_layers=3,
    output_key='SR_energy',
    n_hidden=[32, 16],
    use_batchnorm=False,
    add_linear_nn=True,
    output_scale=1.0,
)

e_add = cace.modules.FeatureAdd(
    feature_keys=['SR_energy', 'ewald_potential'],
    output_key='pred_energy',
)

forces = cace.modules.Forces(
    energy_key='pred_energy',
    forces_key='pred_force',
)

model = NeuralNetworkPotential(
    representation=cace_representation,
    output_modules=[multipoles, les_e, sr_energy, e_add, forces],
)

# ---------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------
e_loss = GetLoss(target_name='energy', predict_name='pred_energy',
                 loss_fn=torch.nn.MSELoss(), loss_weight=1)
f_loss = GetLoss(target_name='forces', predict_name='pred_force',
                 loss_fn=torch.nn.MSELoss(), loss_weight=1000)
losses = [e_loss, f_loss]

e_metric = Metrics(target_name='energy', predict_name='pred_energy',
                   name='e', metric_keys=['rmse'], per_atom=True)
f_metric = Metrics(target_name='forces', predict_name='pred_force',
                   name='f', metric_keys=['rmse'])
metrics = [e_metric, f_metric]

# ---------------------------------------------------------------------------
# Initialise lazy layers
# ---------------------------------------------------------------------------
for batch in data.train_dataloader():
    model(batch)
    break

# ---------------------------------------------------------------------------
# Checkpoint/restart detection
# ---------------------------------------------------------------------------
chkpt = None
dev_run = False
if os.path.isdir(f"lightning_logs/{logs_name}"):
    latest_version = None
    num = 0
    while os.path.isdir(f"lightning_logs/{logs_name}/version_{num}"):
        latest_version = f"lightning_logs/{logs_name}/version_{num}"
        num += 1
    if latest_version:
        chkpts = glob.glob(f"{latest_version}/checkpoints/*.ckpt")
        if chkpts:
            chkpt = chkpts[0]
if chkpt:
    print(f"Checkpoint found: {chkpt}\nRestarting...")

progress_bar = not on_cluster
task = LightningTrainingTask(
    model, losses=losses, metrics=metrics, save_pkl=True,
    logs_directory='lightning_logs', name=logs_name,
    scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
    optimizer_args={'lr': 0.01},
)
task.fit(data, dev_run=dev_run, max_epochs=1000, chkpt=chkpt,
         progress_bar=progress_bar)
