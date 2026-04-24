#!/usr/bin/env python
# coding: utf-8
"""Training script for CEONet + LES with quadrupoles.

Variant of ``fit_ceonet_les.py`` that adds per-atom quadrupoles to the
TensorReadout and feeds them to LesWrapper alongside the permanent dipoles
and polarisability-induced dipoles. Mirrors the quadrupole addition done in
``caceles-uiuQ-dipep.py`` for the dipeptides runs.
"""

import sys
import os
import logging

import numpy as np
import torch
import torch.nn as nn

import cace
from cace.representations import CEONet
from cace.modules import BesselRBF, PolynomialCutoff
from cace.modules import TensorReadout
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask
from cace.modules import LesWrapper

torch.set_default_dtype(torch.float32)
cace.tools.setup_logger(level='INFO')

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
cutoff = 4.5
logging.info("reading data")

on_cluster = False
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
    on_cluster = True
root_xyz = "/home/king1305/Apps/les_fit/data-benchmark/train-H2O_RPBE-D3.xyz"
if on_cluster:
    root_xyz = "/global/scratch/users/king1305/data/train-H2O_RPBE-D3.xyz"

collection = cace.tasks.get_dataset_from_xyz(
    train_path=root_xyz,
    valid_fraction=0.05,
    seed=1,
    cutoff=cutoff,
    data_key={'energy': 'energy', 'forces': 'forces'},
    atomic_energies={1: -5.853064337340629, 8: -2.926532168670322},
)
batch_size = 4

train_loader = cace.tasks.load_data_loader(collection=collection,
                                           data_type='train',
                                           batch_size=batch_size)
valid_loader = cace.tasks.load_data_loader(collection=collection,
                                           data_type='valid',
                                           batch_size=4)

use_device = 'cuda'
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")

# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------
logging.info("building CEONet representation")

radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

ceonet = CEONet(
    zs=[1, 8],
    n_atom_basis=4,
    cutoff=cutoff,
    radial_basis=radial_basis,
    cutoff_fn=cutoff_fn,
    max_l_cace=3,
    max_l_ceonet=2,
    max_nu_cace=3,
    n_radial_basis=12,
    nc=32,
    layers=2,
    avg_neighbors=3,
    stacking=False,
)
ceonet.to(device)

# ---------------------------------------------------------------------------
# Output modules
# ---------------------------------------------------------------------------

# Predict atomic multipoles from equivariant node features — now also quads
multipoles = TensorReadout(
    max_l=2,
    l0_key='kappas',
    l1_key='dipoles',
    l2_key=['alphas', 'quads'],
    l0_output_scale=0.1,
    l1_output_scale=1.0,
    l2_output_scale=1.0,
)

# Long-range electrostatics + dispersion via LES, now consuming the quads
les_e = LesWrapper(
    dipole_key='dipoles',
    quad_key='quads',
    alpha_key='alphas',
    energy_key='ewald_potential',
    compute_bec=False,
    make_alpha_positive=True,
    add_scalar_alpha=True,
)

# Short-range energy
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
    output_key='CACE_energy',
)

forces = cace.modules.Forces(
    energy_key='CACE_energy',
    forces_key='CACE_forces',
)

model = NeuralNetworkPotential(
    representation=ceonet,
    output_modules=[multipoles, les_e, sr_energy, e_add, forces],
)
model.to(device)

# ---------------------------------------------------------------------------
# Initialise lazy layers with one batch
# ---------------------------------------------------------------------------
logging.info("initialising lazy layers")
for batch in train_loader:
    batch.cuda()
    out = model(batch)
    for k in out:
        if out[k] is not None:
            logging.info(f"  {k}: {out[k][0]}")
        else:
            logging.info(f"{k} is None")
    break

# ---------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.1,
)
force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000,
)

e_metric = cace.tools.Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True,
)
f_metric = cace.tools.Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f',
)

# ---------------------------------------------------------------------------
# Progressive training (escalating energy loss weight)
# ---------------------------------------------------------------------------
optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}
scheduler_args = {'step_size': 20, 'gamma': 0.5}

logging.info("Stage 1: force-dominated (weight 0.1 / 1000)")
for i in range(5):
    task = TrainingTask(
        model=model,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False,
        ema_start=10,
        warmup_steps=5,
    )
    task.fit(train_loader, valid_loader, epochs=40, screen_nan=False)

task.save_model('ceonet-quads-model.pth')

logging.info("Stage 2: balanced (weight 1 / 1000)")
energy_loss = cace.tasks.GetLoss(
    target_name='energy', predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(), loss_weight=1,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)
task.save_model('ceonet-quads-model-2.pth')
model.to(device)

logging.info("Stage 3: energy-weighted (weight 10 / 1000)")
energy_loss = cace.tasks.GetLoss(
    target_name='energy', predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(), loss_weight=10,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)
task.save_model('ceonet-quads-model-3.pth')
model.to(device)

logging.info("Stage 4: energy-dominated (weight 1000 / 1000)")
energy_loss = cace.tasks.GetLoss(
    target_name='energy', predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(), loss_weight=1000,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)
task.save_model('ceonet-quads-model-4.pth')

logging.info("Finished")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Trainable parameters: {trainable_params}")
