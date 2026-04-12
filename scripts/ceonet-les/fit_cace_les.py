#!/usr/bin/env python
# coding: utf-8
"""Training script for CACE + LES long-range interactions.

Uses the CACE model from fit_cace_les_lightning.py with the non-lightning
4-stage progressive training schedule from fit_ceonet_les.py.
"""

import os
import logging

import torch

import cace
from cace.representations import Cace
from cace.modules import BesselRBF, PolynomialCutoff
from cace.modules import TensorReadout
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask
from cace.modules.les_wrapper import LesWrapper

torch.set_default_dtype(torch.float32)
cace.tools.setup_logger(level='INFO')

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
cutoff = 4.5
logging.info("reading data")

on_cluster = False
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
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
logging.info("building CACE representation")

radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[1, 8],
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_l_out=2,
    max_nu=3,
    num_message_passing=0,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    timeit=False,
)
cace_representation.to(device)

# ---------------------------------------------------------------------------
# Output modules
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
    predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.1,
)
force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='pred_force',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000,
)

e_metric = cace.tools.Metrics(
    target_name='energy',
    predict_name='pred_energy',
    name='e/atom',
    per_atom=True,
)
f_metric = cace.tools.Metrics(
    target_name='forces',
    predict_name='pred_force',
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

task.save_model('cace-model.pth')

logging.info("Stage 2: balanced (weight 1 / 1000)")
energy_loss = cace.tasks.GetLoss(
    target_name='energy', predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(), loss_weight=1,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)
task.save_model('cace-model-2.pth')
model.to(device)

logging.info("Stage 3: energy-weighted (weight 10 / 1000)")
energy_loss = cace.tasks.GetLoss(
    target_name='energy', predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(), loss_weight=10,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)
task.save_model('cace-model-3.pth')
model.to(device)

logging.info("Stage 4: energy-dominated (weight 1000 / 1000)")
energy_loss = cace.tasks.GetLoss(
    target_name='energy', predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(), loss_weight=1000,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)
task.save_model('cace-model-4.pth')

logging.info("Finished")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Trainable parameters: {trainable_params}")
