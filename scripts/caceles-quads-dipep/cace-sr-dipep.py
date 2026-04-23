#!/usr/bin/env python
# coding: utf-8
"""Short-range-only CACE baseline on the SPICE dipeptides dataset.

Comparison counterpart to ``caceles-quads-dipep.py``. The CACE representation
matches the old ``cace-lr-fit/fit-dipeptides/fit-dipeptides.py`` (no
``max_l_out``, Atomwise + Forces head, no LES / no learned charges). Data
loading and the 4-stage progressive training schedule are identical to the
LES script so the two runs are directly comparable.
"""

import os
import logging

import torch

import cace
from cace.representations import Cace
from cace.modules import BesselRBF, PolynomialCutoff
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)
cace.tools.setup_logger(level='INFO')

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
cutoff = 4.0
logging.info("reading data")

on_cluster = False
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
    on_cluster = True

data_dir = "/home/king1305/Apps/cace-lr-fit/fit-dipeptides/data"
if on_cluster:
    data_dir = "/global/scratch/users/king1305/data"

train_xyz = f"{data_dir}/spice-dipep-dipolar_train.xyz"
valid_xyz = f"{data_dir}/spice-dipep-dipolar_val.xyz"
test_xyz = f"{data_dir}/spice-dipep-dipolar_test.xyz"

atomic_energies = {
    1: -2.8966763404693143,  # H
    6: -6.951186908157155,   # C
    7: -5.014517770054954,   # N
    8: -4.078939874200257,   # O
}

collection = cace.tasks.get_dataset_from_xyz(
    train_path=train_xyz,
    valid_path=valid_xyz,
    test_path=test_xyz,
    cutoff=cutoff,
    data_key={'energy': 'energy', 'forces': 'force'},
    atomic_energies=atomic_energies,
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
# Representation — matches the old fit-dipeptides.py exactly
# ---------------------------------------------------------------------------
logging.info("building CACE representation (SR-only baseline)")

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
    max_nu=3,
    num_message_passing=1,
    type_message_passing=['M', 'Ar', 'Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    timeit=False,
)
cace_representation.to(device)

# ---------------------------------------------------------------------------
# Output modules — Atomwise scalar energy + autograd forces, no LES
# ---------------------------------------------------------------------------
atomwise = cace.modules.Atomwise(
    n_layers=3,
    output_key='pred_energy',
    n_hidden=[32, 16],
    n_out=1,
    use_batchnorm=False,
    add_linear_nn=True,
)

forces = cace.modules.Forces(
    energy_key='pred_energy',
    forces_key='pred_force',
)

model = NeuralNetworkPotential(
    representation=cace_representation,
    output_modules=[atomwise, forces],
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

task.save_model('cace-sr-model.pth')

logging.info("Stage 2: balanced (weight 1 / 1000)")
energy_loss = cace.tasks.GetLoss(
    target_name='energy', predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(), loss_weight=1,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)
task.save_model('cace-sr-model-2.pth')
model.to(device)

logging.info("Stage 3: energy-weighted (weight 10 / 1000)")
energy_loss = cace.tasks.GetLoss(
    target_name='energy', predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(), loss_weight=10,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)
task.save_model('cace-sr-model-3.pth')
model.to(device)

logging.info("Stage 4: energy-dominated (weight 1000 / 1000)")
energy_loss = cace.tasks.GetLoss(
    target_name='energy', predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(), loss_weight=1000,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)
task.save_model('cace-sr-model-4.pth')

logging.info("Finished")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Trainable parameters: {trainable_params}")
