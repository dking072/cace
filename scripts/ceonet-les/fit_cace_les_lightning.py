import os
import glob
import torch
from cace.tasks import LightningTrainingTask

import cace
from cace.representations import Cace
from cace.modules import BesselRBF, PolynomialCutoff
from cace.modules import TensorReadout
from cace.models.atomistic import NeuralNetworkPotential
from cace.modules.les_wrapper import LesWrapper

cutoff = 4.5
batch_size = 4

from cace.data.xyzdata import XYZData
on_cluster = False
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
    on_cluster = True
root_xyz = "/home/king1305/Apps/les_fit/data-benchmark/train-H2O_RPBE-D3.xyz"
if on_cluster:
    root_xyz = "/global/scratch/users/king1305/data/train-H2O_RPBE-D3.xyz"
data = XYZData(root_xyz, batch_size=batch_size, cutoff=cutoff, test_p=0)

logs_name = f"cace_les"

# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
from cace.tasks import GetLoss
e_loss = GetLoss(
    target_name="energy",
    predict_name='pred_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1,
)
f_loss = GetLoss(
    target_name="forces",
    predict_name='pred_force',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000,
)
losses = [e_loss, f_loss]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from cace.tools import Metrics
e_metric = Metrics(
    target_name="energy",
    predict_name='pred_energy',
    name='e',
    metric_keys=["rmse"],
    per_atom=True,
)
f_metric = Metrics(
    target_name="forces",
    predict_name='pred_force',
    metric_keys=["rmse"],
    name='f',
)
metrics = [e_metric, f_metric]

# ---------------------------------------------------------------------------
# Initialise lazy layers
# ---------------------------------------------------------------------------
model.cuda()
for batch in data.train_dataloader():
    batch.cuda()
    out = model(batch)
    for k in out:
        if out[k] is not None:
            print(k, out[k][0])
        else:
            print(f"{k} is None")
    break


# ---------------------------------------------------------------------------
# Resume from checkpoint if available
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
    print("Checkpoint found!", chkpt)
    print("Restarting...")

progress_bar = not on_cluster
task = LightningTrainingTask(
    model,
    losses=losses,
    metrics=metrics,
    save_pkl=True,
    logs_directory="lightning_logs",
    name=logs_name,
    scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
    optimizer_args={'lr': 0.001},
)
task.fit(data, dev_run=dev_run, max_epochs=1000, chkpt=chkpt, progress_bar=progress_bar)
