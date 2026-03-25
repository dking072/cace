import os
import glob
import torch
from cace.tasks import LightningTrainingTask

cutoff = 5.5
batch_size = 2
from cace.data.xyzdata import XYZData
on_cluster = False
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
    on_cluster = True
root_xyz = "/home/king1305/Apps/les_fit/data-benchmark/train-H2O_RPBE-D3.xyz"
if on_cluster:
    root_xyz = "/global/scratch/users/king1305/data/train-H2O_RPBE-D3.xyz"
#5% val, as we have test data
data = XYZData("/home/king1305/Apps/les_fit/data-benchmark/train-H2O_RPBE-D3.xyz", batch_size=batch_size, cutoff=cutoff, test_p=0)

latent_u = True
induced_q = True
induced_u = False
tag = ""
if latent_u:
    tag += "_u"
if induced_q:
    tag += "_iq"
if induced_u:
    tag += "_iu"
logs_name = f"caceles{tag}"

from cace.representations import Cace
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
from cace.modules import PolynomialCutoff

#Model
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

representation = Cace(
    zs=[1,8],
    n_atom_basis=3,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_l_out=2,
    max_nu=3,
    num_message_passing=1,
    type_message_passing=["M", "Ar", "Bchi"],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    timeit=False,
)

from cace.models import NeuralNetworkPotential
from cace.modules import Atomwise, Forces, FeatureAdd
from cace.modules.les_wrapper import LesPolarWrapper

atomwise = Atomwise(n_layers=3,
                    output_key="sr_energy",
                    n_hidden=[32,16],
                    n_out=1,
                    use_batchnorm=False,
                    add_linear_nn=True)

les_polar = LesPolarWrapper(
    feature_key='node_feats_l',
    compute_dipole = False,
    compute_polarizability = False,
    induced_q=induced_q,
    induced_u=induced_u,
    latent_u=latent_u,
    compute_bec=False,
)

add_energy = FeatureAdd(feature_keys=['sr_energy', 'LES_energy'],
                        output_key='pred_energy')

forces = Forces(energy_key="pred_energy",
                forces_key="pred_force")

model = NeuralNetworkPotential(
    input_modules=None,
    representation=representation,
    output_modules=[atomwise, les_polar, add_energy, forces]
)

#Losses
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
losses = [e_loss,f_loss]

#Metrics
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
metrics = [e_metric,f_metric]

#Init lazy layers
for batch in data.train_dataloader():
    exdatabatch = batch
    break
out = model(exdatabatch)
for k in out:
    print(k, out[k][0])

#Check for checkpoint and restart if found:
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
        if len(chkpts) > 0:
            chkpt = glob.glob(f"{latest_version}/checkpoints/*.ckpt")[0]
if chkpt:
    print("Checkpoint found!",chkpt)
    print("Restarting...")
    dev_run = False

progress_bar = True if not on_cluster else False
task = LightningTrainingTask(model,losses=losses,metrics=metrics,save_pkl=True,
                             logs_directory="lightning_logs",name=logs_name,
                             scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
                             optimizer_args={'lr': 0.001},
                            )
task.fit(data,dev_run=dev_run,max_epochs=1000,chkpt=chkpt,progress_bar=progress_bar)
