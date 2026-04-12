import torch
import cace
from cace.representations import CEONet
from cace.modules import TensorReadout
from cace.modules.forces import Forces
from cace.modules.les_wrapper import LesWrapper
from cace.modules import BesselRBF, PolynomialCutoff
from cace.models.atomistic import NeuralNetworkPotential
from cace import data
from cace.tools import torch_geometric

cutoff = 4.5

# --- Build model ---
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
    stacking=True
)

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
)

sr_energy = cace.modules.atomwise.Atomwise(
    n_layers=3,
    output_key='SR_energy',
    n_hidden=[32, 16],
    use_batchnorm=False,
    add_linear_nn=True,
)

e_add = cace.modules.FeatureAdd(
    feature_keys=['SR_energy', 'ewald_potential'],
    output_key='pred_energy',
)

forces = Forces(
    energy_key='pred_energy',
    forces_key='pred_force',
    calc_stress=False,
)

model = NeuralNetworkPotential(
    representation=ceonet,
    output_modules=[multipoles, les_e, sr_energy, e_add, forces],
)
model.eval()

# --- Load data ---
from ase.io import read
atom         = read("test_datasets/water_dimer.xyz")
atom_rotated = read("test_datasets/water_dimer_r.xyz")

atomic_data   = data.AtomicData.from_atoms(atom,         cutoff=cutoff)
atomic_data_r = data.AtomicData.from_atoms(atom_rotated, cutoff=cutoff)
n_atoms = atom.get_global_number_of_atoms()

# ============================================================
# Equivariance of node_feats_l (representation only, no NNP)
# ============================================================
ceonet.eval()
with torch.no_grad():
    rep_out  = ceonet(atomic_data)
    rep_out_r = ceonet(atomic_data_r)

a1 = rep_out["node_feats_l"][1]
b1 = rep_out_r["node_feats_l"][1][:, :, [1, 0, 2]]
assert torch.allclose(a1, b1, atol=1e-5), "node_feats_l[1] equivariance failed."
print("node_feats_l[1] equivariance passed.")

a2 = rep_out["node_feats_l"][2]
b2 = rep_out_r["node_feats_l"][2][:, :, [1, 0, 2], :][:, :, :, [1, 0, 2]]
assert torch.allclose(a2, b2, atol=1e-5), "node_feats_l[2] equivariance failed."
print("node_feats_l[2] equivariance passed.")

# ============================================================
# Single-molecule tests through full NNP (with LES)
# ============================================================
out  = model(atomic_data,   compute_stress=False)
rout = model(atomic_data_r, compute_stress=False)

# Energy must be invariant (scalar)
assert torch.allclose(out["pred_energy"], rout["pred_energy"], atol=1e-4), \
    "Energy invariance failed."
print("Single-molecule energy invariance passed.")

# alphas: [N, 3, 3] — permute both spatial dims
aa2 = out["alphas"]
ba2 = rout["alphas"][:, [1, 0, 2], :][:, :, [1, 0, 2]]
assert torch.allclose(aa2, ba2, atol=1e-4), "alphas (l=2) equivariance failed."
print("Single-molecule alphas equivariance passed.")

# dipoles after LES: [N, 1, 3] — permute spatial dim (last)
ad1 = out["dipoles"]
bd1 = rout["dipoles"][:, :, [1, 0, 2]]
assert torch.allclose(ad1, bd1, atol=1e-4), "dipoles equivariance failed."
print("Single-molecule dipoles equivariance passed.")

# Forces: [N, 3]
af = out["pred_force"]
bf = rout["pred_force"][:, [1, 0, 2]]
assert torch.allclose(af, bf, atol=1e-4), "Force equivariance failed (single molecule)."
print("Single-molecule force equivariance passed.")

# ============================================================
# Batched tests — two copies per batch
# ============================================================
batch   = torch_geometric.Batch.from_data_list([atomic_data,   atomic_data])
batch_r = torch_geometric.Batch.from_data_list([atomic_data_r, atomic_data_r])

bout  = model(batch,   compute_stress=False)
brout = model(batch_r, compute_stress=False)

# Energy: shape [2] — one scalar per molecule in batch
assert torch.allclose(bout["pred_energy"][0:1], brout["pred_energy"][0:1], atol=1e-4), \
    "Batched energy invariance failed."
print("Batched energy invariance passed.")

# alphas: [2*N, 3, 3] — first N atoms are molecule 0
ba2_batch  = bout["alphas"][:n_atoms]
ba2_batch_r = brout["alphas"][:n_atoms, [1, 0, 2], :][:, :, [1, 0, 2]]
assert torch.allclose(ba2_batch, ba2_batch_r, atol=1e-4), "Batched alphas equivariance failed."
print("Batched alphas equivariance passed.")

# dipoles: [2*N, 1, 3] — first N atoms are molecule 0
bd1_batch  = bout["dipoles"][:n_atoms]
bd1_batch_r = brout["dipoles"][:n_atoms, :, [1, 0, 2]]
assert torch.allclose(bd1_batch, bd1_batch_r, atol=1e-4), "Batched dipoles equivariance failed."
print("Batched dipoles equivariance passed.")

# Forces for first molecule in batch
bf_batch  = bout["pred_force"][:n_atoms]
bf_batch_r = brout["pred_force"][:n_atoms, [1, 0, 2]]
assert torch.allclose(bf_batch, bf_batch_r, atol=1e-4), "Force equivariance failed (batched)."
print("Batched force equivariance passed.")

# Consistency: two identical molecules → same forces
assert torch.allclose(bout["pred_force"][:n_atoms], bout["pred_force"][n_atoms:], atol=1e-5), \
    "Batch consistency failed."
print("Batch consistency passed.")
