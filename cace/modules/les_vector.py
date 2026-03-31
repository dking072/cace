import torch
import torch.nn as nn
from typing import Dict, Sequence, Union

from cace.modules.tensornet import TensorFeedForward
from les.util import grad
from les.module import FixedCharges
from ..tools import scatter_sum

class LesVectorWrapper(nn.Module):
    def __init__(self,
                 feature_key: Union[str, Sequence[int]] = 'node_feats_l',
                 e_ext_key: str = 'e_ext', #External field
                 atomic_numbers_key = "atomic_numbers",
                 n_scf: int = 0,
                 compute_energy: bool = True,
                 compute_dipole: bool = True,
                 compute_polarizability: bool = True,
                 compute_bec: bool = False,
                 via_energy_derivatives: bool = False,
                 induced_q: bool = False,
                 induced_u: bool = True,
                 latent_u: bool = True,
                 epsilon_factor: float = 1,
                 ):
        super().__init__()
        from les.les import Les
        self.les = Les() #Just for bec

        self.feature_key = feature_key
        self.e_ext_key = e_ext_key
        self.atomic_numbers_key = atomic_numbers_key
        self.induced_q = induced_q
        self.induced_u = induced_u
        self.latent_u = latent_u
        self.n_scf = n_scf

        self.epsilon_factor = epsilon_factor
        self.normalization_factor = epsilon_factor ** 0.5
        self.compute_energy = compute_energy
        self.compute_dipole = compute_dipole
        self.compute_polarizability = compute_polarizability
        self.compute_bec = compute_bec
        self.via_energy_derivatives = via_energy_derivatives
        if self.compute_dipole or self.compute_polarizability:
            self.compute_energy = True
        if self.compute_bec:
            self.compute_dipole = True
        if self. compute_polarizability:
            self.compute_dipole = True
        self.model_outputs = ["LES_energy"]
        if self.compute_dipole:
            self.model_outputs.append("LES_dipole")
        if self.compute_bec:
            self.model_outputs.append("LES_BEC")
        if self.compute_polarizability:
            self.model_outputs.append("LES_polarizability")
        self.required_derivatives = []
        self.required_derivatives.append('cell')

        #0: q, g0, fukui
        #1: u, kappa
        #2: g1, alpha
        self.tensor_feed_forward = TensorFeedForward(n_scf+4,lomax=2)
        # self.g_readout = nn.Sequential(
        #     nn.LazyLinear(out_features=32),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32,out_features=1)
        # )
        self.fixed_charges = FixedCharges()

    def compute_ewald(self,r,q,batch,u=None,cell=None):
        unique_batches = torch.unique(batch)  # Get unique batch indices

        e_lr_results = []
        field_results = []
        for i in unique_batches.long():
            mask = batch == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now = r[mask], q[mask]

            u_now = u[mask] if u is not None else None
            box_now = cell[i] if cell is not None else None # Get the box for the i-th configuration
            
            # check if the box is periodic or not
            if box_now is None or torch.linalg.det(box_now) < 1e-6:
                # the box is not periodic, we use the direct sum
                result = self.les.ewald.compute_potential_realspace(r_raw=r_raw_now, q=q_now, u=u_now, 
                                                          compute_field=True
                                                          )
            else:
                # the box is periodic, we use the reciprocal sum
                result = self.les.ewald.compute_potential_triclinic(r_raw=r_raw_now, q=q_now, 
                                                          cell_now=box_now, u=u_now, 
                                                          compute_field=True
                                                          )
            e_lr_results.append(result["pot"])
            field_results.append(result["field"])
        e_lr = torch.hstack(e_lr_results)
        field = torch.vstack(field_results)
        return e_lr, field

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:

        # reshape the feature vectors
        if isinstance(self.feature_key, str):
            if self.feature_key not in data:
                raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
            features = data[self.feature_key] #{0: l=0, 1:l=1, 2:l=2...}
        elif isinstance(self.feature_key, list):
            features = torch.cat([data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1)

        e_ext = torch.zeros_like(data["positions"][0])
        e_ext.requires_grad = True
        assert(data["positions"].requires_grad)
        data["cell"] = data["cell"].reshape(-1,3,3)

        #Compute chi, alpha
        out = self.tensor_feed_forward(features)
        latent_charges = out[0][:,0] #[N,1]
        latent_g0 = out[0][:,1] if self.induced_q else None #[N,1]
        latent_fukui = out[0][:,2:] #[N,n_scf+]
        latent_dipoles = out[1][:,0] if self.latent_u else None #[N,3]
        latent_kappas = out[1][:,2:] if self.induced_q else None #[N,n_scf+,3]
        latent_g1 = out[2][:,0] if self.induced_u else None #[N,3,3]
        latent_alphas = out[2][:,1:] if self.induced_u else None #[N,n_scf+,3,3]

        #Atomic charges:
        atomic_numbers = data[self.atomic_numbers_key]
        latent_charges = latent_charges + self.fixed_charges(atomic_numbers)

        #Neutralize charges:
        fukui_now = latent_fukui[:,0]
        latent_charges = latent_charges - fukui_now/fukui_now.sum() * latent_charges.sum()

        #Enforce PSD via AA^T
        latent_g0 = latent_g0**2 if self.induced_q else None
        latent_alphas = torch.einsum("ncij,nckj->ncik",latent_alphas,latent_alphas) if self.induced_u else None
        latent_g1 = torch.einsum("nij,nkj->nik",latent_g1,latent_g1) if self.induced_u else None

        #Do SCF
        q_induced_tot, u_induced_tot = 0, 0
        for i in range(self.n_scf+1):
            #Calc e_lr and field
            e_lr, field = self.compute_ewald(data["positions"],latent_charges,batch=data["batch"],cell=data["cell"],u=latent_dipoles)
            field = field.squeeze() + e_ext[None,:]

            #Induced q
            if self.induced_q:
                fukui_now = latent_fukui[:,i+1]
                kappas_now = latent_kappas[:,i,:]
                q_induced_now = torch.einsum("ni,ni->n",kappas_now,field)
                q_induced_now = q_induced_now - fukui_now/fukui_now.sum() * q_induced_now.sum()
                latent_charges = latent_charges + q_induced_now
                q_induced_tot = q_induced_tot + q_induced_now

            #Induced u
            if self.induced_u:
                alphas_now = latent_alphas[:,i,:,:]
                u_induced_now = torch.einsum("nij,nj->ni",alphas_now,field)
                if latent_dipoles is None:
                    latent_dipoles = u_induced_now
                else:
                    latent_dipoles = latent_dipoles + u_induced_now
                u_induced_tot = u_induced_tot + u_induced_now

        #Calculate final energy:
        if self.induced_q or self.induced_u:
            e_lr, field = self.compute_ewald(data["positions"],latent_charges,data["batch"],u=latent_dipoles,cell=data["cell"])

        #Calculate self-energies
        e_g = torch.zeros_like(latent_charges)
        if self.induced_q:
            e_g = e_g + (q_induced_tot * latent_g0)
        if self.induced_u:
            g1 = torch.einsum("nij,nj->ni",latent_g1,u_induced_tot)
            g1 = torch.einsum("ni,ni->n",u_induced_tot,g1)
            e_g = e_g + g1
        e_g = scatter_sum(
            src=e_g,
            index=data["batch"],
            dim=0,
            dim_size=data["batch"].max().item() + 1  # Ensures correct batch sizing
        )
        e_lr = e_lr + e_g

        #Latent charges/dipoles include induced
        data["latent_charges"] = latent_charges
        data["latent_dipoles"] = latent_dipoles

        if self.compute_dipole:
            from .pol_tools import calc_E_ext
            unique_batches = torch.unique(data["batch"])
            E_ext_list = []
            mu_list = []
            mu_u_list = []
            phase_list = []
            E_ext_u_list = []
            #Calculate coupling to external field
            for i in unique_batches.long():
                mask = data["batch"] == i  # Create a mask for the i-th configurations
                r_now = data["positions"][mask]
                cell_now = data["cell"][i] if (torch.linalg.det(data["cell"][i]) > 0) else None 
                q_now = data["latent_charges"][mask].squeeze()
                u_now = data["latent_dipoles"][mask].squeeze() if (self.latent_u or self.induced_u) else None
                E_ext, mu, mu_u, _, phase, E_ext_u = calc_E_ext(r_now,q_now,e_ext,cell=cell_now,u=u_now,alpha=None,kappa=None)
                phase_list.append(phase)
                E_ext_list.append(E_ext)
                E_ext_u_list.append(E_ext_u)
                mu_list.append(mu)
                mu_u_list.append(mu_u)
            E_ext = torch.hstack(E_ext_list)
            E_ext_u = torch.hstack(E_ext_u_list)
            mu = torch.vstack(mu_list)
            mu_u = torch.vstack(mu_u_list)
            phases = torch.vstack(phase_list)

            if self.via_energy_derivatives:
                from .pol_tools import dipole_from_e_ext_deriv
                dipole, dipole_u = dipole_from_e_ext_deriv(E_ext,e_ext,E_ext_u=E_ext_u,latent_dipoles=data["latent_dipoles"])
            else:
                dipole, dipole_u = mu, mu_u
        else:
            dipole = torch.zeros_like(data["positions"][0])

        if self.compute_bec:
            if self.via_energy_derivatives:
                # print("Dipole:",dipole)
                bec = grad(y=dipole,x=data["positions"]).transpose(1,2).contiguous()
                bec = bec * phases.unsqueeze(2).conj()
                bec = bec.real
                if dipole_u is not None:
                    # print("Dipole_u:",dipole_u)
                    bec_u = grad(y=dipole_u,x=data["positions"]).transpose(1,2).contiguous()
                    bec = bec + bec_u
            else:
                bec = self.les.bec(q=data["latent_charges"],
                            u=data["latent_dipoles"],
                            r=data["positions"],
                            cell=data["cell"],
                            batch=data["batch"],
                    )
                if data["latent_dipoles"] is not None:
                    bec = bec.sum(dim=1)
                    
        if self.compute_polarizability:
            from .pol_tools import polarizability_from_e_ext_deriv
            dipole_tot = (dipole + dipole_u) if dipole_u is not None else dipole
            polarizability = polarizability_from_e_ext_deriv(dipole_tot,e_ext)
            if polarizability is None:
                polarizability = torch.zeros_like(data["cell"])
        else:
            polarizability = torch.zeros_like(data["cell"])

        if self.compute_energy:
            data["LES_energy"] = e_lr
        if self.compute_bec:
            data["LES_BEC"] = bec
        data["LES_dipole"] = dipole
        data["LES_polarizability"] = polarizability
        return data

