import torch
import torch.nn as nn
from typing import Dict, Sequence, Union

__all__ = ['LesWrapper']

class LesWrapper(nn.Module):
    """
    A wrapper for the LES library that does long-range interactions and BECs
    Note that CACE has its own internal implementation of the LES algorithm
    so it is not necessary to use this wrapper in CACE.
    """
    def __init__(self,
                 feature_key: Union[str, Sequence[int]] = 'node_feats',
                 energy_key: str = 'LES_energy',
                 charge_key: str = 'LES_charge',
                 bec_key: str = 'LES_BEC',
                 compute_energy: bool = True,
                 compute_bec: bool = False,
                 use_atomwise: bool = False,
                 bec_output_index: int = None, # option to compute BEC along one axis
                 ):
        super().__init__()
        from les import Les
        self.les = Les(les_arguments={"use_atomwise":use_atomwise})
 
        self.feature_key = feature_key
        self.energy_key = energy_key
        self.charge_key = charge_key
        self.bec_key = bec_key
        self.bec_output_index = bec_output_index

        self.compute_energy = compute_energy        
        self.compute_bec = compute_bec
        self.model_outputs = [charge_key]
        if compute_energy:
            self.model_outputs.append(energy_key)
        if compute_bec:
            self.model_outputs.append(bec_key)
        self.required_derivatives = []
        self.required_derivatives.append('cell')

    def set_compute_energy(self, compute_energy: bool):
        self.compute_energy = compute_energy

    def set_compute_bec(self, compute_bec: bool):
        self.compute_bec = compute_bec

    def set_bec_output_index(self, bec_output_index: int):
        self.bec_output_index = bec_output_index

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:

        # reshape the feature vectors
        if isinstance(self.feature_key, str):
            if self.feature_key not in data:
                raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
            features = data[self.feature_key]
            features = features.reshape(features.shape[0], -1)
        elif isinstance(self.feature_key, list):
            features = torch.cat([data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1)

        result = self.les(desc=features,
            positions=data['positions'],
            cell=data['cell'].view(-1, 3, 3),
            batch=data["batch"],
            compute_energy=self.compute_energy,
            compute_bec=self.compute_bec,
            bec_output_index=self.bec_output_index,
            )

        data[self.charge_key] = result['latent_charges']
        if self.compute_energy:
            data[self.energy_key] = result['E_lr']
        if self.compute_bec:
            data[self.bec_key] = result['BEC']
        return data

from cace.modules.tensornet import TensorFeedForward
from les.util import grad
from les.module import FixedCharges

class LesPolarWrapper(nn.Module):
    def __init__(self,
                 feature_key: Union[str, Sequence[int]] = 'node_feats_l',
                 e_ext_key: str = 'e_ext', #External field
                 atomic_numbers_key = "atomic_numbers",
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
        self.les = Les()

        self.feature_key = feature_key
        self.e_ext_key = e_ext_key
        self.atomic_numbers_key = atomic_numbers_key
        self.induced_q = induced_q
        self.induced_u = induced_u
        self.latent_u = latent_u

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

        self.tensor_feed_forward = TensorFeedForward(3,lomax=1)
        self.fixed_charges = FixedCharges()

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:

        # reshape the feature vectors
        if isinstance(self.feature_key, str):
            if self.feature_key not in data:
                raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
            features = data[self.feature_key] #{0: l=0, 1:l=1, 2:l=2...}
        elif isinstance(self.feature_key, list):
            features = torch.cat([data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1)

        if data[self.e_ext_key] is not None:
            e_ext = data[self.e_ext_key]
        else:
            e_ext = torch.zeros_like(data["positions"][0])
        e_ext.requires_grad = True
        assert(data["positions"].requires_grad)
        data["cell"] = data["cell"].reshape(-1,3,3)

        #Compute chi, alpha
        out = self.tensor_feed_forward(features)
        latent_charges = out[0][:,0] #[N]
        latent_kappas = out[0][:,1]**2 if self.induced_q else None #[N]
        latent_alphas = out[0][:,2]**2 if self.induced_u else None #[N]
        latent_dipoles = out[1][:,0] if self.latent_u else None #[N,3]

        #Fixed charges
        atomic_numbers = data[self.atomic_numbers_key]
        latent_charges = latent_charges + self.fixed_charges(atomic_numbers)

        #Ewald requires charge dummy index
        result = self.les(
            positions=data['positions'],
            cell=data['cell'].view(-1, 3, 3),
            latent_charges = latent_charges[:,None],
            latent_dipoles = latent_dipoles[:,None,:] if self.latent_u else None,
            latent_kappas = latent_kappas[:,None] if self.induced_q else None,
            latent_alphas = latent_alphas[:,None] if self.induced_u else None,
            atomic_numbers = None,
            batch=data["batch"],
            compute_energy=self.compute_energy,
            compute_bec=False,
        )

        #Latent charges/dipoles include induced
        # data["latent_kappas"] = latent_kappas
        # data["latent_alphas"] = latent_alphas
        data["E_lr"] = result['E_lr']
        data["latent_charges"] = result["latent_charges"] #Includes induced w/o e_ext
        data["latent_dipoles"] = result["latent_dipoles"]

        if self.compute_dipole:
            from .pol_tools import calc_E_ext
            unique_batches = torch.unique(data["batch"])
            E_ext_list = []
            mu_list = []
            alpha_list = []
            phase_list = []
            E_ext_u_list = []
            #Calculate coupling to external field
            for i in unique_batches.long():
                mask = data["batch"] == i  # Create a mask for the i-th configurations
                r_now = data["positions"][mask]
                cell_now = data["cell"][i] if (torch.linalg.det(data["cell"][i]) > 0) else None 
                q_now = data["latent_charges"][mask].squeeze()
                u_now = data["latent_dipoles"][mask].squeeze() if (self.latent_u or self.induced_u) else None
                kappa_now = latent_kappas[mask].squeeze() if self.induced_q else None
                a_now = latent_alphas[mask].squeeze() if self.induced_u else None
                E_ext, mu, alpha, phase, E_ext_u = calc_E_ext(r_now,q_now,e_ext,cell=cell_now,u=u_now,alpha=a_now,kappa=kappa_now)
                phase_list.append(phase)
                E_ext_list.append(E_ext)
                E_ext_u_list.append(E_ext_u)
                mu_list.append(mu)
                alpha_list.append(alpha)
            E_ext = torch.hstack(E_ext_list)
            E_ext_u = torch.hstack(E_ext_u_list)
            mu = torch.vstack(mu_list)
            alpha = torch.stack(alpha_list)
            phases = torch.vstack(phase_list)

            if self.via_energy_derivatives:
                from .pol_tools import dipole_from_e_ext_deriv
                dipole, dipole_u = dipole_from_e_ext_deriv(E_ext,e_ext,E_ext_u=E_ext_u,latent_dipoles=data["latent_dipoles"])
            else:
                dipole = mu
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
            if self.via_energy_derivatives:
                from .pol_tools import polarizability_from_e_ext_deriv
                polarizability = polarizability_from_e_ext_deriv(dipole,e_ext)
                if polarizability is None:
                    polarizability = torch.zeros_like(data["cell"])
                if dipole_u is not None:
                    polarizability_u = polarizability_from_e_ext_deriv(dipole_u,e_ext)
                    if polarizability_u is not None:
                        polarizability = polarizability + polarizability_u
            else:
                polarizability = alpha
        else:
            polarizability = torch.zeros_like(data["cell"])

        if self.compute_energy:
            data["LES_energy"] = result['E_lr']
        if self.compute_bec:
            data["LES_BEC"] = bec
        data["LES_dipole"] = dipole
        data["LES_polarizability"] = polarizability
        return data



















