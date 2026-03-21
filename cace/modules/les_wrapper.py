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
                 compute_dipole: bool = True,
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
        self.compute_polarizability = self.induced_u
        if via_energy_derivatives and self.induced_q:
            self.compute_polarizability = None

        self.epsilon_factor = epsilon_factor
        self.normalization_factor = epsilon_factor ** 0.5
        self.compute_dipole = compute_dipole
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

    def calc_E_ext(self,r_raw,q,e_ext,cell=None,u=None,q_induced=None,u_induced=None,subtract_mean=True):
        #q induced not supported
        if cell is None:
            r_raw = r_raw - r_raw.mean(dim=0)[None,:]
            phase = torch.ones_like(r_raw)
        if cell is not None:
            r_frac = torch.matmul(r_raw, torch.linalg.inv(cell)) #[N,3]
            phase = torch.exp(1j * 2.* torch.pi * r_frac) #[N,3]
            r_raw = torch.matmul(phase, cell / (1j * 2.* torch.pi)) #[N,3]

        #Potentially imaginary
        q = q - q.mean()
        mu = (r_raw * q[:,None]).sum(dim=0)
        if q_induced is not None:
            q_induced = q_induced - q_induced.mean()
            dip_induced = (r_raw * q_induced[:,None]).sum(dim=0)
            mu = mu + dip_induced
        E_ext = -(e_ext * mu).sum()

        mu_u = 0
        if u is not None:
            mu_u = u.sum(dim=0)
            mu = mu + mu_u
            E_ext_u = -(e_ext * mu_u).sum()
        if u_induced is not None:
            mu_induced = u_induced.sum(dim=0)
            mu = mu + mu_induced
            mu_u = mu_u + mu_induced
            E_ext_u = -(e_ext * mu_u).sum()
        E_ext_u = -(e_ext * mu_u).sum()
        if cell is None:
            E_ext = E_ext + E_ext_u
        return E_ext, mu, phase, E_ext_u

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
            e_ext = e_ext,
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
        data["latent_kappas"] = latent_kappas
        data["latent_alphas"] = latent_alphas
        data["E_lr"] = result['E_lr']
        data["latent_charges"] = result["latent_charges"]
        data["latent_dipoles"] = result["latent_dipoles"]
        data["q_induced"] = result["q_induced"] #Zero if none
        data["u_induced"] = result["u_induced"]

        if self.compute_dipole:
            unique_batches = torch.unique(data["batch"])
            E_ext_list = []
            mu_list = []
            alpha_list = []
            phase_list = []
            E_ext_u_list = []
            #Calculate coupling to external field
            for i in unique_batches.long():
                mask = data["batch"] == i  # Create a mask for the i-th configurations
                r_now, q_now = data["positions"][mask], latent_charges[mask].squeeze()
                cell_now = data["cell"][i]
                if torch.linalg.det(cell_now) == 0:
                    cell_now = None
                    periodic = False
                else:
                    periodic = True
                u_now = latent_dipoles[mask].squeeze() if self.latent_u else None
                u_induced_now = data["u_induced"][mask].squeeze() if self.induced_u else None
                q_induced_now = data["q_induced"][mask].squeeze() if self.induced_q else None
                a_now = data["latent_alphas"][mask].squeeze() if self.induced_u else torch.zeros_like(q_now)
                E_ext, mu, phase, E_ext_u = self.calc_E_ext(r_now,q_now,e_ext,cell=cell_now,u=u_now,u_induced=u_induced_now,q_induced=q_induced_now)
                phase_list.append(phase)
                E_ext_list.append(E_ext)
                E_ext_u_list.append(E_ext_u)
                mu_list.append(mu)
                alpha_list.append(a_now.sum() * torch.eye(3,device=r_now.device))
            E_ext = torch.hstack(E_ext_list)
            E_ext_u = torch.hstack(E_ext_u_list)
            mu = torch.vstack(mu_list)
            alpha = torch.stack(alpha_list)
            phases = torch.vstack(phase_list)

            if self.via_energy_derivatives:
                #Can't batch non-periodic with periodic
                n_out = E_ext.shape[0]
                eye = torch.eye(n_out, device=E_ext.device, dtype=e_ext.dtype)

                dipole_real = -torch.autograd.grad(
                    outputs=E_ext.real,
                    inputs=e_ext,
                    grad_outputs=eye,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=False,
                    is_grads_batched=True,
                )[0]  # [n_out, 3]

                if torch.is_complex(E_ext):
                    dipole_imag = -torch.autograd.grad(
                        outputs=E_ext.imag,
                        inputs=e_ext,
                        grad_outputs=eye,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=False,
                        is_grads_batched=True,
                    )[0]  # [n_out, 3]
                    dipole = dipole_real + 1j * dipole_imag
                else:
                    dipole = dipole_real

                if periodic:
                    dipole_u = -torch.autograd.grad(
                        outputs=E_ext_u,
                        inputs=e_ext,
                        grad_outputs=eye,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=False,
                        is_grads_batched=True,
                    )[0]  # [n_out, 3]
            else:
                dipole = mu
        else:
            dipole = None

        if self.compute_bec:
            if self.via_energy_derivatives:
                bec = grad(y=dipole,x=data["positions"]).transpose(1,2).contiguous()
                bec = bec * phases.unsqueeze(2).conj()
                bec = bec.real
                if periodic:
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
                    
        if self.compute_polarizability is None:
            polarizability = alpha
        elif self.compute_polarizability:
            if self.via_energy_derivatives:
                dipole_pol = dipole_u if periodic else dipole
                n_out, dim = dipole_pol.shape
                n_total = n_out * dim
                dipole_flat = dipole_pol.reshape(-1)
                grad_outputs = torch.eye(
                    n_total,
                    device=dipole_pol.device,
                    dtype=dipole_pol.dtype,
                )
                polarizability_flat = torch.autograd.grad(
                    outputs=dipole_flat,
                    inputs=e_ext,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                    is_grads_batched=True,
                )[0]
                polarizability = 0.5 * polarizability_flat.view(n_out, dim, dim)
            else:
                polarizability = alpha
        else:
            polarizability = torch.zeros_like(data["cell"])

        if self.compute_energy:
            data["LES_energy"] = result['E_lr']
        if self.compute_bec:
            data["LES_BEC"] = bec
        if self.compute_dipole:
            data["LES_dipole"] = dipole
        data["LES_polarizability"] = polarizability
        return data







































class LesVectorWrapper(nn.Module):
    def __init__(self,
                 feature_key: Union[str, Sequence[int]] = 'node_feats_l',
                 e_ext_key: str = "e_ext",
                 compute_dipole: bool = True,
                 compute_polarizability: bool = True,
                 via_energy_derivatives: bool = False,
                 compute_bec: bool = False,
                 bec_output_index: int = None, # option to compute BEC along one axis
                 ):
        super().__init__()
        from les.les import Les
        self.les = Les()
 
        self.feature_key = feature_key
        self.e_ext_key = e_ext_key
        self.bec_output_index = bec_output_index

        self.compute_dipole = compute_dipole
        self.compute_bec = compute_bec
        self.compute_polarizability = compute_polarizability
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

    def set_compute_energy(self, compute_energy: bool):
        self.compute_energy = compute_energy

    def set_compute_bec(self, compute_bec: bool):
        self.compute_bec = compute_bec

    def set_bec_output_index(self, bec_output_index: int):
        self.bec_output_index = bec_output_index

    def calc_E_ext(self,r_raw,q,e_ext,u=None,q_induced=None,u_induced=None):
        r_raw = r_raw - r_raw.mean(dim=0)[None,:]
        mu = (r_raw * q[:,None]).sum(dim=0)
        if u is not None:
            mu = mu + u.squeeze().sum(dim=0)
        E_ext = -(e_ext * mu).sum()
        if q_induced is not None:
            dip_induced = (q_induced[:,None] * r_raw).sum(dim=0)
            E_ext = E_ext - (e_ext * dip_induced).sum()
            mu = mu + dip_induced
        if u_induced is not None:
            mu_induced = u_induced.squeeze().sum(dim=0)
            E_ext = E_ext - 0.5*(e_ext * mu_induced).sum()
            mu = mu + mu_induced
        return E_ext, mu

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

        #Compute chi, alpha
        out = self.tensor_feed_forward(features)
        latent_charges = out[0][:,0] #[N,1]
        latent_dipoles = out[1][:,0] #[N,1,3]
        latent_kappas = out[1][:,1:] #[N,n_scf+1,3]
        latent_alphas = out[2][:,0:] #[N,n_scf+1,3,3]

        # print(latent_charges.shape,latent_kappas.shape,latent_alphas.shape)

        #Ewald requires charge dummy index
        result = self.les(
            positions=data['positions'],
            cell=data['cell'].view(-1, 3, 3),
            e_ext = e_ext,
            latent_charges = latent_charges[:,None],
            latent_dipoles = latent_dipoles[:,None,:],
            latent_kappas = latent_kappas[:,None],
            latent_alphas = latent_alphas[:,None],
            atomic_numbers = data[self.atomic_numbers_key],
            batch=data["batch"],
            compute_energy=self.compute_energy,
            compute_bec=self.compute_bec,
            bec_output_index=self.bec_output_index,
        )

        #Latent charges/dipoles include induced
        data["latent_kappas"] = latent_kappas
        data["latent_alphas"] = latent_alphas
        
        if self.compute_energy:
            data[self.energy_key] = result['E_lr']
            data["latent_charges"] = result["latent_charges"]
            data["latent_dipoles"] = result["latent_dipoles"]
            data["q_induced"] = result["q_induced"]
            data["u_induced"] = result["u_induced"]

        if self.compute_dipole:
            unique_batches = torch.unique(data["batch"])
            E_ext_list = []
            mu_list = []
            alpha_list = []
            #Calculate coupling to external field
            for i in unique_batches.long():
                mask = data["batch"] == i  # Create a mask for the i-th configurations
                r_now, q_now = data["positions"][mask], latent_charges[mask].squeeze()
                u_now = latent_dipoles[mask].squeeze()
                u_induced_now = data["u_induced"][mask].squeeze()
                q_induced_now = data["q_induced"][mask].squeeze()
                a_now = data["latent_alphas"][mask].squeeze()
                E_ext, mu = self.calc_E_ext(r_now,q_now,e_ext,u=u_now,u_induced=u_induced_now,q_induced=q_induced_now)
                E_ext_list.append(E_ext)
                mu_list.append(mu)
                alpha_list.append(a_now.sum() * torch.eye(3,device=r_now.device))
            E_ext = torch.hstack(E_ext_list)
            mu = torch.vstack(mu_list)
            alpha = torch.stack(alpha_list)

            if self.via_energy_derivatives:
                dipole = []
                for i in range(E_ext.shape[0]):
                    grad_i = torch.autograd.grad(
                        outputs=E_ext[i],
                        inputs=e_ext,
                        retain_graph=self.compute_polarizability, # Make sure the graph is not destroyed during training
                        create_graph=self.compute_polarizability, # Create graph for second derivative
                        allow_unused=False, # For complete dissociation turn to true
                    )[0] # shape [3]
                    dipole.append(grad_i)
                dipole = -torch.stack(dipole, dim=0)
            else:
                dipole = mu
        else:
            dipole = None

        if self.compute_polarizability:
            if self.via_energy_derivatives:
                #This will contain contribution from q_ind
                polarizability_rows = []
                for i in range(dipole.shape[0]):  # 4
                    row_i = []
                    for a in range(dipole.shape[1]):  # 3
                        grad_ia = torch.autograd.grad(
                            outputs=dipole[i, a],
                            inputs=e_ext,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=False,
                        )[0]  # [3]
                        row_i.append(grad_ia)
                    row_i = torch.stack(row_i, dim=0)   # [3, 3]
                    polarizability_rows.append(row_i)
                polarizability = torch.stack(polarizability_rows, dim=0)  # [4, 3, 3]
            else:
                polarizability = alpha
        else:
            polarizability = None

        if self.compute_energy:
            data["LES_energy"] = result['E_lr']
        if self.compute_bec:
            data["LES_BEC"] = result['BEC']
        if self.compute_dipole:
            data["LES_dipole"] = dipole
        if self.compute_polarizability:
            data["LES_polarizability"] = polarizability
        return data



#Shouldn't we go straight to ewald?
#We don't really need les, yeah...
#Just call ewald directly in LesVector...
#And add charge equilibration












































from cace.modules.tensornet import TensorFeedForward

class LesVectorWrapper(nn.Module):
    """
    A wrapper for the LES library that does long-range interactions and BECs
    Note that CACE has its own internal implementation of the LES algorithm
    so it is not necessary to use this wrapper in CACE.
    """
    def __init__(self,
                 feature_key: Union[str, Sequence[int]] = 'node_feats_l',
                 n_scf: int = 2,
                 energy_key: str = 'LES_energy',
                 charge_key: str = 'LES_charge',
                 chi_key: str = 'LES_chi', #vector susceptibility
                 alpha_key: str = 'LES_alpha', #matrix polarizability
                 e_ext_key: str = 'e_ext', #External field
                 bec_key: str = 'LES_BEC',
                 atomic_numbers_key: str = 'atomic_numbers',
                 compute_energy: bool = True,
                 compute_dipole: bool = True,
                 compute_polarizability: bool = True,
                 compute_bec: bool = False,
                 bec_output_index: int = None, # option to compute BEC along one axis
                 ):
        super().__init__()
        from les.les import LesVector
        self.les = Les()
 
        self.feature_key = feature_key
        self.n_scf = n_scf
        self.energy_key = energy_key
        self.charge_key = charge_key
        self.chi_key = chi_key
        self.alpha_key = alpha_key
        self.e_ext_key = e_ext_key
        self.atomic_numbers_key = atomic_numbers_key
        self.bec_key = bec_key
        self.bec_output_index = bec_output_index

        self.compute_energy = compute_energy
        self.compute_dipole = compute_dipole
        self.compute_polarizability = compute_polarizability
        if self.compute_dipole or self.compute_polarizability:
            self.compute_energy = True
        if self. compute_polarizability:
            self.compute_dipole = True
        self.compute_bec = compute_bec
        self.model_outputs = [charge_key]
        if compute_energy:
            self.model_outputs.append(energy_key)
        if compute_bec:
            self.model_outputs.append(bec_key)
        self.required_derivatives = []
        self.required_derivatives.append('cell')

        self.tensor_feed_forward = TensorFeedForward(2+n_scf,lomax=2)

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
            features = data[self.feature_key] #{0: l=0, 1:l=1, 2:l=2...}
        elif isinstance(self.feature_key, list):
            features = torch.cat([data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1)

        if data[self.e_ext_key] is not None:
            e_ext = data[self.e_ext_key]
        else:
            e_ext = torch.zeros_like(data["positions"][0])
        e_ext.requires_grad = True

        #Compute chi, alpha
        out = self.tensor_feed_forward(features)
        latent_charges = out[0][:,0] #[N,1]
        latent_dipoles = out[1][:,0] #[N,1,3]
        # latent_kappas = out[1][:,1:] #[N,n_scf+1,3]
        latent_alphas = out[2][:,0:] #[N,n_scf+1,3,3]

        #Enforce PSD via AA^T
        latent_alphas = torch.einsum("ncij,nckj->ncik",latent_alphas,latent_alphas)

        for i in range(n_scf+1):
            ewald_out = self.ewald(q=latent_charges[:,None],
                                r=positions,
                                e_ext=e_ext,
                                cell=cell,
                                batch=batch,
                                u=latent_dipoles[:,None,:],
                                compute_field=True,
                                )
            efield = ewald_out["field"].squeeze() + e_ext[None,:]
            q_induced = torch.einsum("ni,ni->n",latent_kappas[:,i,:],efield)
            mu_induced = torch.einsum("nij,nj->ni",latent_alphas[:,i,:,:],efield)
            latent_charges = latent_charges + q_induced
            latent_dipoles = latent_dipoles + mu_induced


        result = self.les(
            positions=data['positions'],
            cell=data['cell'].view(-1, 3, 3),
            latent_charges = latent_charges,
            e_ext = e_ext,
            n_scf=self.n_scf,
            latent_dipoles = latent_dipoles,
            latent_kappas = latent_kappas,
            latent_alphas = latent_alphas,
            atomic_numbers = data[self.atomic_numbers_key],
            batch=data["batch"],
            compute_energy=self.compute_energy,
            compute_bec=self.compute_bec,
            bec_output_index=self.bec_output_index,
        )

        data["latent_charges"] = latent_charges
        data["latent_dipoles"] = latent_dipoles
        data["latent_kappas"] = latent_kappas
        data["latent_alphas"] = latent_alphas

        if self.compute_dipole:
            unique_batches = torch.unique(data["batch"])
            E_ext_list = []
            mu_list = []
            alpha_list = []
            #Calculate coupling to external field
            for i in unique_batches.long():
                mask = data["batch"] == i  # Create a mask for the i-th configurations
                r_now, q_now = data["positions"][mask], latent_charges[mask].squeeze()
                u_now = latent_dipoles[mask].squeeze()
                u_induced_now = data["u_induced"][mask].squeeze()
                q_induced_now = data["q_induced"][mask].squeeze()
                a_now = data["latent_alphas"][mask].squeeze()
                E_ext, mu = self.calc_E_ext(r_now,q_now,e_ext,u=u_now,u_induced=u_induced_now,q_induced=q_induced_now)
                E_ext_list.append(E_ext)
                mu_list.append(mu)
                alpha_list.append(a_now.sum() * torch.eye(3,device=r_now.device))
            E_ext = torch.hstack(E_ext_list)
            mu = torch.vstack(mu_list)
            alpha = torch.stack(alpha_list)

            if self.via_energy_derivatives:
                dipole = []
                for i in range(E_ext.shape[0]):
                    grad_i = torch.autograd.grad(
                        outputs=E_ext[i],
                        inputs=e_ext,
                        retain_graph=self.compute_polarizability, # Make sure the graph is not destroyed during training
                        create_graph=self.compute_polarizability, # Create graph for second derivative
                        allow_unused=False, # For complete dissociation turn to true
                    )[0] # shape [3]
                    dipole.append(grad_i)
                dipole = -torch.stack(dipole, dim=0)
            else:
                dipole = mu
        else:
            dipole = None

        if self.compute_polarizability:
            if self.via_energy_derivatives:
                #This will contain contribution from q_ind
                polarizability_rows = []
                for i in range(dipole.shape[0]):  # 4
                    row_i = []
                    for a in range(dipole.shape[1]):  # 3
                        grad_ia = torch.autograd.grad(
                            outputs=dipole[i, a],
                            inputs=e_ext,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=False,
                        )[0]  # [3]
                        row_i.append(grad_ia)
                    row_i = torch.stack(row_i, dim=0)   # [3, 3]
                    polarizability_rows.append(row_i)
                polarizability = torch.stack(polarizability_rows, dim=0)  # [4, 3, 3]
            else:
                polarizability = alpha
        else:
            polarizability = None

        # output = {
        #          'E_lr': result["E_lr"],
        #          'latent_charges':latent_charges,
        #          'latent_dipoles':latent_dipoles,
        #          'latent_kappas':latent_kappas,
        #          'latent_alphas':latent_alphas,
        #          'dipole': dipole,
        #          'polarizability': polarizability,
        #          'BEC':bec,
        #          }

        # data[self.charge_key] = result['latent_charges']
        if self.compute_energy:
            data[self.energy_key] = result['E_lr']
        if self.compute_bec:
            data[self.bec_key] = result['BEC']
        if self.compute_dipole:
            data["LES_dipole"] = dipole
        if self.compute_polarizability:
            data["LES_polarizability"] = polarizability
        return data
