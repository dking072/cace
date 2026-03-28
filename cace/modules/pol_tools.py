import torch

def calc_E_ext(r_raw,q,e_ext,cell=None,u=None,kappa=None,alpha=None):
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

    mu_u = torch.zeros_like(e_ext)
    if u is not None:
        mu_u = mu_u + u.sum(dim=0)
        if cell is None:
            mu = mu + u.sum(dim=0)
        
    polarizability = 0 * torch.eye(3,device=r_raw.device)
    if kappa is not None:
        phi_ext = -(e_ext[None,:]*r_raw).sum(dim=1)
        q_ext_induced = -phi_ext*kappa
        q_ext_induced = q_ext_induced - q_ext_induced.mean()
        dip_induced = (r_raw * q_ext_induced[:,None]).sum(dim=0)
        mu = mu + dip_induced
        #Polarizability w/o derivative:
        polarizability = (kappa[:,None,None] * r_raw[:,:,None] * r_raw[:,None,:]).sum(dim=0)
        kappa_rij = kappa[:,None,None,None] * r_raw[:,None,:,None] * r_raw[None,:,None,:]
        polarizability = polarizability - 1/r_raw.shape[0]*kappa_rij.sum(dim=0).sum(dim=0)

    if alpha is not None:
        polarizability = polarizability + torch.eye(3,device=r_raw.device) * alpha.sum()
        u_ext_induced = e_ext[None,:] * alpha[:,None]
        mu_u = mu_u + u_ext_induced.sum(dim=0)
        if cell is None:
            mu = mu + u_ext_induced.sum(dim=0)

    E_ext = -(e_ext * mu).sum()
    E_ext_u = -(e_ext * mu_u).sum()
    return E_ext, mu, mu_u, polarizability.real, phase, E_ext_u

def dipole_from_e_ext_deriv(E_ext,e_ext,E_ext_u=None,latent_dipoles=None):
    n_out = E_ext.shape[0]
    eye = torch.eye(n_out, device=E_ext.device, dtype=e_ext.dtype)

    dipole = -torch.autograd.grad(
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
        dipole = dipole + 1j * dipole_imag

        if latent_dipoles is not None:
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
            dipole_u = None
    else:
        dipole_u = None

    return dipole, dipole_u

def polarizability_from_e_ext_deriv(dipole,e_ext):
    n_out, dim = dipole.shape
    n_total = n_out * dim
    dipole_flat = dipole.reshape(-1)

    # grad_outputs must match the dtype of the output being differentiated
    grad_outputs = torch.eye(
        n_total,
        device=dipole.device,
        dtype=e_ext.dtype,   # real dtype, since we're differentiating real-valued views
    )

    # d(Re dipole)/d e_ext
    polarizability_real_flat = torch.autograd.grad(
        outputs=dipole_flat.real,
        inputs=e_ext,
        grad_outputs=grad_outputs,
        retain_graph=True,  # keep graph if we still need imag pass
        create_graph=False,
        allow_unused=True,
        is_grads_batched=True,
    )[0]

    #We'd take real, so just ignore imaginary:
    # if dipole_flat.is_complex():
    #     # d(Im dipole)/d e_ext
    #     polarizability_imag_flat = torch.autograd.grad(
    #         outputs=dipole_flat.imag,
    #         inputs=e_ext,
    #         grad_outputs=grad_outputs,
    #         retain_graph=True,
    #         create_graph=False,
    #         allow_unused=False,
    #         is_grads_batched=True,
    #     )[0]

    #     polarizability_flat = polarizability_real_flat + 1j * polarizability_imag_flat
    # else:
    #     polarizability_flat = polarizability_real_flat

    if polarizability_real_flat is None:
        polarizability = None
    else:
        polarizability = 0.5 * polarizability_real_flat.view(n_out, dim, dim)
    return polarizability
        
    