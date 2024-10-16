import os
import torch
import torch.nn as nn
import lightning as L
from . import GetLoss
from ..tools import Metrics
from typing import Dict, Optional, List, Tuple
from lightning.pytorch.callbacks import LearningRateMonitor

__all__ = ["LightningModel","LightningTrainingTask","LightningData"]

def default_losses(e_weight=0.1,f_weight=1000,e_name="energy",f_name="force"):
    e_loss = GetLoss(
                target_name=e_name,
                predict_name='pred_energy',
                loss_fn=torch.nn.MSELoss(),
                loss_weight=e_weight,
                )
    f_loss = GetLoss(
                target_name=f_name,
                predict_name='pred_force',
                loss_fn=torch.nn.MSELoss(),
                loss_weight=f_weight,
            )
    return [e_loss,f_loss]

def default_metrics(e_name="energy",f_name="force"):
    e_metric = Metrics(
                target_name=e_name,
                predict_name='pred_energy',
                name='e',
                metric_keys=["rmse"],
                per_atom=True,
            )
    f_metric = Metrics(
                target_name=f_name,
                predict_name='pred_force',
                metric_keys=["rmse"],
                name='f',
            )
    return [e_metric,f_metric]

#Lightning class for wrapping around nn.Modules
class LightningModel(L.LightningModule):
    def __init__(self,
                 model : nn.Module,
                 losses : List[nn.Module] = None,
                 metrics : List[nn.Module] = None,
                 log_rmse : bool = True,
                 optimizer_args = {'lr': 1e-2},
                 train_args = {"training":True},
                 val_args = {"training":False},
                 lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
                 scheduler_args = {'mode': 'min', 'factor': 0.8, 'patience': 10},
                ):
        super().__init__()
        self.model = model
        self.losses = losses
        self.metrics = metrics
        self.log_rmse = log_rmse
        self.optimizer_args = optimizer_args
        self.train_args = train_args
        self.val_args = val_args
        self.lr_scheduler = lr_scheduler
        self.scheduler_args = scheduler_args

    def forward(self,
                data: Dict[str, torch.Tensor],
                kwargs = None, #dict
               ) -> Dict[str, torch.Tensor]:
        if kwargs is None:
            kwargs = self.val_args
        return self.model.forward(data,**kwargs)

    def calculate_loss(self, 
            data: Dict[str, torch.Tensor], 
            ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        
        #Forward -- get a dictionary of predicted tensors
        results = self.forward(data,self.train_args)
        
        #Calculate loss
        tot_loss = 0
        for i, loss_fn in enumerate(self.losses):
            loss = loss_fn(results,data)
            tot_loss = tot_loss + loss
        return tot_loss, results
        
    def calculate_metrics(self, 
                data: Dict[str, torch.Tensor],
                results : Dict[str, torch.Tensor],
                ) -> Tuple[Dict[int, torch.Tensor], str]:
        dct = {}
        typ = "rmse"
        if not self.log_rmse:
            typ = "mae"
        for metric in self.metrics:
            name = metric.name
            dct[name] = metric(results,data)[typ]
        return dct, typ

    def training_step(self,
                      data : Dict[int, torch.Tensor],
                      batch_idx : int,
                      log_metrics = True,
                     ) -> torch.Tensor:
        loss, results = self.calculate_loss(data)

        #Calc metrics
        if log_metrics:
            batch_size = data.batch.max() + 1
            dct, typ = self.calculate_metrics(data,results)
            for k,v in dct.items():
                self.log(f"train_{k}_{typ}",v,batch_size=batch_size)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_args)
        lr_scheduler = self.lr_scheduler(optimizer,**self.scheduler_args)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
        }
        opt_info = {"optimizer": optimizer,"lr_scheduler": lr_scheduler_config}
        return opt_info

    def validation_step(self,
                      data : Dict[int, torch.Tensor],
                      val_idx : int) -> None:
        batch_size = data.batch.max() + 1

        #Get data
        with torch.enable_grad(): #for forces
            results = self.forward(data,self.val_args)

        #Log loss for lr scheduler
        tot_loss = 0
        for i, loss_fn in enumerate(self.losses):
            loss = loss_fn(results,data)
            tot_loss = tot_loss + loss
        self.log(f"val_loss",tot_loss,batch_size=batch_size)

        #Log metrics
        dct, typ = self.calculate_metrics(data,results)
        for k,v in dct.items():
            self.log(f"val_{k}_{typ}",v,batch_size=batch_size)

class LightningTrainingTask():
    def __init__(self,
                 model : nn.Module,
                 losses : List[nn.Module] = None,
                 metrics : List[nn.Module] = None,
                 log_rmse : bool = True,
                 optimizer_args = {'lr': 1e-2},
                 train_args = {"training":True},
                 val_args = {"training":False},
                 lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
                 scheduler_args = {'mode': 'min', 'factor': 0.8, 'patience': 10},
                ) -> None:
        self.model = LightningModel(model,
                                    losses = losses,
                                    metrics = metrics,
                                    log_rmse = log_rmse,
                                    optimizer_args = optimizer_args,
                                    train_args = train_args,
                                    val_args = val_args,
                                    lr_scheduler = lr_scheduler,
                                    scheduler_args = scheduler_args
        )

    def fit(self,data,chkpt=None,dev_run=False,max_epochs=None,max_steps=None,gradient_clip_val=10,accelerator="auto",name=None,progress_bar=True):
        from lightning.pytorch.loggers import TensorBoardLogger
        logger = TensorBoardLogger("lightning_logs",name=name)
        if (max_steps is None) and (max_epochs is None):
            print("Please input max_steps or max_epochs")
            return None
        if (max_steps is not None) and (max_epochs is not None):
            print("Please specify either max_steps or max_epochs but not both")
            return None
        if chkpt is not None:
            self.load(chkpt)
        lr_monitor = LearningRateMonitor(logging_interval='step') #to log the lr
        if max_epochs:
            trainer = L.Trainer(fast_dev_run=dev_run,max_epochs=max_epochs,enable_progress_bar=progress_bar,
                                gradient_clip_val=gradient_clip_val,callbacks=[lr_monitor],logger=logger,accelerator=accelerator)
        elif max_steps:
            trainer = L.Trainer(fast_dev_run=dev_run,max_steps=max_steps,enable_progress_bar=progress_bar,
                                gradient_clip_val=gradient_clip_val,callbacks=[lr_monitor],logger=logger,accelerator=accelerator)
        trainer.fit(self.model,data,ckpt_path=chkpt)

    def save(self,path):
        print("Saving model to",path,"...")
        state_dict = self.model.state_dict()
        torch.save({"state_dict":state_dict}, path)

    def load(self,path):
        print("Loading model from",path,"...")
        state_dict = torch.load(path, weights_only=True)
        self.epoch = state_dict["epoch"]
        self.global_step = state_dict["global_step"]
        self.model.load_state_dict(state_dict["state_dict"])
        print("Loading successful!")

#Data
from ..tools import torch_geometric
from ..data import AtomicData
from . import get_dataset_from_xyz, load_data_loader

class LightningData(L.LightningDataModule):
    def __init__(self, root,
                 cutoff=5.5,
                 batch_size=4,
                 data_key = {"energy":"energy","force":"force"},
                 atomic_energies=None,
                 valid_p=0.1,
                 seed=1,
                ):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.valid_p = valid_p
        self.cutoff = cutoff
        self.seed = seed
        self.atomic_energies = atomic_energies
        self.data_key = data_key
        self.prepare_data()
    
    def prepare_data(self):
        collection = get_dataset_from_xyz(train_path=self.root,
                                          valid_fraction=self.valid_p,
                                          seed=self.seed,
                                          cutoff=self.cutoff,
                                          data_key=self.data_key,
                                          atomic_energies = self.atomic_energies
                                         )
        
        self.train_dataset = [
                AtomicData.from_atoms(atoms, cutoff=self.cutoff, data_key=self.data_key, atomic_energies=self.atomic_energies)
                for atoms in collection.train
            ]
        self.valid_dataset = [
                AtomicData.from_atoms(atoms, cutoff=self.cutoff, data_key=self.data_key, atomic_energies=self.atomic_energies)
                for atoms in collection.valid
            ]

        self.train_loader = torch_geometric.DataLoader(
            dataset = self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=os.cpu_count()-1,
        )
        
        self.valid_loader = torch_geometric.DataLoader(
            dataset = self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count()-1,
        )
        
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.valid_loader