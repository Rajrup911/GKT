import torch
import pytorch_lightning as pl
import onnx
import onnxsim
import os
import numpy as np
from omegaconf import DictConfig
import onnxruntime
from einops import rearrange, repeat
from onnxruntime.tools import pytorch_export_contrib_ops

class ModelModule(pl.LightningModule):
    def __init__(self, backbone, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None):
        super().__init__()

        self.save_hyperparameters(
            cfg,
            ignore=['backbone', 'loss_func', 'metrics', 'optimizer_args', 'scheduler_args'])
        
        self.cfg = DictConfig(cfg)
        self.backbone = backbone
        self.loss_func = loss_func
        self.metrics = metrics

        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def forward(self, image, intrinsics=None, extrinsics=None, export=False, fixed_IE=False):
        if self.cfg.experiment.export:
            #For Exporting the Model to Onnx

            if not os.path.exists(self.cfg.experiment.onnx_path):
                os.makedirs(self.cfg.experiment.onnx_path)
            
            pytorch_export_contrib_ops.register()
            if fixed_IE:
                #Flag that checks if Intrinsics/Extrinsics are fixed during onnx export
                #Exports only with Image
                torch.onnx.export(
                self.backbone.cpu(),
                (image.cpu(), export, fixed_IE),
                self.cfg.experiment.onnx_path + self.cfg.experiment.onnx_name + "-No_IE" + ".onnx",
                input_names = ['image'],
                output_names = ['logits'],
                verbose = True,
                export_params = True,
                opset_version = 16
                )

                print("Onnx Export Finished!!, \nStarting Onnx Simplification")
                onnx_model = onnx.load(self.cfg.experiment.onnx_path + self.cfg.experiment.onnx_name + "-No_IE" + ".onnx")
                onnx_model, check = onnxsim.simplify(onnx_model)
                onnx.save(onnx_model, self.cfg.experiment.onnx_path + self.cfg.experiment.onnx_name + "-No_IE" + "-SIM.onnx")
                print("Onnx Simplification Done!!")

                exit()
            else:
                #Export to onnx with Image, Intrinsincs, Extrinsics
                torch.onnx.export(
                self.backbone.cpu(),
                (image.cpu(), export, fixed_IE, intrinsics.cpu(), extrinsics.cpu()),
                self.cfg.experiment.onnx_path + self.cfg.experiment.onnx_name + ".onnx",
                input_names = ['image', 'intrinsics', 'extrinsics'],
                output_names = ['logits'],
                verbose = True,
                export_params = True,
                opset_version = 16
                )

                print("Onnx Export Finished!!, \nStarting Onnx Simplification")
                onnx_model = onnx.load(self.cfg.experiment.onnx_path + self.cfg.experiment.onnx_name + ".onnx")
                onnx_model, check = onnxsim.simplify(onnx_model)
                onnx.save(onnx_model, self.cfg.experiment.onnx_path + self.cfg.experiment.onnx_name + "-SIM.onnx")
                print("Onnx Simplification Done!!")

                exit()
        else:
            return self.backbone(image=image, intrinsics=intrinsics, extrinsics=extrinsics, export=export, fixed_IE=fixed_IE)

    def shared_step(self, batch, cfg, prefix='', on_step=False, return_output=True):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']

        image = rearrange(image, 'b n ... -> (b n) ...')
        #np.save(self.cfg.experiment.onnx_path + "intrinsics.npy", intrinsics.cpu().numpy())
        #np.save(self.cfg.experiment.onnx_path + "extrinsics.npy", extrinsics.cpu().numpy())

        if self.cfg.experiment.onnxruntime:
            if self.cfg.experiment.fixed_IE:
                EP_list = ['CUDAExecutionProvider']
                sess = onnxruntime.InferenceSession(self.cfg.experiment.onnx_path + self.cfg.experiment.onnx_name + "-No_IE" + "-SIM.onnx", providers=EP_list)

                inputs = {sess.get_inputs()[0].name: image.cpu().numpy()}
                output = sess.run(None, inputs)
                z = torch.Tensor(output[0])
            else:
                EP_list = ['CUDAExecutionProvider']
                sess = onnxruntime.InferenceSession(self.cfg.experiment.onnx_path + self.cfg.experiment.onnx_name + "-SIM.onnx", providers=EP_list)

                inputs = {sess.get_inputs()[0].name: image.cpu().numpy(), sess.get_inputs()[1].name: intrinsics.cpu().numpy(), sess.get_inputs()[2].name: extrinsics.cpu().numpy()}
                output = sess.run(None, inputs)
                z = torch.Tensor(output[0])

        else:
            z = self(image, intrinsics, extrinsics, export=self.cfg.experiment.export, fixed_IE=self.cfg.experiment.fixed_IE)

        outputs = {'bev': [0, 1], 'center': [1, 2]}
        pred = {k: z[:, start:stop].cuda() for k, (start, stop) in outputs.items()}

        loss, loss_details = self.loss_func(pred, batch)
        self.metrics.update(pred, batch)

        #if self.trainer is not None:
            #self.log(f'{prefix}/loss', loss.detach(), on_step=on_step, on_epoch=True)
            #self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()},
            #              on_step=on_step, on_epoch=True)

        # Used for visualizations
        if return_output:
            return {'loss': loss, 'batch': batch, 'pred': pred}

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train', True,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val', False,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    def on_validation_start(self) -> None:
        self._log_epoch_metrics('train')
        self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    def validation_epoch_end(self, outputs):
        self._log_epoch_metrics('val')

    def _log_epoch_metrics(self, prefix: str):
        """
        lightning is a little odd - it goes

        on_train_start
        ... does all the training steps ...
        on_validation_start
        ... does all the validation steps ...
        on_validation_epoch_end
        on_train_epoch_end
        """
        metrics = self.metrics.compute()

        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    self.log(f'{prefix}/metrics/{key}{subkey}', val)
            else:
                self.log(f'{prefix}/metrics/{key}', value)

        self.metrics.reset()

    def _enable_dataloader_shuffle(self, dataloaders):
        """
        HACK for https://github.com/PyTorchLightning/pytorch-lightning/issues/11054
        """
        for v in dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self, disable_scheduler=False):
        parameters = [x for x in self.backbone.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)

        if disable_scheduler or self.scheduler_args is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_args)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
