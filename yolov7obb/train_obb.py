import argparse
import logging
from functools import partial
from typing import Optional, Dict

import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from yolov7obb.models.experimental import attempt_load
from yolov7obb.models.imodel import IModel
from yolov7obb.models.yolo import Model
from yolov7obb.parameters import DEFAULT_GRID_SIZE
from yolov7obb.utils.autoanchor import check_anchors
from yolov7obb.utils.datasets import create_dataloader
from yolov7obb.utils.general import labels_to_class_weights, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, check_img_size, \
    print_mutation, colorstr, check_file, increment_path, set_logging, get_latest_run
from yolov7obb.utils.google_utils import attempt_download
from yolov7obb.utils.idatasets import IDataloader, IDataset
from yolov7obb.utils.loss import ComputeLossOBB, ComputeLossOTA, IComputeLoss
from yolov7obb.utils.plots import plot_images, plot_results, plot_evolution
from yolov7obb.utils.scheduler import LRScheduler, ILRScheduler
from yolov7obb.utils.torch_utils import ModelEMA, intersect_dicts, torch_distributed_zero_first, \
    is_parallel, select_device
from yolov7obb.utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

import torch_pruning as tp

logger = logging.getLogger(__name__)

# import warnings
# warnings.filterwarnings("error")


def train(
        hyperparameter,
        input_arguments,
        target_device,
        tensorboard_writer=None
):
    logger.info(
        colorstr('hyperparameters: ') +
        ', '.join(f'{param_name}={param_value}' for param_name, param_value in hyperparameter.items())
    )

    save_dir = Path(input_arguments.save_dir)
    epochs = input_arguments.epochs
    global_batch_size = input_arguments.global_batch_size
    local_batch_size = input_arguments.global_batch_size // input_arguments.world_size
    gpu_batch_size = input_arguments.gpu_batch_size
    weights = input_arguments.weights
    rank = input_arguments.global_rank
    freeze = input_arguments.freeze
    plots_debug = input_arguments.plots_debug
    single_class = input_arguments.single_cls
    sparsity = input_arguments.sparsity
    prune_train = input_arguments.prune_train

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as file_pointer:
        yaml.dump(hyperparameter, file_pointer, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as file_pointer:
        yaml.dump(vars(input_arguments), file_pointer, sort_keys=False)

    # Configure
    plots = not input_arguments.evolve  # create plotsnp.int

    cuda = target_device.type != 'cpu'
    init_seeds(2 + rank)
    with open(input_arguments.data) as file_pointer:
        data_dict = yaml.load(file_pointer, Loader=yaml.SafeLoader)  # data dict
    is_coco = input_arguments.data.name == 'coco.yaml'

    # Logging Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    wandb_logger = None
    if rank in [-1, 0]:
        input_arguments.hyp = hyperparameter  # add hyperparameter
        run_id = None
        if weights.endswith('.pt') and os.path.isfile(weights):
            run_id = torch.load(weights, map_location=target_device).get('wandb_id')
        wandb_logger = WandbLogger(input_arguments, Path(input_arguments.save_dir).stem, run_id, data_dict)
        if wandb_logger.wandb is None:
            wandb_logger = None
            loggers['wandb'] = None
        else:
            loggers['wandb'] = wandb_logger.wandb
            data_dict = wandb_logger.data_dict
        if wandb_logger:
            # WandbLogger might update weights, epochs if resuming
            weights = input_arguments.weights
            epochs = input_arguments.epochs
            hyperparameter = input_arguments.hyp

    # TODO: get this information from the training dataset
    classes_number = 1 if input_arguments.single_cls else int(data_dict['nc'])  # number of classes
    # class names
    names = ['item'] if input_arguments.single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == classes_number, '%g names found for nc=%g dataset in %s' %\
                                         (len(names), classes_number, input_arguments.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained and not prune_train:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        checkpoint_state = torch.load(weights, map_location=target_device)  # load checkpoint
        # create model
        model = Model(
            input_arguments.cfg or checkpoint_state['model'].yaml,
            ch=3,
            nc=classes_number,
            anchors=hyperparameter.get('anchors')
        ).to(target_device)
        exclude = []
        # exclude keys
        if (input_arguments.cfg or hyperparameter.get('anchors')) and not input_arguments.resume:
            exclude = ['anchor']
        state_dict = checkpoint_state['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
        del state_dict
    elif pretrained and prune_train:
        checkpoint_state = torch.load(weights, map_location=target_device)  # load checkpoint
        model = checkpoint_state["model"]
        model = model.to(target_device)
    else:
        # create scratch model
        model = Model(input_arguments.cfg, ch=3, nc=classes_number, anchors=hyperparameter.get('anchors')).to(target_device)
        checkpoint_state = {}
    print("Model layers:")
    print(model)
    # TODO: Check why it is commented out and not in the original version
    # Gwang add
    # with torch_distributed_zero_first(rank):
    #     check_dataset(data_dict)  # check
    train_path = Path(data_dict['train'])
    test_path = Path(data_dict['val'])

    # Freeze. Parameter names to freeze (full or partial)
    freeze = [f'model.{layer_name}.' for layer_name in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for param_name, param_value in model.named_parameters():
        param_value.requires_grad = True  # train all layers
        if any(layer_name in param_name for layer_name in freeze):
            print('freezing %s' % param_name)
            param_value.requires_grad = False

    # Optimizer
    # TODO: exclude this step from common block and move to trainer
    accumulate_step = max(round(local_batch_size / gpu_batch_size), 1)  # accumulate loss before optimizing
    hyperparameter['weight_decay'] *= global_batch_size * accumulate_step / local_batch_size  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyperparameter['weight_decay']}")

    pg0, pg1, pg2 = layer_grouping(model)

    if input_arguments.adam:
        # adjust beta1 to momentum
        optimizer = optim.Adam(pg0, lr=hyperparameter['lr0'], betas=(hyperparameter['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyperparameter['lr0'], momentum=hyperparameter['momentum'], nesterov=True)

    # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyperparameter['weight_decay']})
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    trainer = Trainer(
        rank=rank,
        global_batch_size=global_batch_size,
        gpu_batch_size=gpu_batch_size,
        accumulate_step=accumulate_step,
        work_directory=save_dir,
        single_class=single_class,
        wandb_logger=wandb_logger,
        tensorboard_writer=tensorboard_writer,
        sparsity=sparsity
    )

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if checkpoint_state['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint_state['optimizer'])
        if 'best_fitness' in checkpoint_state:
            trainer.load_state_dict(
                {
                    "best_fitness": checkpoint_state['best_fitness']
                }
            )

        # EMA
        if ema and checkpoint_state.get('ema'):
            ema.ema.load_state_dict(checkpoint_state['ema'].float().state_dict())
            ema.updates = checkpoint_state['updates']

        # Epochs
        start_epoch = checkpoint_state['epoch'] + 1
        if input_arguments.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, checkpoint_state['epoch'], epochs))
            epochs += checkpoint_state['epoch']  # finetune additional epochs

        del checkpoint_state

    # Image sizes
    grid_size = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # verify imgsz are gs-multiples
    image_size = check_img_size(tuple(input_arguments.img_size), grid_size)

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if input_arguments.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(target_device)
        logger.info('Using SyncBatchNorm()')

    # Create train dataloader
    train_dataloader = create_dataloader(
        path=train_path,
        image_size=image_size,
        batch_size=gpu_batch_size,
        grid_size=grid_size,
        opt=input_arguments,
        hyp=hyperparameter,
        augment=True,
        cache=input_arguments.cache_images,
        rect=input_arguments.rect,
        rank=rank,
        world_size=input_arguments.world_size,
        workers=input_arguments.workers,
        image_weights=input_arguments.image_weights,
        quad=input_arguments.quad,
        prefix=colorstr('train: ')
    )
    max_class_label = np.concatenate(train_dataloader.dataset.annotations, 0)[:, 0].max()  # max label class
    batches_number = len(train_dataloader)  # number of batches

    warmup_iter_num = max(round(hyperparameter['warmup_epochs'] * batches_number), 1000)
    scheduler = LRScheduler(
        optimizer=optimizer,
        epochs=epochs,
        lr_final=hyperparameter['lrf'],
        warmup_iterations=warmup_iter_num,
        warmup_bias_lr=hyp['warmup_bias_lr'],
        warmup_momentum=hyp['warmup_momentum'],
        momentum=hyp['momentum']
    )

    assert max_class_label < classes_number, \
        'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' \
        % (max_class_label, classes_number, input_arguments.data, classes_number - 1)

    # Process 0
    test_dataloader = None
    if rank in [-1, 0]:
        # create test dataloader only in main process
        test_dataloader = create_dataloader(
            test_path,
            image_size,
            gpu_batch_size,
            grid_size,
            input_arguments,  #
            hyp=hyperparameter,
            cache=input_arguments.cache_images and not input_arguments.notest,
            rect=input_arguments.rect,
            rank=-1,
            world_size=input_arguments.world_size,
            workers=input_arguments.workers,
            pad=0.5,
            prefix=colorstr('val: ')
        )
        dataset = test_dataloader.dataset
        if not input_arguments.resume:
            np_annotations = np.concatenate(dataset.annotations, 0)
            all_classes_id = torch.tensor(np_annotations[:, 0])  # classes
            if plots:
                if tensorboard_writer:
                    tensorboard_writer.add_histogram('classes', all_classes_id, 0)

            # Anchors
            if not input_arguments.noautoanchor:
                check_anchors(dataset, model=model, thr=hyperparameter['anchor_t'], imgsz=image_size)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DistributedDataParallel(
            model,
            device_ids=[input_arguments.local_rank],
            output_device=input_arguments.local_rank,
            # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
            find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
        )

    # Model parameters
    hyperparameter['box'] *= 3. / nl  # scale to layers
    hyperparameter['cls'] *= classes_number / 80. * 3. / nl  # scale to classes and layers
    hyperparameter['obj'] *= (max(image_size) / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyperparameter['label_smoothing'] = input_arguments.label_smoothing
    model.nc = classes_number  # attach number of classes to model
    model.hyp = hyperparameter  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # attach class weights
    model.class_weights = \
        labels_to_class_weights(train_dataloader.dataset.annotations, classes_number).to(target_device) * classes_number
    model.names = names

    # Start training
    t0 = time.time()
    # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    # result_metrics = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    if hyperparameter["loss_ota"]:
        compute_loss = ComputeLossOTA(model)  # init loss class
    else:
        compute_loss = ComputeLossOBB(model=model, nc=classes_number)  # init loss class
    logger.info(f'Image sizes {image_size} train, {image_size} test\n'
                f'Using {train_dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    torch.save(model, trainer.weights_dir / 'init.pt')

    epoch, result_metrics = trainer.train_loop(
        model=model,
        batch_size=gpu_batch_size,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_function=compute_loss,
        data_dict=data_dict,
        ema=ema,
        image_size_test=image_size,
        is_coco=is_coco,
        total_epoch_batches=batches_number,
        optimizer=optimizer,
        plots=plots,
        plots_debug=plots_debug,
        scheduler=scheduler,
        epochs=epochs

    )
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger:
                files = [
                    'results.png',
                    'confusion_matrix.png',
                    *[f'{prefix_name}_curve.png' for prefix_name in ('F1', 'PR', 'P', 'R')]
                ]
                wandb_logger.log({
                    "Results": [
                        wandb_logger.wandb.Image(str(save_dir / image_file), caption=image_file)
                        for image_file in files if (save_dir / image_file).exists()
                    ]
                })
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if input_arguments.data.endswith('coco.yaml') and classes_number == 80:  # if COCO
            # speed, mAP tests
            for m in (trainer.last_checkpoint, trainer.best_checkpoint) \
                    if trainer.best_checkpoint.exists() else trainer.last_checkpoint:
                result_metrics, _, _ = test.test(
                    input_arguments.data,
                    batch_size=gpu_batch_size * 2,
                    imgsz=image_size,
                    conf_thres=0.001,
                    iou_thres=0.7,
                    model=attempt_load(m, target_device).half(),
                    single_cls=input_arguments.single_cls,
                    dataloader=test_dataloader,
                    work_directory=save_dir,
                    save_json=True,
                    plots=False,
                    is_coco=is_coco
                )

        # Strip optimizers
        final = trainer.best_checkpoint if trainer.best_checkpoint.exists() else trainer.last_checkpoint  # final model
        for file_pointer in trainer.last_checkpoint, trainer.best_checkpoint:
            if file_pointer.exists():
                strip_optimizer(file_pointer)  # strip optimizers
        if input_arguments.bucket:
            os.system(f'gsutil cp {final} gs://{input_arguments.bucket}/weights')  # upload
        if wandb_logger.wandb and not input_arguments.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return result_metrics


def layer_grouping(model):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for param_name, param_value in model.named_modules():
        if hasattr(param_value, 'bias') and isinstance(param_value.bias, nn.Parameter):
            pg2.append(param_value.bias)  # biases
        if isinstance(param_value, nn.BatchNorm2d):
            pg0.append(param_value.weight)  # no decay
        elif hasattr(param_value, 'weight') and isinstance(param_value.weight, nn.Parameter):
            pg1.append(param_value.weight)  # apply decay
        if hasattr(param_value, 'im'):
            if hasattr(param_value.im, 'implicit'):
                pg0.append(param_value.im.implicit)
            else:
                for iv in param_value.im:
                    pg0.append(iv.implicit)
        if hasattr(param_value, 'imc'):
            if hasattr(param_value.imc, 'implicit'):
                pg0.append(param_value.imc.implicit)
            else:
                for iv in param_value.imc:
                    pg0.append(iv.implicit)
        if hasattr(param_value, 'imb'):
            if hasattr(param_value.imb, 'implicit'):
                pg0.append(param_value.imb.implicit)
            else:
                for iv in param_value.imb:
                    pg0.append(iv.implicit)
        if hasattr(param_value, 'imo'):
            if hasattr(param_value.imo, 'implicit'):
                pg0.append(param_value.imo.implicit)
            else:
                for iv in param_value.imo:
                    pg0.append(iv.implicit)
        if hasattr(param_value, 'ia'):
            if hasattr(param_value.ia, 'implicit'):
                pg0.append(param_value.ia.implicit)
            else:
                for iv in param_value.ia:
                    pg0.append(iv.implicit)
        if hasattr(param_value, 'attn'):
            if hasattr(param_value.attn, 'logit_scale'):
                pg0.append(param_value.attn.logit_scale)
            if hasattr(param_value.attn, 'q_bias'):
                pg0.append(param_value.attn.q_bias)
            if hasattr(param_value.attn, 'v_bias'):
                pg0.append(param_value.attn.v_bias)
            if hasattr(param_value.attn, 'relative_position_bias_table'):
                pg0.append(param_value.attn.relative_position_bias_table)
        if hasattr(param_value, 'rbr_dense'):
            if hasattr(param_value.rbr_dense, 'weight_rbr_origin'):
                pg0.append(param_value.rbr_dense.weight_rbr_origin)
            if hasattr(param_value.rbr_dense, 'weight_rbr_avg_conv'):
                pg0.append(param_value.rbr_dense.weight_rbr_avg_conv)
            if hasattr(param_value.rbr_dense, 'weight_rbr_pfir_conv'):
                pg0.append(param_value.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(param_value.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                pg0.append(param_value.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(param_value.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                pg0.append(param_value.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(param_value.rbr_dense, 'weight_rbr_gconv_dw'):
                pg0.append(param_value.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(param_value.rbr_dense, 'weight_rbr_gconv_pw'):
                pg0.append(param_value.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(param_value.rbr_dense, 'vector'):
                pg0.append(param_value.rbr_dense.vector)
    return pg0, pg1, pg2


class Trainer:
    def __init__(
            self,
            rank: int,
            global_batch_size: int,
            gpu_batch_size: int,
            accumulate_step: int,
            work_directory: Path,
            single_class: bool,
            wandb_logger: Optional[WandbLogger] = None,
            tensorboard_writer: Optional[SummaryWriter] = None,
            sparsity: bool = False,
            sparsity_regularization: float = 5e-4,
            sparsity_global_pruning: bool = True
    ):
        self.rank = rank
        self.wandb_logger = wandb_logger
        self.tensorboard_writer = tensorboard_writer
        self.work_directory = work_directory
        self.plot_images_directory = work_directory / 'plot_img'
        self.plot_images_directory.mkdir(exist_ok=True)
        self.global_batch_size = global_batch_size
        self.gpu_batch_size = gpu_batch_size
        self.accumulate_step = accumulate_step
        self.single_class = single_class
        self.format_number = 6 if self.single_class else 7
        # Directories
        self.weights_dir = self.work_directory / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last_checkpoint = self.weights_dir / 'last.pt'
        self.best_checkpoint = self.weights_dir / 'best.pt'
        self.results_file = self.work_directory / 'results.txt'
        self.best_fitness = 0.0
        if self.single_class:
            self.results = (0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        else:
            self.results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

        self.sparsity = sparsity
        if self.sparsity:
            self.imp = tp.importance.BNScaleImportance()
            self.pruner_entry = partial(
                tp.pruner.BNScalePruner,
                reg=sparsity_regularization,
                global_pruning=sparsity_global_pruning
            )
        self.sparsity_regularization = sparsity_regularization

    def load_state_dict(self, state: Dict):
        self.best_fitness = state["best_fitness"]

    def train_loop(
            self,
            model: IModel,
            batch_size: int,
            train_dataloader: IDataloader,
            test_dataloader: IDataloader,
            loss_function: IComputeLoss,
            data_dict: Dict,
            ema: ModelEMA,
            image_size_test: int,
            is_coco: bool,
            total_epoch_batches: int,
            optimizer: Optimizer,
            plots: bool,
            plots_debug: bool,
            scheduler: ILRScheduler,
            epochs: int = 200,
            start_epoch: int = 1,
            update_image_weights: bool = False,
            multi_scale: bool = False,
            loss_ota: bool = False

    ):
        cuda = model.device != 'cpu'
        scaler = amp.GradScaler(enabled=cuda)
        dataset = train_dataloader.dataset
        grid_size = max(int(model.stride.max()), DEFAULT_GRID_SIZE)  # grid size (max stride)
        map_per_class = np.zeros(dataset.classes_number)  # mAP per class
        # epoch ------------------------------------------------------------------
        epoch = 0
        for epoch in range(start_epoch, epochs):
            model.train()
            # Update image weights (optional)
            if update_image_weights:
                self.calc_image_weights(dataset=dataset, map_per_class=map_per_class, model=model)

            # mean losses
            mean_losses = torch.zeros(4 if self.single_class else 5, device=model.device)

            if self.rank != -1:
                train_dataloader.sampler.set_epoch(epoch)

            pbar = enumerate(train_dataloader)
            if self.single_class:
                logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'theta',
                                                   'total', 'labels', 'img_size'))
            else:
                logger.info(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'theta',
                                                   'total', 'labels', 'img_size'))

            if self.rank in [-1, 0]:
                pbar = tqdm(pbar, total=total_epoch_batches)  # progress bar

            optimizer.zero_grad()

            # batch -------------------------------------------------------------
            for batch_number, (batch_images, targets, paths, _) in pbar:
                # number integrated batches (since train start)
                global_batch_number = batch_number + total_epoch_batches * (epoch-1)

                # TODO: move scaling to dataset preprocessing
                # uint8 to float32, 0-255 to 0.0-1.0
                batch_images = batch_images.to(device, non_blocking=True).float() / 255.0

                scheduler.warmup_step(global_batch_number, epoch)

                # Multi-scale
                if multi_scale:
                    batch_images = Trainer.rescale_batch_images(batch_images, grid_size)

                # Forward
                with amp.autocast(enabled=device.type != 'cpu'):
                    pred = model(batch_images)  # forward
                    if loss_ota:
                        # loss scaled by batch_size
                        loss, loss_items = loss_function(pred, targets.to(device), batch_images)
                    else:
                        # loss scaled by batch_size
                        loss, loss_items = loss_function(pred, targets.to(device))
                    if self.rank != -1:
                        loss *= opt.world_size  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                if self.sparsity:
                    # self.pruner.regularize(model)  # for sparsity learning
                    for m in model.model.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine == True:
                            m.weight.grad.data.add_(self.sparsity_regularization * torch.sign(m.weight.data))

                # Optimize
                if global_batch_number % self.accumulate_step == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Print
                if self.rank in [-1, 0]:
                    mean_losses = (mean_losses * batch_number + loss_items) / (batch_number + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    log_str = ('%10s' * 2 + '%10.4g' * self.format_number) % (
                        '%g/%g' % (epoch, epochs - 1), mem, *mean_losses, targets.shape[0], batch_images.shape[-1])
                    pbar.set_description(log_str)

                    # Plot
                    if plots and batch_number < 5:
                        fp = self.plot_images_directory / f'train_batch_{epoch}_{batch_number}.jpg'  # filename
                        Thread(target=plot_images, args=(batch_images, targets, paths, fp), daemon=True).start()
                    elif plots and global_batch_number == 10 and self.wandb_logger:
                        self.wandb_logger.log(
                            {"Mosaics": [
                                self.wandb_logger.wandb.Image(str(image_path), caption=image_path.name)
                                for image_path in self.work_directory.glob('train*.jpg') if image_path.exists()
                            ]})

                # end batch ------------------------------------------------------------------------------------------
            # end epoch ----------------------------------------------------------------------------------------------

            # Scheduler
            lr = [param['lr'] for param in optimizer.param_groups]  # for tensorboard
            scheduler.step()

            # DDP process 0 or single-GPU
            if self.rank in [-1, 0]:
                # mAP
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
                final_epoch = epoch + 1 == epochs
                results = ()
                if not opt.notest or final_epoch:  # Calculate mAP
                    if self.wandb_logger:
                        self.wandb_logger.current_epoch = epoch + 1
                    results, maps, times = test.test(
                        data=data_dict,
                        batch_size=batch_size * 2,
                        imgsz=image_size_test,
                        model=ema.ema,
                        single_cls=self.single_class,
                        dataloader=test_dataloader,
                        work_directory=self.plot_images_directory,
                        verbose=test_dataloader.dataset.classes_number < 50 and final_epoch,
                        plots=plots and final_epoch,
                        wandb_logger=self.wandb_logger,
                        compute_loss=loss_function,
                        is_coco=is_coco,
                        epoch=epoch,
                        plots_debug=plots_debug
                    )

                # Write
                log_str = ''
                with open(self.results_file, 'a') as fp:
                    fp.write(log_str + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
                if len(opt.name) and opt.bucket:
                    os.system('gsutil cp %s gs://%s/results/results%s.txt' % (self.results_file, opt.bucket, opt.name))

                # Log
                if self.single_class:
                    tags = ['train/box_loss', 'train/obj_loss', 'train/theta_loss',  # train loss
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                            'val/box_loss', 'val/obj_loss',  # val loss
                            'x/lr0', 'x/lr1', 'x/lr2']  # params
                else:
                    tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/theta_loss',  # train loss
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                            'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                            'x/lr0', 'x/lr1', 'x/lr2']  # params
                for param, tag in zip(list(mean_losses[:-1]) + list(results) + lr, tags):
                    if tb_writer:
                        tb_writer.add_scalar(tag, param, epoch)  # tensorboard
                    if self.wandb_logger:
                        self.wandb_logger.log({tag: param})  # W&B

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > self.best_fitness:
                    self.best_fitness = fi
                if self.wandb_logger:
                    self.wandb_logger.end_epoch(best_result=self.best_fitness == fi)

                # Save model
                if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                    checkpoint_state = \
                        {
                            'epoch': epoch,
                            'best_fitness': self.best_fitness,
                            'training_results': self.results_file.read_text(),
                            'model': deepcopy(model.module if is_parallel(model) else model).half(),
                            'ema': deepcopy(ema.ema).half(),
                            'updates': ema.updates,
                            'optimizer': optimizer.state_dict(),
                            'wandb_id': self.wandb_logger.wandb_run.id if self.wandb_logger else None
                        }

                    # Save last, best and delete
                    torch.save(checkpoint_state, self.last_checkpoint)
                    if self.best_fitness == fi:
                        torch.save(checkpoint_state, self.best_checkpoint)
                    if (self.best_fitness == fi) and (epoch >= 1):
                        torch.save(checkpoint_state, self.weights_dir / 'best_{:03d}.pt'.format(epoch))
                    if epoch == 0:
                        torch.save(checkpoint_state, self.weights_dir / 'epoch_{:03d}.pt'.format(epoch))
                    elif ((epoch + 1) % 25) == 0:
                        torch.save(checkpoint_state, self.weights_dir / 'epoch_{:03d}.pt'.format(epoch))
                    elif epoch >= (epochs - 5):
                        torch.save(checkpoint_state, self.weights_dir / 'epoch_{:03d}.pt'.format(epoch))
                    if self.wandb_logger:
                        if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                            self.wandb_logger.log_model(
                                self.last_checkpoint.parent, opt, epoch, fi, best_model=self.best_fitness == fi)
                    del checkpoint_state

            # end epoch -----------------------------------------------------------------------------------------------
        # end training
        return epoch, results

    def calc_image_weights(
            self,
            dataset: IDataset,
            map_per_class: np.ndarray,
            model: IModel,
    ):
        # Generate indices
        if self.rank in [-1, 0]:
            # calculate class weights
            class_weights = model.class_weights.cpu().numpy() * (1 - map_per_class) ** 2 / dataset.classes_number
            # calculate image weights
            iw = labels_to_image_weights(
                labels=dataset.labels,
                classes_number=dataset.classes_number,
                class_weights=class_weights
            )
            # rand weighted idx
            dataset.indices = random.choices(range(dataset.classes_number), weights=iw, k=dataset.classes_number)
        # Broadcast if DDP
        if self.rank != -1:
            indices = (torch.tensor(dataset.indices) if self.rank == 0 else torch.zeros(dataset.classes_number)).int()
            dist.broadcast(indices, 0)
            if self.rank != 0:
                dataset.indices = indices.cpu().numpy()

    @staticmethod
    def rescale_batch_images(batch_images, grid_size):
        image_size = batch_images.shape[2]
        # TODO: switch on generate new scale factor to take into account both image size
        new_image_size = random.randrange(
            image_size * 0.5,
            image_size * 1.5 + grid_size
        ) // grid_size * grid_size  # size
        scale_factor = new_image_size / max(batch_images.shape[2:])  # scale factor
        if scale_factor != 1:
            # new shape (stretched to gs-multiple)
            ns = [math.ceil(x * scale_factor / grid_size) * grid_size for x in batch_images.shape[2:]]
            batch_images = func.interpolate(batch_images, size=ns, mode='bilinear', align_corners=False)
        return batch_images

    def  restore(self):
        pass


def evolve_pyperparameters():
    global f, hyp, opt
    # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
    meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
            'paste_in': (1, 0.0, 1.0)}  # segment copy-paste (probability)
    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
        if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            hyp['anchors'] = 3
    assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
    opt.notest, opt.nosave = True, True  # only test/save final epoch
    # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
    yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
    if opt.bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists
    for _ in range(300):  # generations to evolve
        if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
            # Select parent(s)
            parent = 'single'  # parent selection method: 'single' or 'weighted'
            x = np.loadtxt('evolve.txt', ndmin=2)
            n = min(5, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness(x))][:n]  # top n mutations
            w = fitness(x) - fitness(x).min()  # weights
            if parent == 'single' or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == 'weighted':
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            mp, s = 0.8, 0.2  # mutation probability, sigma
            npr = np.random
            npr.seed(int(time.time()))
            g = np.array([x[0] for x in meta.values()])  # gains 0-1
            ng = len(meta)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
            for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                hyp[k] = float(x[i + 7] * v[i])  # mutate

        # Constrain to limits
        for k, v in meta.items():
            hyp[k] = max(hyp[k], v[1])  # lower limit
            hyp[k] = min(hyp[k], v[2])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

        # Train mutation
        results = train(hyp.copy(), opt, device)

        # Write mutation results
        print_mutation(hyp.copy(), results, yaml_file, opt.bucket)
    # Plot results
    plot_evolution(yaml_file)
    print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
          f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--global-batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--gpu-batch-size', type=int, default=8, help='batch size per one GPU')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image sizes for train and test (width, height)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--plots_debug', action='store_true', help='plot images each epoch')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--sparsity', action='store_true', help='train model with additional regularization for pruning')
    parser.add_argument('--prune-train', action='store_true', help='')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    opt = parser.parse_args()
    opt.data = Path(opt.data)
    if not opt.data.is_absolute():
        opt.data = opt.data.resolve()
    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    # if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.sparsity:
        logger.info('Train sparsity model')
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg = ''
        opt.weights = ckpt
        opt.resume = True
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    device = select_device(opt.device, batch_size=opt.global_batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.global_batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.local_batch_size = opt.global_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        evolve_pyperparameters()
