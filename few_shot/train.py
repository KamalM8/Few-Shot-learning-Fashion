"""
The `fit` function in this file implements a slightly modified version
of the Keras `model.fit()` API.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union

from few_shot.callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback
from few_shot.metrics import NAMED_METRICS


def gradient_step(model: Module, optimiser: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):
    """Takes a single gradient step.

    # Arguments
        model: Model to be fitted
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples
        y: Input targets
    """
    model.train()
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


def batch_metrics(model: Module, y_pred: torch.Tensor, y: torch.Tensor, metrics: List[Union[str, Callable]],
                  batch_logs: dict):
    """Calculates metrics for the current training batch

    # Arguments
        model: Model being fit
        y_pred: predictions for a particular batch
        y: labels for a particular batch
        batch_logs: Dictionary of logs for the current batch
    """
    model.eval()
    for m in metrics:
        if isinstance(m, str):
            batch_logs[m] = NAMED_METRICS[m](y, y_pred)
        else:
            # Assume metric is a callable function
            batch_logs = m(y, y_pred)

    return batch_logs


def unnormalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Removes torch.transforms image normalization

    # Arguments
        imgs: torch tensor batch of normalized images
        mean: mean used in prior normalization
        std: std used in prior normalization
    """
    unnorm_imgs = torch.empty_like(imgs)
    mean_tensor = torch.FloatTensor(mean).view(3,1,1)
    std_tensor = torch.FloatTensor(std).view(3,1,1)
    for i, img in enumerate(imgs):
        unnorm_imgs[i] = img*std_tensor + mean_tensor
    return unnorm_imgs


def matplotlib_imshow(img, one_channel=False):
    """ Plots image

    # Arguments
        img: image to be drawn
        one_channel: Is it one channel (default: False)
    """

    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    npimg = (npimg*255).astype(np.uint8)
    if one_channel:
        plt.imshow(np.img, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))


def plot_classes_preds(images, n_shot, k_way):
    """ Returns image grid to be embedded into tensorboard

    # Arguments
        images: torch tensor batch of images to be plotted
        n_shot: number of shots per class
        k_way: number of classes

    """

    support = images[:n_shot*k_way].cpu()
    support = unnormalize(support)
    fig = plt.figure(figsize=(10,10))
    for idx in np.arange(support.shape[0]):
        ax = fig.add_subplot(k_way, n_shot, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(support[idx], one_channel=False)

    return fig


def fit(model: Module, optimiser: Optimizer, loss_fn: Callable, epochs: int, dataloader: DataLoader, writer: SummaryWriter,
        prepare_batch: Callable, metrics: List[Union[str, Callable]] = None, callbacks: List[Callback] = None,
        verbose: bool =True, fit_function: Callable = gradient_step,
        stnmodel = None,
        stnoptim = None,
        args = None,
        fit_function_kwargs: dict = {}):
    """Function to abstract away training loop.

    The benefit of this function is that allows training scripts to be much more readable and allows for easy re-use of
    common training functionality provided they are written as a subclass of voicemap.Callback (following the
    Keras API).

    # Arguments
        model: Model to be fitted.
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        epochs: Number of epochs of fitting to be performed
        dataloader: `torch.DataLoader` instance to fit the model to
        writer: `tensorboard.SummaryWriter` instance to write plots to tensorboard
        prepare_batch: Callable to perform any desired preprocessing
        metrics: Optional list of metrics to evaluate the model with
        callbacks: Additional functionality to incorporate into training such as logging metrics to csv, model
            checkpointing, learning rate scheduling etc... See voicemap.callbacks for more.
        verbose: All print output is muted if this argument is `False`
        fit_function: Function for calculating gradients. Leave as default for simple supervised training on labelled
            batches. For more complex training procedures (meta-learning etc...) you will need to write your own
            fit_function
        fit_function_kwargs: Keyword arguments to pass to `fit_function`
    """
    # Determine number of samples:
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimiser': optimiser
    })

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()

    for epoch in range(1, epochs+1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            callbacks.on_batch_begin(batch_index, batch_logs)

            x, y = prepare_batch(batch)

            fit_function_kwargs['stnmodel'] = stnmodel
            fit_function_kwargs['stnoptim'] = stnoptim
            fit_function_kwargs['args'] = args

            n_shot = fit_function_kwargs['n_shot']
            k_way = fit_function_kwargs['k_way']

            loss, y_pred, aug_imgs = fit_function(model, optimiser, loss_fn, x, y, **fit_function_kwargs)
            batch_logs['loss'] = loss.item()

            # Useful for viewing images for debugging (Doesn't work for maml)
            #TODO (kamal): customize for maml
            # if batch_index % 100 == 99:
                # writer.add_figure('episode', plot_classes_preds(aug_imgs, n_shot, k_way),
                        # global_step=len(dataloader)*(epoch-1) + batch_index)

            # Loops through all metrics
            batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)

            # Log training loss and categorical accuracy
            writer.add_scalar('Train_loss', batch_logs['loss'], len(dataloader)*(epoch-1) + batch_index)
            writer.add_scalar('categorical_accuracy', batch_logs['categorical_accuracy'], len(dataloader)*(epoch-1) + batch_index)

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()
