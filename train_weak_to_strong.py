import json
import os
from typing import Dict, List, Optional, Sequence, Union

import fire
import numpy as np
import torch

import weak_to_strong.logger as logger
from weak_to_strong.utils import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset, tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model

# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
model_conf_params = {}

#create configs
param_sizes = ["160m","410m","1b","1.4b","2.8b"]
learning_rates = [5e-5,5e-5,5e-5,5e-5,5e-5]
batch_sizes = [48,48,36,24,18]
model_names = [f"EleutherAI/pythia-{params}" for params in param_sizes]

for idx,param in enumerate(param_sizes):
    model_name = f"EleutherAI/pythia-{param}"
    model_conf_params[model_name] = {"default_lr":5e-5,
    "eval_batch_size":batch_sizes[idx],
    "batch_size":batch_sizes[idx],
    "minibatch_size_per_device":int(batch_sizes[idx]/6)
    }

MODELS_DICT = {}

for key, value in model_conf_params.items():

    MODELS_DICT[key] = model_config_instance = ModelConfig(
        name=key,
        default_lr=value["default_lr"],
        eval_batch_size=value["eval_batch_size"],
        batch_size=value["batch_size"],
        minibatch_size_per_device=value["minibatch_size_per_device"]
    )



loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())


def train_w2s(
    batch_size: int = 18,
    max_ctx: int = 1024,
    ds_name: str = "boolq",
    transfer_loss: Union[str, Sequence[str]] = "xent,logconf",
    n_docs: int = 10000,
    n_test_docs: int = 200,
    weak_model_size: str = "EleutherAI/pythia-70m",
    weak_model_ckpt: str = "step1000",
    weak_lr: Optional[float] = None,
    strong_model_size: str = "EleutherAI/pythia-410m",
    strong_model_ckpt: str = "step2000",
    strong_lr: Optional[float] = None,
    # Defaults to strong_lr
    transfer_lr: Optional[float] = None,
    # Optims default to default_optimizer in the model definitions
    weak_optim: Optional[str] = None,
    strong_optim: Optional[str] = None,
    transfer_optim: Optional[str] = None,
    gt_epochs: int = 2,
    # defaults to gt_epochs
    transfer_epochs: Optional[int] = None,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[int] = 3,
    train_with_dropout: bool = False,
    results_folder: str = "/tmp/results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    log_prefix: str = "",
    # Set to an absurdly high value so we don't do intermediate evals by default.
    eval_every: int = 100000000,
):
    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    if isinstance(transfer_loss, str):
        transfer_losses = transfer_loss.split(",")
    else:
        transfer_losses = transfer_loss
    del transfer_loss
    for tloss in transfer_losses:
        assert tloss in VALID_LOSSES, f"Unknown loss {tloss} not in {VALID_LOSSES}"
    assert (
        weak_model_size in MODELS_DICT
    ), f"Unknown model size {weak_model_size} not in {MODELS_DICT}"
    weak_model_config = MODELS_DICT[weak_model_size]
    assert (
        strong_model_size in MODELS_DICT
    ), f"Unknown model size {strong_model_size} not in {MODELS_DICT}"
    strong_model_config = MODELS_DICT[strong_model_size]

    if weak_lr is None:
        #assert batch_size == 32
        weak_lr = weak_model_config.default_lr
    if strong_lr is None:
        #assert batch_size == 32
        strong_lr = strong_model_config.default_lr
    if transfer_lr is None:
        transfer_lr = strong_lr
    if transfer_epochs is None:
        transfer_epochs = gt_epochs
    if weak_optim is None:
        weak_optim = weak_model_config.default_optimizer
    if strong_optim is None:
        strong_optim = strong_model_config.default_optimizer
    if transfer_optim is None:
        transfer_optim = strong_optim

    weak_eval_batch_size = weak_model_config.eval_batch_size
    strong_eval_batch_size = strong_model_config.eval_batch_size

    # Load dataset
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]

    split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
    train1_ds, train2_ds = split_data["train"], split_data["test"]
    print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))

    ########### train_model function ###################################
    def train_model(
        model_config: ModelConfig,
        checkpoint: str,
        train_ds: torch.utils.data.Dataset,
        test_ds: torch.utils.data.Dataset,
        *,
        loss_type: str,
        label: str,
        subpath,
        lr,
        eval_batch_size,
        epochs=1,
        inference_ds: Optional[torch.utils.data.Dataset] = None,
        linear_probe: bool = False,
        optimizer_name: str = "adam"):

        save_path = os.path.join(results_folder, subpath)
        linprobe_str = "_linprobe" if linear_probe else ""
        logger.configure(
            name="{log_prefix}{label}_{base_model_name}_{checkpoint}_{ds_name}_{loss_type}_{optimizer_name}_{lr}_{lr_schedule}{linprobe_str}_{datetime_now}",
            label=label,
            ds_name=ds_name,
            truncation_max_len=n_docs or "none",
            loss_type=loss_type,
            lr=lr,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            minibatch_size_per_device=minibatch_size_per_device,
            save_path=save_path,
            base_model_name=model_config.name,
            epochs=epochs,
            linprobe_str=linprobe_str,
            lr_schedule=lr_schedule,
            log_prefix=log_prefix,
            optimizer_name=optimizer_name,
        )
        # Tokenize datasets
        tokenizer = get_tokenizer(model_config.name)
        train_ds = tokenize_dataset(train_ds, tokenizer, max_ctx)
        test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)
        if inference_ds:
            inference_ds = tokenize_dataset(inference_ds, tokenizer, max_ctx)

        loss_fn = loss_dict[loss_type]
        print(f"minibatch_size_per_device is :{minibatch_size_per_device}")
        return train_and_save_model(
            model_config,
            checkpoint,
            train_ds,
            test_ds,
            inference_ds=inference_ds,
            batch_size=model_config.batch_size,
            save_path=save_path,
            loss_fn=loss_fn,
            lr=lr,
            epochs=epochs,
            force_retrain=force_retrain,
            eval_batch_size=model_config.eval_batch_size,
            minibatch_size_per_device=model_config.minibatch_size_per_device,
            train_with_dropout=train_with_dropout,
            linear_probe=linear_probe,
            lr_schedule=lr_schedule,
            optimizer_name=optimizer_name,
            eval_every=eval_every)
    ######################################################################

    # Train the weak model on the first half of the training data
    print(f"Training weak model, size {weak_model_size}")
    weak_test_results, weak_ds = train_model(
        weak_model_config,
        weak_model_ckpt,
        train1_ds,
        test_ds,
        loss_type="xent",
        label="weak",
        subpath=os.path.join("weak_model_gt", weak_model_size.replace("/", "_"),weak_model_ckpt),
        lr=weak_lr,
        eval_batch_size=weak_eval_batch_size,
        inference_ds=train2_ds,
        epochs=gt_epochs,
        linear_probe=linear_probe,
        optimizer_name=weak_optim,
    )

    # Train the strong model on the second half of the training data
    print(f"Training strong model, size {strong_model_size}")
    strong_test_results, _ = train_model(
        strong_model_config,
        strong_model_ckpt,
        train2_ds,
        test_ds,
        loss_type="xent",
        label="strong",
        subpath=os.path.join("strong_model_gt", strong_model_size.replace("/", "_"),strong_model_ckpt),
        lr=strong_lr,
        eval_batch_size=strong_eval_batch_size,
        epochs=gt_epochs,
        linear_probe=linear_probe,
        optimizer_name=strong_optim,
    )

    # Train the strong model on the second half of the training data with labels generated by the weak model
    all_transfer_test_results = {}
    for tloss in transfer_losses:
        print(
            f"Training transfer model, size {strong_model_size} on labels from {weak_model_size}, with loss {tloss}"
        )
        transfer_test_results, _ = train_model(
            strong_model_config,
            strong_model_ckpt,
            weak_ds,
            test_ds,
            loss_type=tloss,
            label="weak2strong",
            subpath=os.path.join(
                "strong_model_transfer",
                f"{weak_model_size.replace('/', '_')}_{weak_model_ckpt}_{strong_model_size.replace('/', '_')}_{strong_model_ckpt}_{tloss}",
            ),
            lr=transfer_lr,
            eval_batch_size=strong_eval_batch_size,
            epochs=transfer_epochs,
            linear_probe=linear_probe,
            optimizer_name=transfer_optim,
        )
        all_transfer_test_results[tloss] = transfer_test_results
        del transfer_test_results

    weak_acc = np.mean([x["acc"] for x in weak_test_results])
    strong_acc = np.mean([x["acc"] for x in strong_test_results])
    res_dict = { "weak_acc": weak_acc, "strong_acc": strong_acc,}
    print("weak acc:", weak_acc)
    print("strong acc:", strong_acc)
    for tloss, transfer_test_results in all_transfer_test_results.items():
        transfer_acc = np.mean([x["acc"] for x in transfer_test_results])
        res_dict[f"transfer_acc_{tloss}"] = transfer_acc
        print(f"transfer acc ({tloss}):", transfer_acc)

    with open(os.path.join(results_folder,f"{weak_model_size.replace('/', '_')}_{weak_model_ckpt}_{strong_model_size.replace('/', '_')}_{strong_model_ckpt}_.results_summary.json",),"w",) as f:
        json.dump(res_dict,f,)
    return

if __name__ == "__main__":

    train_params = {
        'batch_size': 18,
        'max_ctx': 1024,
        'ds_name': "boolq",
        'transfer_loss': "xent,logconf",
        'n_docs': 10000,
        'n_test_docs': 200,
        'weak_model_size': "EleutherAI/pythia-70m",
        'weak_model_ckpt': "step1000",
        'weak_lr': None,
        'strong_model_size': "EleutherAI/pythia-410m",
        'strong_model_ckpt': "step2000",
        'strong_lr': None,
        'transfer_lr': None,
        'weak_optim': None,
        'strong_optim': None,
        'transfer_optim': None,
        'gt_epochs': 2,
        'transfer_epochs': None,
        'force_retrain': True,
        'seed': 0,
        'minibatch_size_per_device': 3,
        'train_with_dropout': False,
        'results_folder': "results",
        'linear_probe': False,
        'lr_schedule': "cosine_anneal",
        'log_prefix': "",
        'eval_every': 100000000}
    
    from itertools import product,combinations
    from tqdm import tqdm

    param_list =  list(reversed(["160m","410m","1b","1.4b","2.8b"]))
    step_list = list(range(1, 155, 30))

    model_combinations = product(param_list, repeat=2)
    checkpoint_combinations = product(step_list, repeat=2)

    for (weak_model_param, strong_model_param), (weak_ckpt_step, strong_ckpt_step) in tqdm(product(model_combinations, checkpoint_combinations)):

        weak_model_size_str = f"EleutherAI/pythia-{weak_model_param}"
        strong_model_size_str = f"EleutherAI/pythia-{strong_model_param}"
        weak_model_ckpt_str = f"step{str(int(weak_ckpt_step * 1000))}"
        strong_model_ckpt_str = f"step{str(int(strong_ckpt_step * 1000))}"

        train_params["weak_model_size"] = weak_model_size_str
        train_params["strong_model_size"] = strong_model_size_str
        train_params["weak_model_ckpt"] = weak_model_ckpt_str
        train_params["strong_model_ckpt"] = strong_model_ckpt_str

        file_path = os.path.join("results", f"{weak_model_size_str.replace('/', '_')}_{weak_model_ckpt_str}_{strong_model_size_str.replace('/', '_')}_{strong_model_ckpt_str}_.results_summary.json")
        
        if weak_model_param not in ["1b", "1.4b", "2.8b"]:
            if (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
                print (f"filepath: {file_path} exists")
            else:
                print(
                    f"Train parameters: "
                    f"weak_model_size={weak_model_size_str}, strong_model_size={strong_model_size_str}; "
                    f"weak_model_ckpt={weak_model_ckpt_str}, strong_model_ckpt={strong_model_ckpt_str}"
                )

                try:
                    train_w2s(**train_params)
                except Exception as e:
                    print(f"An exception occurred: {type(e).__name__}: {e}")
