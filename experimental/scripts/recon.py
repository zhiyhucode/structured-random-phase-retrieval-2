from datetime import datetime
import os
from pathlib import Path
import shutil
import sys

from loguru import logger
import pandas as pd
import torch
from tqdm import trange
import yaml

import deepinv as dinv
from deepinv.optim.data_fidelity import L2, AmplitudeLoss
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.phase_retrieval import (
    compute_lipschitz_constant,
    cosine_similarity,
    generate_signal,
    init_with,
    spectral_methods,
)
from deepinv.optim.prior import Zero
from deepinv.physics import RandomPhaseRetrieval, StructuredRandomPhaseRetrieval


if __name__ == "__main__":
    config_path = "../config/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # run
    model = config["run"]["model"]
    name = config["run"]["name"]
    algo = config["run"]["algo"]
    save = config["run"]["save"]["enable"]
    verbose = config["run"]["verbose"]

    # signal
    img_size = config["signal"]["shape"]
    signal_mode = config["signal"]["mode"]
    signal_config = config["signal"]["adversarial"]

    # model
    img_res = img_size[-1]  # image resolution
    if model == "dense":
        mode = config["model"]["dense"]["mode"]
        product = config["model"]["dense"]["product"]
    elif model == "structured":
        n_layers = config["model"]["structured"]["n_layers"]
        transforms = config["model"]["structured"]["transforms"]
        diagonals = config["model"]["structured"]["diagonals"]
        pad_powers_of_two = config["model"]["structured"]["pad_powers_of_two"]
        shared_weights = config["model"]["structured"]["shared_weights"]
        include_zero = config["model"]["structured"]["include_zero"]
        manual_spectrum = config["model"]["structured"]["manual_spectrum"]["mode"]

        structure = StructuredRandomPhaseRetrieval.get_structure(n_layers)
    else:
        raise ValueError(f"Invalid model: {model}")
    noise_percentage = float(config["model"]["noise_percentage"]) / 100

    # recon
    n_repeats = config["recon"]["n_repeats"]
    if config["recon"]["series"] == "arange":
        if model == "dense":
            start = config["recon"]["dense"]["start"]
            end = config["recon"]["dense"]["end"]
            step = config["recon"]["dense"]["step"]
            oversampling_ratios = torch.arange(start, end, step)
        elif model == "structured":
            start = config["recon"]["structured"]["start"]
            end = config["recon"]["structured"]["end"]
            output_reses = torch.arange(start, end, 2)
            oversampling_ratios = output_reses**2 / img_res**2
        else:
            raise ValueError(f"Invalid model: {model}")
    elif config["recon"]["series"] == "list":
        if model == "dense":
            oversampling_ratios = torch.tensor(config["recon"]["dense"]["list"])
        elif model == "structured":
            output_reses = torch.tensor(config["recon"]["structured"]["list"])
            oversampling_ratios = output_reses**2 / img_res**2
        else:
            raise ValueError(f"Invalid model: {model}")
    else:
        raise ValueError("Invalid series type.")
    n_oversampling = oversampling_ratios.shape[0]
    # spec
    max_iter_spec = config["recon"]["spec"]["max_iter"]
    # gd
    loss = config["recon"]["gd"]["loss"]
    if loss == "intensity":
        data_fidelity = L2()
    elif loss == "amplitude":
        data_fidelity = AmplitudeLoss()
    else:
        raise ValueError(f"Invalid data fidelity: {loss}")
    if config["recon"]["gd"]["prior"] == "zero":
        prior = Zero()
    else:
        raise ValueError(f"Invalid prior: {config['recon']['gd']['prior']}")
    early_stop = config["recon"]["gd"]["early_stop"]
    max_iter_gd = config["recon"]["gd"]["max_iter"]

    logger.remove()
    logger.add(sys.stderr, format="{message}", level="INFO")

    # save
    if save:
        if model == "dense":
            res_name = f"{model}_{name}_{algo}.csv"
        elif model == "structured":
            res_name = f"{model}_{structure}_{name}_{algo}.csv"
        else:
            raise ValueError(f"Invalid model: {model}")

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = Path(config["run"]["save"]["path"])
        save_dir = save_dir / (current_time + "_" + name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        logger.add(
            f"{save_dir}/experiment.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
        )
        shutil.copy(config_path, save_dir / "config.yaml")
        os.chmod(save_dir / "config.yaml", 0o444)  # read-only

        logger.info(f"Experiment: {model}_{name}_{algo}")
        logger.info(f"Save directory: {save_dir}")

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    # Set up the signal to be reconstructed.
    if signal_mode == ["adversarial"]:
        pass
    else:
        x = generate_signal(
            shape=img_size,
            mode=signal_mode,
            config=signal_config,
            phase_range=(0, 2 * torch.pi),
            dtype=torch.complex64,
            device=device,
        )

    df = pd.DataFrame(index=range(n_repeats), columns=oversampling_ratios.tolist())
    df.index.name = "run"
    df.columns.name = "oversampling_ratio"

    last_oversampling_ratio = -0.1

    try:
        for i in trange(n_oversampling):
            oversampling_ratio = oversampling_ratios[i]
            if oversampling_ratio - last_oversampling_ratio < 0.05:
                continue
            # skip oversampling 1 as it takes too much time to sample
            if oversampling_ratio > 0.99 and oversampling_ratio < 1.01:
                continue
            if model == "structured":
                output_res = output_reses[i]
                logger.info(f"Output size: {output_res}")
            logger.info(f"Oversampling ratio: {oversampling_ratio:.4f}")
            for j in range(n_repeats):
                if model == "dense":
                    physics = RandomPhaseRetrieval(
                        m=int(oversampling_ratio * img_res**2),
                        img_size=(1, img_res, img_res),
                        mode=mode,
                        product=product,
                        dtype=torch.complex64,
                        device=device,
                    )
                elif model == "structured":
                    # use spectrum from a full matrix
                    if manual_spectrum != "unit":
                        example = RandomPhaseRetrieval(
                            m=output_res**2,
                            img_size=(1, img_res, img_res),
                            mode=manual_spectrum,
                            product=config["model"]["structured"]["manual_spectrum"][
                                "product"
                            ],
                            dtype=torch.complex64,
                            device=device,
                        )
                        spectrum = torch.linalg.svdvals(example.B._A)
                        # bootstrap the spectrum to have the dimension of img_res**2
                        extra = img_res**2 - output_res**2
                        if extra > 0:
                            extra_indices = torch.randint(0, spectrum.numel(), (extra,))
                            extra_spectrum = spectrum[extra_indices]
                            spectrum = torch.cat((spectrum, extra_spectrum))
                        # permute the spectrum
                        spectrum = spectrum[torch.randperm(spectrum.numel())]
                        spectrum = spectrum.reshape(1, img_res, img_res)
                    else:
                        spectrum = "unit"
                    physics = StructuredRandomPhaseRetrieval(
                        img_size=(1, img_res, img_res),
                        output_size=(1, output_res, output_res),
                        n_layers=n_layers,
                        transforms=transforms,
                        diagonals=diagonals,
                        manual_spectrum=spectrum,
                        pad_powers_of_two=pad_powers_of_two,
                        shared_weights=shared_weights,
                        include_zero=include_zero,
                        dtype=torch.complex64,
                        device=device,
                        verbose=verbose,
                    )
                    if signal_mode == ["adversarial"]:
                        signal_config["physics"] = physics
                        x = generate_signal(
                            shape=img_size,
                            mode=signal_mode,
                            config=signal_config,
                            dtype=torch.complex64,
                        )
                else:
                    raise ValueError(f"Invalid model: {model}")

                y = physics(x)
                noise = torch.randn_like(y) * noise_percentage * y.mean()
                y = torch.clip(y + noise, min=0)

                if "gd" in algo:
                    if "rand" in algo:
                        x_recon = torch.randn_like(x)
                    elif "spec" in algo:
                        x_recon = spectral_methods(y, physics, n_iter=max_iter_spec)
                    else:
                        raise ValueError(f"Invalid algo: {algo}")

                    step_size = compute_lipschitz_constant(
                        x_recon, y, physics, config["recon"]["gd"]["spectrum"], loss
                    )
                    params_algo = {"stepsize": 2 / step_size.item(), "g_params": 0.00}
                    optimizer = optim_builder(
                        iteration="PGD",
                        prior=prior,
                        data_fidelity=data_fidelity,
                        early_stop=early_stop,
                        max_iter=max_iter_gd,
                        verbose=verbose,
                        params_algo=params_algo,
                        custom_init=init_with(x_recon),
                    )
                    x_recon = optimizer(y, physics, x_gt=x)
                elif algo == "spec":
                    x_recon = spectral_methods(y, physics, n_iter=max_iter_spec)
                else:
                    raise ValueError(f"Invalid algo: {algo}")

                df.iloc[j, i] = cosine_similarity(x, x_recon).item()
                logger.info(f"Run {j}, cosine similarity: {df.iloc[j, i]:.4f}")

                if save:
                    df.to_csv(save_dir / res_name)

                if model == "dense":
                    physics.release_memory()
            last_oversampling_ratio = oversampling_ratio
        if save:
            logger.info(f"Experiment {res_name} finished. Results saved at {save_dir}.")
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted!")
        if save:
            response = input(
                f"Do you want to delete the data folder {save_dir}? (y/n): "
            )
            if response.lower() in ["y", "yes"]:
                shutil.rmtree(save_dir)
                logger.info(f"Data folder {save_dir} deleted.")
            else:
                logger.info(f"Data folder {save_dir} retained.")
