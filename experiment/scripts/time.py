from time import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from deepinv.utils.demo import load_url_image, get_image_url

from src.pr import RandomPhaseRetrieval, StructuredRandomPhaseRetrieval

img_sizes: list[int] = [8, 16, 32, 64, 96, 128, 160, 192, 224, 240]
print("Image sizes:", img_sizes)
n_repeats = 100

url = get_image_url("SheppLogan.png")

model = "structured"  # 'dense' or 'structured'
device = "cpu"  # 'cpu' or 'cuda'

df = pd.DataFrame(
    {
        **{f"img_size_{img_size}": None for img_size in np.array(img_sizes)},
    },
    index=[0],
)

for img_size in tqdm(img_sizes):
    x = load_url_image(
        url=url, img_size=img_size, grayscale=True, resize_mode="resize", device=device
    )
    x_phase = torch.exp(1j * x * torch.pi - 0.5j * torch.pi).to(device)

    for i in range(n_repeats):
        if model == "dense":
            physics = RandomPhaseRetrieval(
                m=int(torch.prod(torch.tensor(x.shape))),
                img_size=(1, img_size, img_size),
                dtype=torch.cfloat,
                device=device,
            )
        elif model == "structured":
            physics = StructuredRandomPhaseRetrieval(
                img_size=(1, img_size, img_size),
                output_size=(1, img_size, img_size),
                dtype=torch.cfloat,
                device=device,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        init_time = time()
        if device == "cuda":
            torch.cuda.synchronize()
        y = physics.forward(x_phase)
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time()
        if device == "cuda":
            torch.cuda.synchronize()
            physics.release_memory()
            torch.cuda.empty_cache()
        df.loc[i, f"img_size_{img_size}"] = end_time - init_time

df.to_csv(f"../runs/df_{model}_{device}.csv", index=False)
print("Results saved to", f"../runs/df_{model}_{device}.csv")
