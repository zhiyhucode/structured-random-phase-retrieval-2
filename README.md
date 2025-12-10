Structured Random Phase Retrieval 2
===========================================================

This repository hosts the source code for the paper "[Structured Random Models for Phase Retrieval with Optical Diffusers](https://www.arxiv.org/abs/2510.14490)".

Overview
--------

This work proposes a novel type of random models for phase retrieval utilizing structured transforms. It achieves the same reconstruction performance as classical dense random models while reducing the time complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$. The implementation is based on the open-source computational imaging library [deepinv](https://deepinv.github.io/deepinv/).

Repository Structure
--------------------

- `src/`: Contains the implementation of the structured random phase retrieval model.
- `experiment/config/`: Contains the configuration to run the reconstruction script.
- `experiment/scripts/`: Contains the scripts to benchmark the reconstruction performance and time complexity of different random models.

Getting Started
---------------

1. Clone the repository:
   
   ```bash
   git clone https://github.com/zhiyhucode/structured-random-phase-retrieval-2.git
   cd structured-random-phase-retrieval-2
   ```

2. Install the required dependencies:
   1. Run `pip install -e .` to install dependencies;

   1. Optionally, run `pip install --no-build-isolation fast-hadamard-transform`, if you have a GPU;
   
   Besides the `pip` option, we recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/), which has the same installation process except replacing the command `pip` to `uv pip`. 

   After installing the dependencies, activate the environment using `source .venv/bin/activate`;

3. Navigate to the scripts directory to generate the benchmarks, e.g.:
   
   ```bash
   cd experiment/scripts
   python recon.py
   ```
   
   The config to define the experiments is stored in `experiment/config/config.yaml`

Contributing
------------

We welcome any contributions to improve the code or extend the research. If you encounter any bugs, have feature requests, or would like to contribute improvements, please:

- Submit a pull request with your proposed changes
- Open an issue to report bugs or suggest new features
- Contact the developer directly at [zhiyuan.hu@epfl.ch](mailto:zhiyuan.hu@epfl.ch)

License
-------

This project is licensed under the BSD 3-Clause License - see the `LICENSE` file for details.

Citation
--------

If you find this work useful, please consider citing our paper:

```bibtex
@article{hu2025structured,
  title={Structured Random Models for Phase Retrieval with Optical Diffusers},
  author={Hu, Zhiyuan and Mammadova, Fakhriyya and Tachella, Juli{\'a}n and Unser, Michael and Dong, Jonathan},
  journal={arXiv preprint arXiv:2510.14490},
  year={2025}
}
```
