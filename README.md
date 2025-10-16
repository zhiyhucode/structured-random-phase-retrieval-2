Structured Random Phase Retrieval Model using Optical Diffusers
===========================================================

This repository contains the source code for the paper "Structured Random Phase Retrieval Model using Optical Diffusers".

Overview
--------

We propose a novel type of random model for phase retrieval utilizing structured transforms. It achieves the same reconstruction performance as classical dense random models while reducing the time complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(n\,\log\,n)$. The implementation is based on the open-source computational imaging library [deepinv](https://github.com/deepinv/deepinv).

Repository Structure
--------------------

- `deepinv/`: Contains the deepinv source code including the implementation of the structured random phase retrieval model.
- `experimental/config/`: Contains the configuration to run the reconstruction script.
- `experimental/scripts/`: Contains the scripts to benchmark the reconstruction performance and time complexity of different random models.

Getting Started
---------------

1. Clone the repository:
   
   ```bash
   git clone https://github.com/zhiyhucode/structured-random-phase-retrieval-v2.git
   cd structured-random-phase-retrieval-v2
   ```

2. Install the required dependencies:
   1. Run `pip install -e .` to install dependencies including `deepinv` from the `deepinv/` folder;

   1. Optionally, run `pip install --no-build-isolation fast-hadamard-transform`, if you have a GPU;
   
   Besides the `pip` option, we recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/), which has the same installing process except changing the command `pip` to `uv pip`. 

   After installing the dependencies, activate the environment using `source .venv/bin/activate`;

3. Navigate to the scripts directory to generate the benchmarks, e.g.:
   
   ```bash
   cd experimental/scripts
   python recon.py
   ```
   
   The config to define the experiments is stored in `experimental/config/config.yaml`

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

If you are interested in citing the work, please cite our paper:

```bibtex
TBD
```