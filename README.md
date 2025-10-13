Structured Random Phase Retrieval Model using Optical Diffusors
===========================================================

This repository contains the source code for the paper "Structured Random Phase Retrieval Model using Optical Diffusors".

Overview
--------

We propose a novel type of random model for phase retrieval utilizing structured transforms. It achieves the same reconstruction performance as classical dense random models while reducing the time complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$. The implementation is based on the open-source computational imaging library [deepinv](https://github.com/deepinv/deepinv).

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
   cd structured-random-phase-retrieval
   ```

2. Install the required dependencies:
   
   1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/);

   2. Run `uv sync` under the root directory of the project;

   3. Optionally, run `uv pip install --no-build-isolation fast-hadamard-transform`, if you have a GPU;

   4. Activate the environment using `source .venv/bin/activate`;

3. Navigate to the scripts directory to generate the benchmarks, e.g.:
   
   ```bash
   cd experimental/paper/scripts
   python recon.py
   ```
   
   The config to define the experiments is stored in `experimental/config/config.yaml`

Contributing
------------

We welcome contributions to improve the code or extend the research. Please submit a pull request or open an issue for any bugs or feature requests.

License
-------

This project is licensed under the BSD 3-Clause License - see the `LICENSE` file for details.

Citation
--------

If you are interested in citing the work, please cite our paper:

```bibtex
TBD
```
