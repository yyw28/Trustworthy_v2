# Machine Learning Template Repository

This repository contains a template for all of my ML projects. The template includes the following components:

- Basic project structure built around PyTorch Lightning, including example models, dataloaders, tests, etc.
- A basic pyproject.toml file, including imports I use in most projects.
- A simple executable that bootstraps [Lightning CLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html).

Once you make your own project from this repository, create a virtual environment, enter it, then run:

```pip install -e .```

Please use the `config/` directory to manage different model configurations.

Once you have customized the repository, start training with:

```
<executable name> fit --config config/<configuration file>.json
```
