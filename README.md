# AMIL
Adversarial Mutual Information Learning for Network Embedding


## Overview
Here we provide an implementation of AMIL in PyTorch, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files;
- `results/` contains the embedding results;
- `layers.py` contains the implementation of a GCN layer;
- `utils.py` contains the necessary processing function.
- `model.py` contains the definition of the whole model, including discriminator, generator and GAE model.
- `optimizer.py` contains the implementation of the reconstruction loss.

Finally, `train.py` puts all of the above together and may be used to execute a full training run on Cora.

## Reference
If you make advantage of AMIL in your research, please cite the following in your manuscript:

He, Dongxiao, et al. "Adversarial Mutual Information Learning for Network Embedding." Proceedings of IJCAI. 2020.

## License
Tianjin University
