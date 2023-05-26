# Model Compression 
- Explore methods for neural network pruning: attempting to compress a simple neural network which has 592,933 parameters. The baseline network achieves ~63% test accuracy on a 5-class classification task after training for 50 epochs with the default parameters in the notebook.

# Chosen pruning methods
- Magnitude-based pruning: https://arxiv.org/abs/1506.02626
- ThiNet: https://arxiv.org/abs/1707.06342
- Network Slimming: https://arxiv.org/abs/1708.06519

## 1. baseline_model.py 
- generates the baseline pretrained model
`python3 baseline_model.py`

## 2. thinet.py
- generates a .h5 model which is a pruned, fine-tuned baseline model
`python3 thinet.py sparsity_val`

## 3. network_slimming.py
- generates a .h5 model which is a pruned, fine-tuned baseline model
`python3 network_slimming.py sparsity_val`

## 4. magnitude_pruning.py
- generates a .h5 model which is a pruned, fine-tuned baseline model
`python3 magnitude_pruning.py sparsity_val`
