"""
B⊕LD configuration: tanh' re-weighting, no backprop scaling, beta per layer.
This is the baseline: the paper's own choices (App. C.3, Eq. 11 at layer level),
without the sqrt(2/m) scaling.

    bash run_config.sh bold_tanh_noscale_layer.py        # everything, on ada
    python bold_tanh_noscale_layer.py search <id>         # or phase by phase
    python bold_tanh_noscale_layer.py choose
    python bold_tanh_noscale_layer.py final <id>
    python bold_tanh_noscale_layer.py summary

Other configurations are copies of this file with different settings below; the file
name is the name of the results folder (results/bold_tanh_noscale_layer/).
"""

from bold_experiment import cli

CONFIG = {
    "reweight": "tanh",     # "tanh", "inv_sq", "exp", "window", "none"   (BOLD App. C.3)
    "window": 2,            # only for "window": signal passes if the unit is <= this many steps from switching
    "bp_scale": False,      # sqrt(2/m) backprop scaling                  (BOLD App. C.2)
    "beta": "layer",        # "layer", "neuron", "network", or 0 (no accumulator)   (BOLD Eq. 11)
}

if __name__ == "__main__":
    cli(__file__, CONFIG)
