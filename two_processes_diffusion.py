"""This example is reproducible."""
import configparser
from typing import Any

import numpy as np
np.random.seed(0)  # fix all seeds of the numpy which is used in the library

from network_diffusion import (
    MultilayerNetwork,
    MultiSpreading,
    PropagationModel,
)


def set_node_state(
    experiment: MultiSpreading, layer_name: str, node_name: Any, state: str
) -> None:
    """Allows to set up the initial state of certain node."""
    experiment._network.layers[layer_name].nodes[node_name]["status"] = state


# read global config
config = configparser.ConfigParser()
config.read("config.ini")
output_dir = config.get("PATHS", "output_dir")

# initialise graph
network = MultilayerNetwork()
network.load_mlx("auxiliaries/florentine.mpx")
network.describe()

# initialise propagation model
model = PropagationModel()
phenomenas = [["S", "I"], ["UV", "V"]]
for l, p in zip(network.get_nodes_states(), phenomenas):
    model.add(l, p)

# set possible transitions with weights
model.compile(background_weight=0.01)

model.set_transition_fast("marriage.S", "marriage.I", ["business.UV"], 0.9)
model.set_transition_fast("marriage.S", "marriage.I", ["business.V"], 0.3)

model.set_transition_fast("business.UV", "business.V", ["marriage.S"], 0.05)
model.set_transition_fast("business.UV", "business.V", ["marriage.I"], 0.03)
model.set_transition_fast("business.V", "business.UV", ["marriage.S"], 0)
model.set_transition_fast("business.V", "business.UV", ["marriage.I"], 0)

# initialise experiment
experiment = MultiSpreading(model, network)

# initialise starting parameters of propagation in network
experiment.set_initial_states({"marriage": (15, 0), "business": (11, 0)})
set_node_state(experiment, "marriage", "Ginori", "I")
set_node_state(experiment, "marriage", "Ridolfi", "I")
set_node_state(experiment, "marriage", "Medici", "I")
set_node_state(experiment, "business", "Lamberteschi", "V")

# perform propagation experiment
logs = experiment.perform_propagation(n_epochs=200)

# save experiment results
logs.report(to_file=True, path=output_dir, visualisation=True)
