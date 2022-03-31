"""This example is reproducible."""
import configparser
from typing import Any

import networkx as nx
import numpy as np
np.random.seed(0)  # fix all seeds of the numpy which is used in the library

from network_diffusion import (
    MultilayerNetwork,
    MultiSpreading,
    PropagationModel,
)

# read global config
config = configparser.ConfigParser()
config.read("config.ini")
output_dir = config.get("PATHS", "output_dir")

# initialise graph
network = MultilayerNetwork()
network.load_layer_nx(nx.erdos_renyi_graph(150, 0.1), ["candidate_a", "candidate_b"])

# candidate_a uses "positive" propaganda, i.e. he gains popularity by promotinh himself
# candidate_b uses "negative" propaganda, i.e. he gains popularity by blackmailing candidate_a

# initialise propagation model
model = PropagationModel()
model.add("candidate_a", ["negative", "neutral", "positive"])
model.add("candidate_b", ["negative", "neutral", "positive"])

# by default all transition are pseudo-probability equals to 0
model.compile(background_weight=0.01)

# attitude to candidate_a changes having attitude to candidate_b constantly neutral
model.set_transition_fast("candidate_a.neutral", "candidate_a.positive", ["candidate_b.neutral"], 0.1)
model.set_transition_fast("candidate_b.neutral", "candidate_b.negative", ["candidate_a.neutral"], 0.1)

# attitude to candidate_b changes having attitude to candidate_a constantly neutral
model.set_transition_fast("candidate_b.neutral", "candidate_b.positive", ["candidate_a.neutral"], 0.1)
model.set_transition_fast("candidate_a.neutral", "candidate_a.negative", ["candidate_b.neutral"], 0.1)

# auxiliary transitions that polarise attitude of the population
model.set_transition_fast("candidate_a.neutral", "candidate_a.negative", ["candidate_b.positive"], 0.1)
model.set_transition_fast("candidate_b.neutral", "candidate_b.positive", ["candidate_a.negative"], 1)
model.set_transition_fast("candidate_b.neutral", "candidate_b.negative", ["candidate_a.positive"], 0.1)
model.set_transition_fast("candidate_a.neutral", "candidate_a.positive", ["candidate_b.negative"], 0.1)


# initialise experiment
experiment = MultiSpreading(model, network)

# initialise starting parameters of propagation in network
experiment.set_initial_states({"candidate_a": (2, 146, 2), "candidate_b": (2, 146, 2)})

# perform propagation experiment
logs = experiment.perform_propagation(n_epochs=200)

# save experiment results
logs.report(to_file=True, path=output_dir, visualisation=True)
