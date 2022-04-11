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


def set_node_state(
    experiment: MultiSpreading, layer_name: str, node_name: Any, state: str
) -> None:
    """Allows to set up the initial state of certain node."""
    experiment._network.layers[layer_name].nodes[node_name]["status"] = state


# read global config
config = configparser.ConfigParser()
config.read("config.ini")
output_dir = config.get("PATHS", "output_dir")

# initialise propagation model
M = PropagationModel()
T = ["ill", "awar", "vacc"]
t = [["s", "i", "r"], ["n", "a"], ["u", "v"]]
for proc_name, states in zip(T, t):
    M.add(proc_name, states)

M.compile(background_weight=0.005)

M.set_transition_fast("ill.s", "ill.i", ("vacc.u", "awar.n"), 0.4)
M.set_transition_fast("ill.s", "ill.i", ("vacc.v", "awar.a"), 0.05)
M.set_transition_fast("ill.s", "ill.i", ("vacc.u", "awar.a"), 0.2)
M.set_transition_fast("ill.i", "ill.r", ("vacc.u", "awar.n"), 0.1)
M.set_transition_fast("ill.i", "ill.r", ("vacc.v", "awar.a"), 0.7)
M.set_transition_fast("ill.i", "ill.r", ("vacc.u", "awar.a"), 0.3)
M.set_transition_fast("vacc.u", "vacc.v", ("awar.a", "ill.s"), 0.03)
M.set_transition_fast("vacc.u", "vacc.v", ("awar.a", "ill.i"), 0.1)
M.set_transition_fast("awar.n", "awar.a", ("vacc.u", "ill.s"), 0.05)
M.set_transition_fast("awar.n", "awar.a", ("vacc.v", "ill.s"), 1)
M.set_transition_fast("awar.n", "awar.a", ("vacc.u", "ill.i"), 0.2)

# initialise graph
N = MultilayerNetwork()
N.load_layer_nx(nx.les_miserables_graph(), T)

# perform propagation experiment
experiment = MultiSpreading(M, N)

experiment.set_initial_states({"ill": (77, 0, 0), "awar": (77, 0), "vacc": (77, 0)})

set_node_state(experiment, "ill", "Javert", "i")
set_node_state(experiment, "ill", "Simplice", "i")
set_node_state(experiment, "ill", "Scaufflaire", "i")
set_node_state(experiment, "ill", "Cochepaille", "i")
set_node_state(experiment, "ill", "Eponine", "i")
set_node_state(experiment, "ill", "Anzelma", "i")
set_node_state(experiment, "ill", "Woman2", "i")
set_node_state(experiment, "ill", "Gavroche", "i")
set_node_state(experiment, "ill", "Magnon", "i")
set_node_state(experiment, "ill", "Courfeyrac", "i")

set_node_state(experiment, "ill", "Child2", "r")
set_node_state(experiment, "ill", "MmeHucheloup", "r")

set_node_state(experiment, "awar", "Toussaint", "a")
set_node_state(experiment, "awar", "Gueulemer", "a")
set_node_state(experiment, "awar", "Feuilly", "a")
set_node_state(experiment, "awar", "LtGillenormand", "a")
set_node_state(experiment, "awar", "Magnon", "a")
set_node_state(experiment, "awar", "Gavroche", "a")
set_node_state(experiment, "awar", "Anzelma", "a")
set_node_state(experiment, "awar", "Chenildieu", "a")
set_node_state(experiment, "awar", "Brevet", "a")
set_node_state(experiment, "awar", "Champmathieu", "a")
set_node_state(experiment, "awar", "Woman1", "a")
set_node_state(experiment, "awar", "Bamatabois", "a")
set_node_state(experiment, "awar", "Thenardier", "a")
set_node_state(experiment, "awar", "Fantine", "a")
set_node_state(experiment, "awar", "Dahlia", "a")
set_node_state(experiment, "awar", "Listolier", "a")
set_node_state(experiment, "awar", "Isabeau", "a")

set_node_state(experiment, "vacc", "Napoleon", "v")
set_node_state(experiment, "vacc", "Myriel", "v")
set_node_state(experiment, "vacc", "MlleBaptistine", "v")
set_node_state(experiment, "vacc", "MmeMagloire", "v")
set_node_state(experiment, "vacc", "CountessDeLo", "v")
set_node_state(experiment, "vacc", "Geborand", "v")
set_node_state(experiment, "vacc", "Champtercier", "v")

logs = experiment.perform_propagation(n_epochs=30)
logs.report(to_file=True, path=output_dir, visualisation=True)
