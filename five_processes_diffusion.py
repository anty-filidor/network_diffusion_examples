"""This example contains random calculation and is not reproducible."""
import configparser

from network_diffusion import MultiSpreading
from network_diffusion import MultilayerNetwork
from network_diffusion import PropagationModel

# read global config
config = configparser.ConfigParser()
config.read("config.ini")
output_dir = config.get("PATHS", "output_dir")

# initialise multilayer network from mlx file
network = MultilayerNetwork()
network.load_mlx("auxiliaries/aucs.mpx")
network.describe()

# initialise propagation model and set possible transitions with probabilities
model = PropagationModel()
phenomenas = [
    ("S", "I", "R"),
    ("UA", "A"),
    ("UV", "V"),
    ("S", "I", "R"),
    ("UV", "V"),
]
for l, p in zip(network.get_nodes_states(), phenomenas):
    model.add(l, p)
model.compile(background_weight=0.2)
model.set_transitions_in_random_edges(
    [[0.4, 0.5], [0.3, 0.2, 0.1], [0.9], [0.8, 0.6], [0.7]]
)
model.describe()

# initialise starting parameters of propagation in network
phenomenas = {
    "facebook": (10, 3, 19),
    "lunch": (59, 1),
    "coauthor": (23, 2),
    "leisure": (40, 5, 2),
    "work": (1, 59),
}

# perform propagation experiment
experiment = MultiSpreading(model, network)
experiment.set_initial_states(phenomenas)
logs = experiment.perform_propagation(200)
logs.report(to_file=True, path=output_dir, visualisation=True)
