import os

from network_diffusion import (
    MultilayerNetwork,
    MultiSpreading,
    PropagationModel,
)

# initialise graph
network = MultilayerNetwork()
network.load_mlx("aux/florentine.mpx")
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

# initialise starting parameters of propagation in network
phenomenas = {"marriage": (12, 3), "business": (10, 1)}

# perform propagation experiment
experiment = MultiSpreading(model, network)
experiment.set_initial_states(phenomenas)
logs = experiment.perform_propagation(n_epochs=200)

# save experiment results
logs.report(to_file=True, path=os.getcwd(), visualisation=True)
