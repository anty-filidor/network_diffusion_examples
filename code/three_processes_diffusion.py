import networkx as nx
from network_diffusion import (
    MultilayerNetwork,
    MultiSpreading,
    PropagationModel,
)

# initialise graph
network = MultilayerNetwork()
names = ["illness", "awareness", "vaccination"]
network.load_layer_nx(nx.les_miserables_graph(), names)

# initialise propagation model
model = PropagationModel()
phenomenas = [["S", "I", "R"], ["UA", "A"], ["UV", "V"]]
for l, p in zip(names, phenomenas):
    model.add(l, p)

# set possible transitions with weights
model.compile(background_weight=0.005)

model.set_transition_fast(
    "illness.S", "illness.I", ("vaccination.UV", "awareness.UA"), 0.4
)
model.set_transition_fast(
    "illness.S", "illness.I", ("vaccination.V", "awareness.A"), 0.05
)
model.set_transition_fast(
    "illness.S", "illness.I", ("vaccination.UV", "awareness.A"), 0.2
)
model.set_transition_fast(
    "illness.I", "illness.R", ("vaccination.UV", "awareness.UA"), 0.1
)
model.set_transition_fast(
    "illness.I", "illness.R", ("vaccination.V", "awareness.A"), 0.7
)
model.set_transition_canonical(
    "illness",
    (
        ("awareness.A", "illness.I", "vaccination.UV"),
        ("awareness.A", "illness.R", "vaccination.UV"),
    ),
    0.3,
)

model.set_transition_fast(
    "vaccination.UV", "vaccination.V", ("awareness.A", "illness.S"), 0.03
)
model.set_transition_fast(
    "vaccination.UV", "vaccination.V", ("awareness.A", "illness.I"), 0.01
)

model.set_transition_fast(
    "awareness.UA", "awareness.A", ("vaccination.UV", "illness.S"), 0.05
)
model.set_transition_fast(
    "awareness.UA", "awareness.A", ("vaccination.V", "illness.S"), 1
)
model.set_transition_fast(
    "awareness.UA", "awareness.A", ("vaccination.UV", "illness.I"), 0.2
)


# initialise starting parameters of propagation in network
phenomenas = {
    "illness": (65, 10, 2),
    "awareness": (60, 17),
    "vaccination": (70, 7),
}

# perform propagation experiment
experiment = MultiSpreading(model, network)
experiment.set_initial_states(phenomenas)
logs = experiment.perform_propagation(n_epochs=50)

# save experiment results
logs.report(to_file=False, path=None, visualisation=True)
