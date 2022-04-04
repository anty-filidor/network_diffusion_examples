
from network_diffusion import MultilayerNetwork, MultiSpreading, PropagationModel
from ndlib.models.epidemics import SIRModel
from ndlib.models.ModelConfig import Configuration

import networkx as nx
import numpy as np


def sir_ndlib(
        graph: nx.Graph,
        beta: float,
        gamma: float,
        fraction_infected: float,
        number_epochs: int
) -> None:
    """Perform SIR simulation using NDLIB."""

    # pass parameters to the configuration class
    config = Configuration()
    config.add_model_parameter("beta", beta)
    config.add_model_parameter("gamma", gamma)
    config.add_model_parameter("fraction_infected", fraction_infected)

    # initialise model
    model = SIRModel(graph)
    model.set_initial_status(config)

    # perform simulation and extract trends
    iterations = model.iteration_bunch(number_epochs)
    trends = model.build_trends(iterations)
    # from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
    # import matplotlib.pyplot as plt
    # viz = DiffusionTrend(model, trends)
    # viz.plot()
    # plt.show()


def sir_ndiff(
        graph: nx.Graph,
        beta: float,
        gamma: float,
        fraction_infected: float,
        number_epochs: int
) -> None:
    """Perform SIR simulation using Network Diffusion."""

    # initialise graph
    target_graph = MultilayerNetwork()
    target_graph.load_layer_nx(graph, ["ill"])

    # initialise model
    propagation_model = PropagationModel()
    propagation_model.add("ill", ["s", "i", "r"])
    propagation_model.compile(background_weight=0)
    propagation_model.set_transition_fast("ill.s", "ill.i", (), beta)
    propagation_model.set_transition_fast("ill.i", "ill.r", (), gamma)

    # configure experiment
    experiment = MultiSpreading(propagation_model, target_graph)
    num_nodes = len(target_graph.layers["ill"])
    num_infected = int(np.ceil(fraction_infected * num_nodes))
    num_recovered = max(num_infected // 10, 1)
    num_suspected = num_nodes - num_infected - num_recovered
    experiment.set_initial_states(
        {"ill": (num_suspected, num_infected, num_recovered)}
    )

    # perform simulation and extract trends
    logs = experiment.perform_propagation(number_epochs)
    # logs.plot()
