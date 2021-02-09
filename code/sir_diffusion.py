import matplotlib.pyplot as plt
from network_diffusion import FlatSpreading as fs
import networkx as nx

# initialise graph
M = nx.barabasi_albert_graph(200, 50)

# perform propagation
(
    list_S,
    list_I,
    list_R,
    list_iter,
    nodes_infected,
    nodes_recovered,
    par,
) = fs.sir_diffusion(
    M,
    fract_I=0.08,
    beta_coeff=0.2,
    gamma_coeff=0.2,
    name="Barabasi_Albert_graph",
)

# plot bulk chart for experiment
plt.plot(list_iter, list_S, label="suspected")
plt.plot(list_iter, list_I, label="infected")
plt.plot(list_iter, list_R, label="recovered")
plt.title("SIR diffusion")
plt.legend()
plt.grid()
plt.savefig("{}.png".format(par[0]), dpi=150)
plt.show()

# prepare animated visualisations of experiment
fs.visualise_sir_nodes(
    M, nodes_infected, nodes_recovered, par, "/results"
)
fs.visualise_sir_nodes_edges(
    M, nodes_infected, nodes_recovered, par, "/results"
)
