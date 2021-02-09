import matplotlib.pyplot as plt
import networkx as nx
from network_diffusion import FlatSpreading

# initialise graph
M = nx.les_miserables_graph()

# perform propagation
list_S, list_I, list_iter, nodes_infected, par = FlatSpreading.si_diffusion(
    M, fract_I=0.05, beta_coeff=0.2, name="Les_miserables_V_Hugo_graph"
)

# plot bulk chart for experiment
fig, ax = plt.subplots(1)
plt.plot(list_iter, list_S, label="suspected")
plt.plot(list_iter, list_I, label="infected")
plt.title("SI diffusion")
plt.legend()
plt.grid()
plt.savefig("{}.png".format(par[0]), dpi=150)
plt.show()

# prepare animated visualisations of experiment
FlatSpreading.visualise_si_nodes(M, nodes_infected, par, "/results")
FlatSpreading.visualise_si_nodes_edges(M, nodes_infected, par, "/results")
