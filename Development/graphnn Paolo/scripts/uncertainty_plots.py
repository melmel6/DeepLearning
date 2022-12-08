import matplotlib.pyplot as plt

def plot_epistemic(x='num_nodes', y='H', dictionary=input_dict, graph_output=graph_output, cmap=plt.cm.cool):
    aleatoric = graph_output[:,1]
    epistemic = graph_output[:,2]
    X = dictionary[x]
    Y = dictionary[y]
    plt.scatter(X, Y, s=200, c=epistemic, cmap=cmap)









# Plot uncertainty
print('_________UUUUUUNNNNNNNCCCCERRRTAINTYYYYYYYYYYYYYYYYYYYYY______________')
X = 'num_nodes'
Y = 'num_edges'

# Setup the normalization and the colormap
normalize = mcolors.Normalize(vmin=epistemic_uncertainty.detach().min(), vmax=epistemic_uncertainty.detach().max())
colormap = cm.cool

# Plot
plt.scatter(batch[X], batch[Y], s=150, c=epistemic_uncertainty.detach(), cmap='cool', alpha=0.4) 

# Setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(epistemic_uncertainty.detach())
plt.colorbar(scalarmappaple)

# Title and labels
plt.title('Epistemic uncertainty')
plt.xlabel(X)
plt.ylabel(Y)

# Save plot
plt.savefig('Epistemic.svg', bbox_inches='tight')    
plt.savefig('Epistemic.pdf', bbox_inches='tight')    
