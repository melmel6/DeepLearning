# %% Libraries
import matplotlib.pyplot as plt
import json
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from continuous_loss_pytorch import EvidentialRegression
import torch

# %% Load data
# =============================================================================
# # Load graph_data from paolinos
# data = {'num_nodes':[], 'H_':[], 'G_': []}
# for i in range(4184):
#     path_to_data = 'paolinos/paolino_%d' %i
#     f = open(path_to_data) 
#     data_i = json.load(f) 
#     data['num_nodes'] += data_i['num_nodes']
#     data['H_'] += data_i['H_']
#     data['G_'] += data_i['targets']
# del(i)
#     
# with open('graph_data.json', 'w') as f:
#     json.dump(data, f)
# =============================================================================

# Load graph_data (whole)
path_to_data = 'graph_data.json'
f = open(path_to_data)
data = json.load(f)

# Load predictions
path_to_pred = 'predictions_key.txt'
predictions = pd.read_csv(path_to_pred, sep=' ')

# %% Parameters
# Calculate uncertainty
v = predictions['v']
alpha = predictions['alpha']
beta = predictions['beta']

epistemic = (beta/(v * (alpha - 1))).apply(np.sqrt)
aleatoric = (beta/(alpha - 1)).apply(np.sqrt)

epistemic_cutoff = 300
aleatoric_cutoff = 0.5
visualized_epistemic = [epistemic_cutoff if i >= epistemic_cutoff else i for i in epistemic]
visualized_aleatoric = [aleatoric_cutoff if i >= aleatoric_cutoff else i for i in aleatoric]


# Plot parameters
X = 'num_nodes'
Y = 'H_'
X_threshold = 16
Y_threshold = -98
dot_size = 0.2
transparency = 0.8
linewidth = 0.5

epistemic_color = visualized_epistemic
aleatoric_color = visualized_aleatoric
# %% EPISTEMIC
# Setup the normalization and the colormap
normalize = mcolors.Normalize(vmin=min(epistemic_color), vmax=max(epistemic_color))
colormap = cm.cool

# Plot 
plt.scatter(data[X], data[Y], s=dot_size, c=epistemic_color, cmap='cool', alpha=transparency) 
plt.plot((X_threshold, 30), (Y_threshold, Y_threshold), 'k--', linewidth=linewidth)  # Horizontal threshold
plt.plot((X_threshold, X_threshold), (0, Y_threshold), 'k--', linewidth=linewidth)  # Vertical threshold

# Setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(epistemic)
plt.colorbar(scalarmappaple)

# Title and labels
plt.title('Epistemic uncertainty')
plt.xlabel('Number of atoms')
plt.ylabel('Enthalpy')

# Save plot
plt.savefig('Epistemic.svg', bbox_inches='tight')    
plt.savefig('Epistemic.pdf', bbox_inches='tight')  
plt.savefig('Epistemic.png', dpi=300, bbox_inches='tight')  

plt.close()  

# %% ALEATORIC
# Setup the normalization and the colormap
normalize = mcolors.Normalize(vmin=min(aleatoric_color), vmax=max(aleatoric_color))
colormap = cm.cool

# Plot 
plt.scatter(data[X], data[Y], s=dot_size, c=aleatoric_color, cmap='cool', alpha=transparency) 
plt.plot((X_threshold, 30), (Y_threshold, Y_threshold), 'k-', linewidth=linewidth)  # Horizontal threshold
plt.plot((X_threshold, X_threshold), (0, Y_threshold), 'k-', linewidth=linewidth) # Vertical threshold

# Setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(aleatoric)
plt.colorbar(scalarmappaple)

# Title and labels
plt.title('Aleatoric uncertainty')
plt.xlabel('Number of atoms')
plt.ylabel('Enthalpy')

# Save plot
plt.savefig('Aleatoric.svg', bbox_inches='tight')    
plt.savefig('Aleatoric.pdf', bbox_inches='tight')  
plt.savefig('Aleatoric.png', dpi=300, bbox_inches='tight')  

plt.close()  

# %% Loss In-distribution VS OOD

# Dict to dataframe
data_df = pd.DataFrame.from_dict(data)
data_df = pd.concat([data_df, predictions], axis=1)
data_df_ID = data_df.loc[(data_df['num_nodes'] > 16) & (data_df['H_'] > -98)]
data_df_OOD = data_df.loc[(data_df['num_nodes'] <= 16) | (data_df['H_'] <= -98)]

# Define ID and OOT (tensor)
G_ID = data_df_ID['G_'].tolist() 
G_OOD = data_df_OOD['G_'].tolist() 

pred_ID = data_df_ID[['gamma', 'v', 'alpha', 'beta']]
pred_OOD = data_df_OOD[['gamma', 'v', 'alpha', 'beta']]

pred_ID_torch = torch.from_numpy(pred_ID.values)
pred_OOD_torch = torch.from_numpy(pred_OOD.values)
G_torch_ID = torch.FloatTensor(G_ID)
G_torch_OOD = torch.FloatTensor(G_OOD)

# Calculate loss
ID_loss = EvidentialRegression(y=G_torch_ID, evidential_output=pred_ID_torch, lmbda=1e-3).numpy()
OOD_loss = EvidentialRegression(y=G_torch_OOD, evidential_output=pred_OOD_torch, lmbda=1e-3).numpy()


print('ID_loss')
print(ID_loss)
print('OOD_loss')
print(OOD_loss)
