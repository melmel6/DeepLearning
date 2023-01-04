# %% Libraries
import pandas as pd
import matplotlib.pyplot as plt

# %% Load data
path_to_losses = 'printlog_new.txt'
losses = pd.read_csv(path_to_losses, sep=' ')

# %% Subset
max_steps = 1 * 1e5
losses_subset = losses.loc[losses['step'] <= max_steps]

# %% Plot

transparency  = 1

# Plot 
plt.plot(losses_subset['step'], losses_subset['val_mae'], label='Validation MAE', c=plt.cm.Set2(0), alpha=transparency) 
plt.plot(losses_subset['step'], losses_subset['val_rmse'], label='Validation RMSE', c=plt.cm.Set2(1), alpha=transparency) 
plt.plot(losses_subset['step'], losses_subset['train_loss'], label='Train loss', c=plt.cm.Set2(2), alpha=transparency) 

# Title and labels
plt.legend()
plt.title('Training and validation loss')
plt.xlabel('Steps')
plt.ylabel('Loss')

# Save plot
plt.savefig('Loss.svg', bbox_inches='tight')    
plt.savefig('Loss.pdf', bbox_inches='tight')  
plt.savefig('Loss.png', dpi=300, bbox_inches='tight')  

plt.close()  

# %% Plot
min_steps = 2 * 1e4
losses_subset = losses.loc[losses['step'] >= min_steps]

transparency  = 1

# Plot 
plt.plot(losses_subset['step'], losses_subset['val_mae'], label='Validation MAE', c=plt.cm.Set2(0), alpha=transparency) 
plt.plot(losses_subset['step'], losses_subset['val_rmse'], label='Validation RMSE', c=plt.cm.Set2(1), alpha=transparency) 
plt.plot(losses_subset['step'], losses_subset['train_loss'], label='Train loss', c=plt.cm.Set2(2), alpha=transparency) 

# Title and labels
plt.legend()
plt.title('Training and validation loss')
plt.xlabel('Steps')
plt.ylabel('Loss')

# Save plot
plt.savefig('Loss_zoom.svg', bbox_inches='tight')    
plt.savefig('Loss_zoom.pdf', bbox_inches='tight')  
plt.savefig('Loss_zoom.png', dpi=300, bbox_inches='tight')  

plt.close()  
