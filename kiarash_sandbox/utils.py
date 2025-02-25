import torch 
import numpy as np
import sklearn
import sklearn.manifold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
import scipy
############# VISUALIZATION UTILITY FUNCTIONS #############

def visualize_tsne(x, y_label, num_components=2, perplexity=5,
                   fig=None, ax=None, ax2=None, title_str=''):
  # perplexity = np.sqrt(n_points)) # TODO(loganesian): Evaluate default?
  tsne_tool = sklearn.manifold.TSNE(n_components=num_components, perplexity=perplexity)
  pca_tool = sklearn.decomposition.PCA(n_components=num_components)
  z_this = tsne_tool.fit_transform(x)
  pca_z_this = pca_tool.fit_transform(x)

  df = pd.DataFrame({'label' : y_label.flatten()})
  df = df.dropna(subset=['label'])
  df['row_id'] = range(0, len(df))
  df['z_1'], df['z_1_pca'] = z_this[:, 0], pca_z_this[:, 0]
  if num_components == 2:
    df['z_2'], df['z_2_pca'] = z_this[:, 1], pca_z_this[:, 1]
  n_cat = len(pd.unique(df['label']))

  if fig is None or ax is None:
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(121)
  if num_components == 2:
    scatter_ax = sns.scatterplot(x='z_1', y='z_2', data=df, hue='label', s=30,
                                 palette=sns.color_palette("Set2", n_cat))
  else:
    scatter_ax = sns.scatterplot(x='z_1', y='z_1', data=df, hue='label', s=30,
                                 palette=sns.color_palette("Set2", n_cat))
  scatter_ax.set(title=f'{title_str} t-SNE Perplexity: {perplexity}, Components: {num_components}')

  if ax2 is None:
      ax2 = fig.add_subplot(122)
  if num_components == 2:
    scatter_ax = sns.scatterplot(x='z_1_pca', y='z_2_pca', data=df, hue='label',
                                 s=30, palette=sns.color_palette("Set2", n_cat))
  else:
    scatter_ax = sns.scatterplot(x='z_1_pca', y='z_1_pca', data=df, hue='label',
                                 s=30, palette=sns.color_palette("Set2", n_cat))
  scatter_ax.set(title=f'{title_str} PCA Components: {num_components}')
  return fig, ax, ax2


def convert_py_conf_file_to_text(conf_file_name):
    """
    Read the lines of a .py configuration file into a list, skip the lines with comments in them.
    _____________________________________________________________________________________________
    Input parameters:
        
    conf_file_name: The name of the configuration file we want to read.
    
    _____________________________________________________________________________________________
    """
    
    lines = []
    multiline_comment = 0
    with open(conf_file_name) as f:
        for line in f:
            if len(line.rstrip()) > 0:
                if line.rstrip()[0] != '#':
                    if not multiline_comment:
                        if len(line.rstrip()) > 2:
                            if line.rstrip()[0:3] == '"""' or line.rstrip()[0:3] == "'''":
                                if line.rstrip().count('"""') == 1 or line.rstrip().count("'''") == 1:
                                    multiline_comment = 1
                            else:
                                lines.append(line.rstrip())
                        else:
                            lines.append(line.rstrip())
                    else:
                        if len(line.rstrip()) > 2:
                            if line.rstrip()[0:3] == '"""' or line.rstrip()[0:3] == "'''":
                                if line.rstrip().count('"""') == 1 or line.rstrip().count("'''") == 1:
                                    multiline_comment = 0
            
    return lines


def prepare_data(type='words'):
    if type == 'phonemes':
        data = scipy.io.loadmat('tuningTasks/t12.2022.04.21.mat')
    elif type == 'words':
        data = scipy.io.loadmat('tuningTasks/t12.2022.05.03_fiftyWordSet.mat')

    # mean-subtract within block
    def meanSubtract(dat, brainArea='6v'):
        if brainArea=='6v':
            dat['feat'] = np.concatenate([dat['tx2'][:,0:128].astype(np.float32), dat['spikePow'][:,0:128].astype(np.float32)], axis=1)
            # dat['feat'] = dat['tx2'][:,0:128].astype(np.float32)
        elif brainArea=='44':
            dat['feat'] = np.concatenate([dat['tx2'][:,128:].astype(np.float32), dat['spikePow'][:,128:].astype(np.float32)], axis=1)

        blockList = np.squeeze(np.unique(dat['blockNum']))
        for b in blockList:
            loopIdx = np.squeeze(dat['blockNum']==b)
            dat['feat'][loopIdx,:] -= np.mean(dat['feat'][loopIdx,:],axis=0,keepdims=True)
        return dat

    # mean subtraction (de-mean)
    data_6v = meanSubtract(data)

    # Sample piece: [sequence_length, channel_index]
    # NOTE: data_6v['feat'] are the continuous features
    # print(f"Feature length: {len(data_6v['feat'])}")

    feats = data_6v['feat'][:, 0:128]
    cueList = data_6v['cueList']
    trialCues = data_6v['trialCues']
    goTrialEpochs = data_6v['goTrialEpochs']

    max_length = max(end - start for start, end in goTrialEpochs)
    # Extract trial data based on goTrialEpochs
    trial_data = [feats[start:end] for start, end in goTrialEpochs]
    # Convert list to NumPy array (padding might be needed if trial lengths vary)
    padded_trials = np.array([np.pad(trial, ((0, max_length - len(trial)), (0, 0))) for trial in trial_data])

    # Example sigma for Gaussian smoothing (adjust as needed)
    sigma = 4  
    # Extract and smooth trials
    trial_data = [gaussian_filter1d(feats[start:end], sigma=sigma, axis=0) for start, end in goTrialEpochs]
    padded_trials_gs = np.array([np.pad(trial, ((0, max_length - len(trial)), (0, 0))) for trial in trial_data])
    window_start, window_end = 0, 101
    padded_trials_gs = padded_trials_gs[:, window_start:window_end]

    # Flatten labels for compatibility
    labels = trialCues.flatten()

    # CrossEntropyLoss requires labels starting from 0
    labels = labels - 1

    X_train, X_test, y_train, y_test = train_test_split(padded_trials_gs, labels, test_size=0.3, stratify=labels, random_state=30)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=30)

    return X_train, X_validation, X_test, y_train, y_validation, y_test