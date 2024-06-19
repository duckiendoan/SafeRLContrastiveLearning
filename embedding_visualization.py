from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    X = np.load('runs/MiniGrid-LavaCrossingS9N1-v0_obs_embeddings.npy')
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(X_tsne.shape)
    print(tsne.kl_divergence_)
    df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    sns.scatterplot(data=df, x='TSNE1', y='TSNE2')
    plt.title('t-SNE visualization')
    plt.show()
