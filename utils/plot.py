from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_population(population: np.array, winner_idx: int = None, seed: int = None, perplexity: int = 30) -> None:
    tsne = TSNE(n_components=2, verbose=0, random_state=seed, perplexity=perplexity)
    tsne_population = tsne.fit_transform(population)
    population_df = pd.DataFrame()
    population_df["x"] = tsne_population[:, 0]
    population_df["y"] = tsne_population[:, 1]
    if winner_idx is not None:
        population_df["is_winner"] = False
        population_df.iloc[winner_idx, population_df.columns.get_loc('is_winner')] = True
        sns.scatterplot(x="x", y="y", hue=population_df.is_winner.tolist(),
                        data=population_df, palette=['b', 'r']).set(title="Population Plot")
    else:
        sns.scatterplot(x="x", y="y", data=population_df).set(title="Archive Plot")
    plt.show()
