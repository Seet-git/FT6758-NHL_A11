import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_correlations_2variables(q, stat1, stat2):
    df = pd.DataFrame(q).drop('Total')
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Create bar plots for Total and goals
    bars_col1 = df[stat1].plot(kind='bar', color='orange', position=0.5, width=0.4, ax=ax1, label=stat1)
    bars_goal = df[stat2].plot(kind='bar', color='skyblue', position=0.5, width=0.2, ax=ax1, label=stat2)

    # Add titles and labels
    ax1.set_title(f"Graphe: Corrélation entre {stat2} et {stat1} par {df.index.name}", fontsize=14, weight='bold')
    ax1.set_xlabel(f"{q.index.name}", fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xticks(range(len(df.index)))
    ax1.set_xticklabels(df.index, rotation=45, ha='center', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a second y-axis for % goal
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"{df.columns[-1]}", color='green', fontsize=12)

    # Plot % goal on the secondary y-axis
    df[f"{df.columns[-1]}"].plot(kind='line', marker='o', color='green', ax=ax2, label=f"{df.columns[-1]}", linewidth=2)

    # Set y-axis limits for better visibility
    ax1.set_ylim(0, df[stat1].max() * 1.1)  # Adjust the limit to give space for bars
    ax2.set_ylim(0, df[f"{df.columns[-1]}"].max() * 1.2
    if df[f"{df.columns[-1]}"].max() * 1.5 < 100 else 100)  # Set limits for % goal

    # Annotate values on the bars
    for index, value in enumerate(df[stat1]):
        ax1.text(index, value + 1000, f"{value}", color='black', ha='center', fontsize=10)

    for index, value in enumerate(df[stat2]):
        ax1.text(index, value + 50, f"{value}", color='black', ha='center', fontsize=10)

    # Annotate values on the line
    for index, value in enumerate(df[f"{df.columns[-1]}"]):
        ax2.text(index, value + 2, f"{round(value, 1):.1f}", color='green', ha='center', fontsize=12)
    ax2.grid(False)
    # fig.add_artist(legend1)
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_graph_correlations(q, modality, df, column):
    plt.figure(figsize=(8, 4))
    columns = pd.Series(df[column].unique()).dropna()
    for elem in columns:
        data = q[q.columns[q.columns.str.startswith(str(elem))]]
        sns.lineplot(data=data, x=q.index.name, y=f'{elem}_{modality}_%', markers=True, dashes=False, label=elem)

    plt.title(f'Correlation entre {q.index.name} et {modality}, par {column}')
    plt.xlabel(f"{q.index.name}")
    plt.ylabel(f'%{modality}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_boxplot_correlations(q, var1, var2, category):
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))
    palette = {'goal': "#1f77b4", 'shot-on-goal': "#ff7f0e"}  # Adjust colors as needed

    sns.boxplot(data=q, x=var1, y=var2, hue=category, palette=palette, dodge=True)
    plt.title(f'Boxplot: {category} et {var2}, par {var1}')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend(title=f'{category} status', loc='upper right')
    plt.grid(True)
    plt.show()


def plot_goals_by_distance(data):
    data = data[data['typeDescKey'] == 'goal']

    bins = np.linspace(data['shotDistance'].min(), data['shotDistance'].max(), 10)  # 4 intervalles, 5 bords
    interval_labels = [f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)]
    interval_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    # Créer l'histogramme
    plt.figure(figsize=(10, 9))
    sns.histplot(
        data=data,
        x="shotDistance",
        hue="emptyGoalNet",
        bins=bins,
        multiple="dodge",
        shrink=0.8
    )

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(500))

    # Définir les étiquettes personnalisées de l'axe x
    plt.xticks(interval_centers, interval_labels)
    plt.ylabel('Nombre de buts')
    plt.xlabel("Distance au filet (en pieds)")
    #plt.legend(title="Filet vide")
    plt.legend(title="Filet vide", labels=['Oui', 'Non'])
    plt.title("Nombre de buts marqués entre les saisons 2016 et 2020 selon la distance au filet")

    # plt.savefig("C:/Users/franc/Documents/etudes univ/Udem/automne 2024/IFT6758 - Science de données/projet_NHL/IFT6758-NHL_A11/blogpost-template-main/public/goals_by_distance_milestone2.png",
    #           dpi=300, bbox_inches='tight')

    plt.show()

