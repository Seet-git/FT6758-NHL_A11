import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# Convert clean_data tuple dictionaries into tuple dataframes
def convert_dictionaries_to_dataframes(data1: dict, data2: dict, years: list) -> pd.DataFrame:
    df_data1_all = pd.DataFrame()
    df_data2_all = pd.DataFrame()
    df_data12_all = pd.DataFrame()
    for year in set(years):
        df_data1 = pd.concat(data1[year])
        df_data2 = pd.concat(data2[year])
        df_data1.insert(0, 'Year', df_data1['idGame'].astype(str).str[:4])
        df_data2.insert(0, 'Year', df_data2['idGame'].astype(str).str[:4])
        df_data12 = pd.concat([df_data1, df_data2], axis=0)
        df_data1_all = pd.concat([df_data1_all, df_data1], axis=0)
        df_data2_all = pd.concat([df_data2_all, df_data2], axis=0)
        df_data12_all = pd.concat([df_data12_all, df_data12], axis=0)
    return df_data12_all


# Créer un tableau pivot pour la relation entre deux variables categorielles
def get_correlations_2variables(df, index, column):
    q = pd.crosstab(index=df[index], columns=df[column], margins=True, margins_name="Total")
    q[f"%{q.columns[0]}"] = round(q.iloc[:, 0] / q['Total'] * 100, 2)
    q.sort_values(by=[f"%{q.columns[0]}"], ascending=False, inplace=True)
    return q.fillna(0)


def get_correlations_3variables(df, index, column1, column2, column2_modality):
    q = pd.crosstab(index=df[index], columns=[df[column1], df[column2]]
                    # , margins=True, margins_name='Total'
                    )
    q = q.T.reset_index()
    q.fillna(0, inplace=True)
    q[f"{column1}_{column2}"] = q[column1].astype(str).fillna(0) + '_' + q[column2].astype(str).fillna(0)
    q = q.drop([column1, column2], axis=1)
    q = q.set_index(f"{column1}_{column2}")
    q = q.T
    q.columns.name = None
    q2 = {}
    for elem in df[column1][pd.notna(df[column1])].unique():
        q2[f"{elem}_Total"] = 0
        for elem2 in df[column2][pd.notna(df[column2])].unique():
            if elem2 == column2_modality:
                q2[f"{elem}_{elem2}"] = q[f"{elem}_{elem2}"]
            q2[f"{elem}_Total"] += q[f"{elem}_{elem2}"]
        q2[f"{elem}_{column2_modality}_%"] = round(q2[f"{elem}_{column2_modality}"] / q2[f"{elem}_Total"] * 100, 2)
    q2 = pd.DataFrame(q2)
    return q2


def plot_correlations_2variables(q, stat1, stat2):
    df = pd.DataFrame(q).drop('Total')
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Create bar plots for Total and goals
    bars_col1 = df[stat1].plot(kind='bar', color='orange', position=1, width=0.4, ax=ax1, label=stat1)
    bars_goal = df[stat2].plot(kind='bar', color='skyblue', position=1.5, width=0.2, ax=ax1, label=stat2)

    # Add titles and labels
    ax1.set_title(f"Graphe: Corrélation entre {stat2} et {stat1} par {df.index.name}")
    ax1.set_xlabel(f"{q.index.name}")
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(df.index, rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a second y-axis for % goal
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"{df.columns[-1]}", color='green')

    # Plot % goal on the secondary y-axis
    df[f"{df.columns[-1]}"].plot(kind='line', marker='o', color='green', ax=ax2, label=f"{df.columns[-1]}", linewidth=2)

    # Set y-axis limits for better visibility
    ax1.set_ylim(0, df[stat1].max() * 1.1)  # Adjust the limit to give space for bars
    ax2.set_ylim(0, df[f"{df.columns[-1]}"].max() * 1.2 if df[
                                                               f"{df.columns[-1]}"].max() * 1.5 < 100 else 100)  # Set limits for % goal

    legend1 = ax1.legend(loc='upper center')
    # Annotate values on the bars
    for index, value in enumerate(df[stat1]):
        ax1.text(index + 0.2, value + 1000, f"{value}", color='black', ha='center')

    for index, value in enumerate(df[stat2]):
        ax1.text(index - 0.2, value + 50, f"{value}", color='black', ha='center')

    # Annotate values on the line
    for index, value in enumerate(df[f"{df.columns[-1]}"]):
        ax2.text(index, value + 2, f"{round(value, 1):.1f}", color='green', ha='center')

    legend2 = plt.legend(loc='upper right')
    # fig.add_artist(legend1)
    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_graph_correlations(q, modality, df, column):
    plt.figure(figsize=(6, 4))
    column2_modality = 'goal'
    for elem in df[column].unique():
        data = q[q.columns[q.columns.str.startswith(str(elem))]]
        sns.lineplot(data=data, x=q.index.name, y=f'{elem}_{modality}_%', markers=True, dashes=False, label=elem)

    plt.title(f'Correlation entre {q.index.name} et {modality}, by {column}')
    plt.xlabel(f"{q.index.name}")
    plt.ylabel(f'%{modality}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_boxplot_correlations(q, var1, var2, category):
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))
    palette = {'goal': "#1f77b4", 'shot-on-goal': "#ff7f0e"}  # Adjust colors as needed

    sns.boxplot(data=q, x=var1, y=var2, hue=category, palette=palette, dodge=True)
    plt.title(f'Boxplot of {category} and {var2}, by {var1}')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend(title=f'{category} status', loc='upper right')
    plt.grid(True)
    plt.show()