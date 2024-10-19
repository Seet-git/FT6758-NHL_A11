import pandas as pd


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
        df_data1.insert(2, 'gameType', 'regular-season')
        df_data2.insert(2, 'gameType', 'playoffs')
        df_data12 = pd.concat([df_data1, df_data2], axis=0)
        df_data1_all = pd.concat([df_data1_all, df_data1], axis=0)
        df_data2_all = pd.concat([df_data2_all, df_data2], axis=0)
        df_data12_all = pd.concat([df_data12_all, df_data12], axis=0)
    return df_data12_all


# Créer un tableau pivot pour la relation entre deux catégories
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
