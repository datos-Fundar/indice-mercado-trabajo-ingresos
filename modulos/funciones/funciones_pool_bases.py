import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_two_years(year1, year2, full_path, type='Aglomerado', base='Individual'):
    
    if type == 'Aglomerado':
        filename = 'tot_urb_3T'
    elif type == 'Urbano':
        filename = 'tot_urb_3T'
            
    df_year1 = pd.read_csv(full_path + filename, delimiter=';', low_memory=False)
    df_year2 = pd.read_csv(full_path + filename, delimiter=';', low_memory=False)

    return


def merge_and_calculate_people(df_year1, df_year2, year1, year2, conversion_factor):

    """
    Combina dos DataFrames de dos anios de la base de personas de la EPH utilizando las columnas 'CODUSU' y 'COMPONENTE'. Realiza cálculos en los datos fusionados.

    Args:
        df_year1 (pandas.DataFrame): DataFrame para el año 1.
        df_year2 (pandas.DataFrame): DataFrame para el año 2.
        year1 (int): Año de los datos de df_year1.
        year2 (int): Año de los datos de df_year2.
        conversion_factor (float): Factor de conversión para la transformación de datos.

    Returns:
        pandas.DataFrame: DataFrame fusionado con los valores calculados. Base pool de base personas sin replicas.

    """
        
    merged_df = pd.merge(df_year1[['CODUSU', 'COMPONENTE']], df_year2[['CODUSU', 'COMPONENTE']],
                         on=['CODUSU', 'COMPONENTE'], how='outer', indicator=True)

    left_only_rows = merged_df[merged_df['_merge'] == 'left_only']
    right_only_rows = merged_df[merged_df['_merge'] == 'right_only']
    both_rows = merged_df[merged_df['_merge'] == 'both']
    right_both_rows = pd.concat([both_rows, right_only_rows])

    columns = ['PONDERA', 'PONDII', 'PONDIIO']
    df_years = [df_year1, df_year2]

    for col in columns:
        for df in df_years:
            df[f'rel_{col}'] = df[col] / df[col].sum()

    df_year1_no_dupl = pd.merge(left_only_rows, df_year1, on=["CODUSU", "COMPONENTE"], how='inner')
    df_year2_no_dupl = pd.merge(right_both_rows, df_year2, on=["CODUSU", "COMPONENTE"], how='inner')

    new_columns = ['P21_new', 'P47T_new', 'TOT_P12_new']
    df_merged_without_copies = pd.concat([df_year1_no_dupl, df_year2_no_dupl])
    df_merged_without_copies[new_columns] = pd.NaT

    mask_year1 = df_merged_without_copies['ANO4'] == year1
    mask_year2 = df_merged_without_copies['ANO4'] == year2

    for col in ['P21', 'P47T', 'TOT_P12']:
        df_merged_without_copies.loc[mask_year1 & (df_merged_without_copies[col] != -9), f'{col}_new'] = \
            df_merged_without_copies.loc[mask_year1 & (df_merged_without_copies[col] != -9), col] / conversion_factor
        df_merged_without_copies.loc[mask_year1 & (df_merged_without_copies[col] == -9), f'{col}_new'] = \
            df_merged_without_copies.loc[mask_year1 & (df_merged_without_copies[col] == -9), col]

    for col in ['P21', 'P47T', 'TOT_P12']:
        df_merged_without_copies.loc[mask_year2, f'{col}_new'] = df_merged_without_copies.loc[mask_year2, col]

    for col in ['PONDERA', 'PONDII', 'PONDIIO']:
        temp_rel = df_merged_without_copies[f'rel_{col}'].div(df_merged_without_copies[f'rel_{col}'].sum())
        pob_prom = (df_year1[col].sum() + df_year2[col].sum()) / 2
        df_merged_without_copies[f'{col}_new'] = temp_rel * pob_prom

    return df_merged_without_copies

def merge_and_calculate_houses(df_year1, df_year2, year1, year2, conversion_factor):

    """
    Combina dos DataFrames de dos anios de la base de hogares de la EPH utilizando las columnas 'CODUSU' y 'NRO_HOGAR'. Realiza cálculos en los datos fusionados.

    Args:
        df_year1 (pandas.DataFrame): DataFrame para el año 1.
        df_year2 (pandas.DataFrame): DataFrame para el año 2.
        year1 (int): Año de los datos de df_year1.
        year2 (int): Año de los datos de df_year2.
        conversion_factor (float): Factor de conversión para la transformación de datos.

    Returns:
        pandas.DataFrame: DataFrame fusionado con los valores calculados. Base pool de base hogares sin replicas.

    """
        
    merged_df = pd.merge(df_year1[['CODUSU', 'NRO_HOGAR']], df_year2[['CODUSU', 'NRO_HOGAR']],
                         on=['CODUSU', 'NRO_HOGAR'], how='outer', indicator=True)

    left_only_rows = merged_df[merged_df['_merge'] == 'left_only']
    right_only_rows = merged_df[merged_df['_merge'] == 'right_only']
    both_rows = merged_df[merged_df['_merge'] == 'both']
    right_both_rows = pd.concat([both_rows, right_only_rows])

    columns = ['PONDERA', 'PONDIH']
    df_years = [df_year1, df_year2]

    for col in columns:
        for df in df_years:
            df[f'rel_{col}'] = df[col] / df[col].sum()

    df_year1_no_dupl = pd.merge(left_only_rows, df_year1, on=["CODUSU", "NRO_HOGAR"], how='inner')
    df_year2_no_dupl = pd.merge(right_both_rows, df_year2, on=["CODUSU", "NRO_HOGAR"], how='inner')

    new_columns = ['ITF_new']
    df_merged_without_copies = pd.concat([df_year1_no_dupl, df_year2_no_dupl])
    df_merged_without_copies[new_columns] = pd.NaT

    mask_year1 = df_merged_without_copies['ANO4'] == year1
    mask_year2 = df_merged_without_copies['ANO4'] == year2

    for col in ['ITF']:
        df_merged_without_copies.loc[mask_year1 & (df_merged_without_copies[col] != -9), f'{col}_new'] = \
            df_merged_without_copies.loc[mask_year1 & (df_merged_without_copies[col] != -9), col] / conversion_factor
        df_merged_without_copies.loc[mask_year1 & (df_merged_without_copies[col] == -9), f'{col}_new'] = \
            df_merged_without_copies.loc[mask_year1 & (df_merged_without_copies[col] == -9), col]

    for col in ['ITF']:
        df_merged_without_copies.loc[mask_year2, f'{col}_new'] = df_merged_without_copies.loc[mask_year2, col]

    for col in ['PONDERA', 'PONDIH']:
        temp_rel = df_merged_without_copies[f'rel_{col}'].div(df_merged_without_copies[f'rel_{col}'].sum())
        pob_prom = (df_year1[col].sum() + df_year2[col].sum()) / 2
        df_merged_without_copies[f'{col}_new'] = temp_rel * pob_prom

    return df_merged_without_copies
