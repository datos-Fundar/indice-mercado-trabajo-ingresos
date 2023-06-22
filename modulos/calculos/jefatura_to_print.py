# %%
%run /home/daniufundar/Documents/Fundar/indice-mercado-trabajo-ingresos/modulos/funciones/01_funciones_insercion_laboral.ipynb
#%run /Users/danielarisaro/Documents/Fundar/indice-mercado-trabajo-ingresos/modulos/funciones/01_funciones_insercion_laboral.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore')

# %%
# Definimos path
pathdata = '/home/daniu/Documentos/fundar/indice-mercado-trabajo-ingresos/'
pathdata = '/Users/danielarisaro/Documents/Fundar/indice-mercado-trabajo-ingresos/'
pathdata = '/home/daniufundar/Documents/Fundar/indice-mercado-trabajo-ingresos/'

# %%
with open(pathdata + 'modulos/diccionarios/' + 'diccionario_aglomerados.pickle', 'rb') as file:
    dict_cod_aglomerado = pickle.load(file)

with open(pathdata + 'modulos/diccionarios/' + 'diccionario_provincia.pickle', 'rb') as file:
    dict_cod_provincia = pickle.load(file)

with open(pathdata + 'modulos/diccionarios/' + 'diccionario_aglomerado_provincia.pickle', 'rb') as file:
    map_aglomerado_provincia = pickle.load(file)

with open(pathdata + 'modulos/diccionarios/' + 'diccionario_aglomerado_region.pickle', 'rb') as file:
    map_aglomerado_region = pickle.load(file)

with open(pathdata + 'modulos/diccionarios/' + 'diccionario_provincia_region.pickle', 'rb') as file:
    map_provincia_region = pickle.load(file)

# %%
df_people_pool = pd.read_csv(pathdata + 'data_output/Base_pool_individuos_solo_con_replicas_actuales.csv', low_memory=False, index_col=0)
df_houses_pool = pd.read_csv(pathdata + 'data_output/Base_pool_hogares_solo_con_replicas_actuales.csv', low_memory=False, index_col=0)


# %%
df_CBT = pd.read_csv(pathdata + 'data_output/Canasta_Basica_Total_Regiones_2016-2022-promedios-moviles.csv', delimiter=',', header=0, index_col=[0])
df_adultos_equiv = pd.read_csv(pathdata + 'data_input/canastas_basicas/adultos_equivalente.csv')

# %%
def capitalize_first_letter(s):
    return s.capitalize()

df_CBT = df_CBT.rename(columns=capitalize_first_letter)
df_CBT.rename(columns={'Gran_buenos_aires':'Gran Buenos Aires'}, inplace=True)

# %%
# Filter and merge data
df_temp = df_houses_pool.loc[(df_houses_pool['IX_TOT']>1) & (df_houses_pool['REALIZADA']==1) & (~df_houses_pool['NRO_HOGAR'].isin([51, 71]))]
merged_df_pool = df_people_pool[['CODUSU', 'NRO_HOGAR']].merge(df_houses_pool[['CODUSU', 'NRO_HOGAR']], on=['CODUSU', 'NRO_HOGAR'], how='outer', indicator=True)


# %%
# Create a dictionary with the information
duplicated_rows = merged_df_pool['_merge'].value_counts()['both']
rows_people = len(df_people_pool)
hogares_unicos = merged_df_pool[['CODUSU', 'NRO_HOGAR']].value_counts()
count_houses = len(hogares_unicos)
count_viviendas = len(df_temp['CODUSU'].value_counts())
data = {
    'Description': ['Duplicated Rows', 'Total People Rows', 'Total Houses', 'Total Dwellings'],
    'Count': [duplicated_rows, rows_people, count_houses, count_viviendas],
    'Comments': ['Si esta cantidad es la misma a Total People Rows, entonces todos los hogares fueron relevados en la base individual tambien.',
                    'Número de individuos en la encuesta de personas.',
                    'Número de hogares en la encuesta de hogares.',
                    'Número de viviendas únicas']
}

df_counts = pd.DataFrame(data)
df_counts

# %%
def calculate_equivalent_adults(grouped_df, df_adultos_equiv):
    grouped_df['EQUIVALENT_ADULTS'] = 0
    for index, row in grouped_df.iterrows():
        total_equivalent_adults = 0
        for age, gender in zip(row['CH06'], row['CH04']):
            equiv_adults_row = df_adultos_equiv[(df_adultos_equiv['Edad inferior'] <= age) & (df_adultos_equiv['Edad superior'] > age)]
            if not equiv_adults_row.empty:
                equiv_adults = equiv_adults_row.iloc[0]['Mujer'] if gender == 2 else equiv_adults_row.iloc[0]['Varon']
                total_equivalent_adults += equiv_adults
        grouped_df.at[index, 'EQUIVALENT_ADULTS'] = total_equivalent_adults

def merge_and_add_columns(grouped_df, df_houses):
    custom_merged_df = grouped_df.merge(df_houses, on=['CODUSU', 'NRO_HOGAR'], how='left')
    grouped_df['INCOME'] = custom_merged_df['ITF']
    grouped_df['INCOME_PONDERATOR'] = custom_merged_df['PONDIH_new']
    grouped_df['PROVINCIA'] = custom_merged_df['PROVINCIA']
    grouped_df['AGLOMERADO'] = custom_merged_df['AGLOMERADO']

def reshape_and_filter_data(grouped_df, df_CBT, dict_cod_provincia, dict_cod_aglomerado, map_aglomerado_region):
    melted_df_CBT = df_CBT.melt(id_vars='Trimestre', var_name='Region', value_name='CBT')
    filtered_melted_df_CBT = melted_df_CBT[melted_df_CBT['Trimestre'] == '4T2022']
    grouped_df['Provincia'] = grouped_df['PROVINCIA'].map(dict_cod_provincia)
    grouped_df['Aglomerado'] = grouped_df['AGLOMERADO'].map(dict_cod_aglomerado)
    grouped_df['Region'] = grouped_df['Aglomerado'].map(map_aglomerado_region)
    map_region_CBT = filtered_melted_df_CBT.set_index('Region')['CBT'].to_dict()
    grouped_df['CBT'] = grouped_df['Region'].map(map_region_CBT)
    grouped_df['THRESHOLD'] = grouped_df['CBT'] * grouped_df['EQUIVALENT_ADULTS']

def calculate_poverty_table(df_people, df_houses, df_CBT, df_adultos_equiv, dict_cod_provincia, map_provincia_region):
    grouped_df = df_people.groupby(['CODUSU', 'NRO_HOGAR']).agg({
        'CH03': list,
        'COMPONENTE': list,
        'CH04': list,
        'CH06': list
    })
    
#    grouped_df = grouped_df[grouped_df['COMPONENTE'].apply(lambda x: 2 not in x)]
    grouped_df['NUM_PEOPLE_IN_HOUSE'] = grouped_df['CH06'].apply(len)
    grouped_df['MEAN_AGE'] = grouped_df['CH06'].apply(lambda ages: sum(ages) / len(ages) if ages else None)
    grouped_df = grouped_df[grouped_df['CH06'].apply(lambda x: any(age < 25 for age in x))]
    grouped_df['GENDER_PERSON_IN_CHARGE'] = grouped_df.apply(lambda row: 'MALE' if 1 in row['CH03'] and row['CH04'][row['CH03'].index(1)] == 1 else 'FEMALE', axis=1)
    grouped_df['PERSON_IN_CHARGE_AGE'] = grouped_df.apply(lambda row: row['CH06'][row['CH03'].index(1)] if 1 in row['CH03'] else None, axis=1)
    grouped_df = grouped_df.reset_index()

    calculate_equivalent_adults(grouped_df, df_adultos_equiv)
    merge_and_add_columns(grouped_df, df_houses)
    reshape_and_filter_data(grouped_df, df_CBT, dict_cod_provincia, dict_cod_aglomerado, map_aglomerado_region)

    return grouped_df


# %%
grouped_df = calculate_poverty_table(df_people_pool, df_houses_pool, df_CBT, df_adultos_equiv, dict_cod_provincia, map_provincia_region)
#hogares_no_pobres = grouped_df[grouped_df['INCOME'] >= grouped_df['THRESHOLD']]
#agrupado = hogares_no_pobres.groupby(['GENDER_PERSON_IN_CHARGE','Provincia'])['INCOME_PONDERATOR'].sum().unstack(level=0)

hogares_pobres = grouped_df[grouped_df['INCOME'] < grouped_df['THRESHOLD']]
agrupado = hogares_pobres.groupby(['GENDER_PERSON_IN_CHARGE','Provincia'])['INCOME_PONDERATOR'].sum().unstack(level=0)


gender_counts = grouped_df.groupby(['Provincia', 'GENDER_PERSON_IN_CHARGE'])['INCOME_PONDERATOR'].sum().unstack()

fraccion = gender_counts / agrupado

row_counts = grouped_df.groupby(['Provincia', 'GENDER_PERSON_IN_CHARGE']).size().unstack()

gender_counts['ROW_COUNTS_FEMALE'] = row_counts['FEMALE']
gender_counts['ROW_COUNTS_MALE'] = row_counts['MALE']

#gender_ratios_pool = fraccion['FEMALE'] / fraccion['MALE']
gender_ratios_pool = fraccion['MALE'] / fraccion['FEMALE']



# %%
grouped_df.to_excel(pathdata + "jefatura.xlsx")  

# %%
tasa = fraccion
tasa.columns = ['Mujer', 'Varon']

size = row_counts
size.columns = ['N_m', 'N_v']

ratio = gender_ratios_pool.to_frame()
ratio.rename(columns={0: 'No-Pobreza'}, inplace=True)

error = pd.DataFrame(index=size.index, columns=['LI', 'LS', 'ME', 'ER'])

# %%
#hogares_no_pobres_jefatura = pd.concat([tasa*100, size, ratio*100, error*100], axis=1)
hogares_pobres_jefatura = pd.concat([tasa*100, size, ratio*100, error*100], axis=1)


# %%
#hogares_no_pobres_jefatura.to_pickle(pathdata + 'data_output/df_hogares_no_pobres_pool.pickle')
hogares_pobres_jefatura.to_pickle(pathdata + 'data_output/df_hogares_no_pobres_pool.pickle')

# %%
# Definimos colores fundar
fundar_colores_primarios = ['#7BB5C4', '#9FC1AD', '#D3D3E0',  '#8d9bff', '#FF9750', '#FFD900',]
fundar_colores_secundarios = ['#B5E0EA', '#B3B3B3', '#848279', '#AFA36E', '#5D896F', '#9C9CBC', '#E27124']
fundar_white = '#F5F5F5'
fundar_black = '#151515'

# %%
import plotly.graph_objects as go

bar_trace = go.Bar(
    x=hogares_pobres_jefatura.index,
    y=hogares_pobres_jefatura['No-Pobreza'],
    name='Autonomia economica',
    marker_color=fundar_colores_primarios[3],
    hovertemplate='No-Pobreza: %{y:.2f}<extra></extra>'
)

layout = go.Layout(
    title='Indicador pobreza',
    barmode='group',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(x=0.6, y=1.1, orientation='h'),
    yaxis_title="Ratio"

)

fig1 = go.Figure(data=[bar_trace], layout=layout)

fig1.add_shape(
    type='line',
    x0=hogares_pobres_jefatura.index[0],
    y0=100,
    x1=hogares_pobres_jefatura.index[-1],
    y1=100,
    line=dict(
        color='black',
        width=1,
        dash='dash'
    ),
)

fig1.update_yaxes(range=[0, 180])
file_path = pathdata + 'figs/'
filename = 'panel-pobreza-pool.html'
fig1.write_html(file_path + filename)




