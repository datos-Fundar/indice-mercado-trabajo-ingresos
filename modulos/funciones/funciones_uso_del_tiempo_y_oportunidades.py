import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def poblacion_inactiva_sin_ingresos(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):

    # En este caso dividir el indicador segun algun percentil de ingresos. Por ejemplo de 90 a 60. Ver literatura al respecto 

    """
    INPUTS
    df: DataFrame. Tabla input EPH
    tipo: string. Tipo de encuesta de la EPH, Aglomerado o Urbano. Default Aglomerado
    base: string. Tipo de base de la encuesta de la EPH, Individual u Hogar. Default Individual

    OUTPUTS
    ratio: DataFrame. Tabla con Ratios en tasa M/V de población inactiva que no estudia y no tiene ingresos propios, desagregado por Aglomerado o Provincia
    error: DataFrame. Tabla con los errores asociados a los Ratios

    """

    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
    
    if pool:
        pondera = 'PONDII_new'
        p47t = 'P47T_new'
    else:
        pondera = 'PONDII'
        p47t = 'P47T'

    df_inactivos = df_temp[(df_temp[p47t]!=-9) & (df_temp['ESTADO']==3) & (df_temp['CAT_INAC']!=3)][[var, pondera, p47t, 'CH04']]
    numerador = df_inactivos[(df_inactivos[p47t]==0)].groupby(['CH04', var])[pondera].sum().unstack(level=0)
    denominador = df_inactivos.groupby(['CH04', var])[pondera].sum().unstack(level=0)

    tasa = numerador.div(denominador, fill_value=np.nan)
    tasa.columns = ['Varon', 'Mujer']

    size = df_inactivos[(df_inactivos[p47t]==0)].groupby(['CH04', var]).size().to_frame().unstack(level=0)
    size.columns = ['N_v', 'N_m']

    p1 = tasa['Varon']
    p2 = tasa['Mujer']
    n1 = size['N_v']
    n2 = size['N_m']
    ratio = (p1 / p2).to_frame()
    ratio.rename(columns={0: 'Dependencia'}, inplace=True)
    
    # Calculate the standard error of the proportion ratio
    standard_error = np.sqrt(((1 / n1) * (p1 * (1 - p1))) + ((1 / n2) * (p2 * (1 - p2))))

    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z * standard_error
    lower_bound = p1/p2 - margin_of_error
    upper_bound = p1/p2 + margin_of_error

    relative_standard_error = margin_of_error/(p1/p2)
    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa*100, size, ratio*100, error*100


def poblacion_inactiva_con_ingresos(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):

    """
    INPUTS
    df: DataFrame. Tabla input EPH
    tipo: string. Tipo de encuesta de la EPH, Aglomerado o Urbano. Default Aglomerado
    base: string. Tipo de base de la encuesta de la EPH, Individual u Hogar. Default Individual

    OUTPUTS
    ratio: DataFrame. Tabla con Ratios en tasa M/V de población inactiva que no estudia y no tiene ingresos propios, desagregado por Aglomerado o Provincia
    error: DataFrame. Tabla con los errores asociados a los Ratios

    """

    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
    
    if pool:
        pondera = 'PONDII_new'
        p47t = 'P47T_new'
    else:
        pondera = 'PONDII'
        p47t = 'P47T'

    df_inactivos = df_temp[(df_temp[p47t]!=-9) & (df_temp['ESTADO']==3) & (df_temp['CAT_INAC']!=3)][[var, pondera, p47t, 'CH04']]
    numerador = df_inactivos[(df_inactivos[p47t]>0)].groupby(['CH04', var])[pondera].sum().unstack(level=0)
    denominador = df_inactivos.groupby(['CH04', var])[pondera].sum().unstack(level=0)

    tasa = numerador.div(denominador, fill_value=np.nan)
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_inactivos[(df_inactivos[p47t]>0)].groupby(['CH04', var]).size().unstack(level=0)
    size = size.rename(columns= {1:'N_v', 2:'N_m'})

    n_pob = numerador.copy()
    n_pob = n_pob.rename(columns= {1:'N_pob_v', 2:'N_pob_m'})

    size[['N_pob_v', 'N_pob_m']] = n_pob[['N_pob_v', 'N_pob_m']]
    size['N_pob_tot'] = n_pob['N_pob_v'] + n_pob['N_pob_m']

    p1 = tasa['Mujer']
    p2 = tasa['Varon']
    n1 = size['N_m']
    n2 = size['N_v']
    ratio = (p1 / p2).to_frame()
    ratio.rename(columns={0: 'Dependencia'}, inplace=True)
    
    # Calculate the standard error of the proportion ratio
    standard_error = np.sqrt(((1 / n1) * (p1 * (1 - p1))) + ((1 / n2) * (p2 * (1 - p2))))

    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z * standard_error
    lower_bound = p1/p2 - margin_of_error
    upper_bound = p1/p2 + margin_of_error

    relative_standard_error = margin_of_error/(p1/p2)
    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa*100, size, ratio*100, error*100


def tabla_ingresos_inactivos_no_estudiantes(df, pool=False):
    
    if pool:
        pondera = 'PONDII_new'
    else:
        pondera = 'PONDII'    
    
    df_temp = df[df['CH06'] >= 14].copy()
    bins = [14, 30, 46, 66, df_temp['CH06'].max()]
    labels = ['14-29', '30-45', '46-65', '65+']
    df_temp['Rango edad'] = pd.cut(df_temp['CH06'], bins=bins, labels=labels, right=False)

    pondera = 'PONDERA_new'
    p47t = 'P47T_new'

    df_inactivos_con_ingreso = df_temp[(df_temp['ESTADO']==3) & (df_temp[p47t]==0) & (df_temp['CAT_INAC']!=3)].groupby(['CH04','Rango edad'])[pondera].sum().unstack(level=0)
    df_inactivos_sin_ingreso = df_temp[(df_temp['ESTADO']==3) & (df_temp[p47t]>0) & (df_temp['CAT_INAC']!=3)].groupby(['CH04','Rango edad'])[pondera].sum().unstack(level=0)
    df_inactivos_no_declara_ingreso = df_temp[(df_temp['ESTADO']==3) & (df_temp[p47t]==-9) & (df_temp['CAT_INAC']!=3)].groupby(['CH04','Rango edad'])[pondera].sum().unstack(level=0)

    n_inactivos_con_ingreso = df_temp[(df_temp['ESTADO']==3) & (df_temp[p47t]==0) & (df_temp['CAT_INAC']!=3)].groupby(['CH04','Rango edad'])[pondera].size().unstack(level=0)
    n_inactivos_sin_ingreso = df_temp[(df_temp['ESTADO']==3) & (df_temp[p47t]>0) & (df_temp['CAT_INAC']!=3)].groupby(['CH04','Rango edad'])[pondera].size().unstack(level=0)
    n_inactivos_no_declara_ingreso = df_temp[(df_temp['ESTADO']==3) & (df_temp[p47t]==-9) & (df_temp['CAT_INAC']!=3)].groupby(['CH04','Rango edad'])[pondera].size().unstack(level=0)
    
    size_colums_names = pd.MultiIndex.from_product([['Ingreso > 0', 'Ingreso = 0', 'Ingreso = -9'], ['N_v', 'N_m']])
    size_colums = pd.concat([n_inactivos_con_ingreso, n_inactivos_sin_ingreso, n_inactivos_no_declara_ingreso], axis=1)
    size_colums.loc['Total Col'] = size_colums.sum(numeric_only=True, axis=0)

    df_inactivos_con_ingreso.columns = pd.MultiIndex.from_tuples([('Ingreso > 0', 'Varon'), ('Ingreso > 0', 'Mujer')])
    df_inactivos_sin_ingreso.columns = pd.MultiIndex.from_tuples([('Ingreso = 0', 'Varon'), ('Ingreso = 0', 'Mujer')])
    df_inactivos_no_declara_ingreso.columns = pd.MultiIndex.from_tuples([('Ingreso = -9', 'Varon'), ('Ingreso = -9', 'Mujer')])

    df_inactivos = pd.concat([df_inactivos_con_ingreso, df_inactivos_sin_ingreso, df_inactivos_no_declara_ingreso], axis=1)
    df_inactivos.loc['Total Col'] = df_inactivos.sum(numeric_only=True, axis=0)

    df_inactivos_porcentaje = df_inactivos.iloc[:-1,:] / df_inactivos.iloc[-1,:] * 100
    df_inactivos_porcentaje.loc['Total Col'] = df_inactivos_porcentaje.sum(numeric_only=True, axis=0)
    tabla_pob_inactiva_porcentaje = df_inactivos_porcentaje.style.format('{:,.2f}')

    df_inactivos[size_colums_names] = size_colums.values

    df_inactivos = df_inactivos.sort_index(axis=1, ascending=False)
    tabla_pob_inactiva = df_inactivos.style.format('{:,.0f}')

    ratio_pob_inactiva = pd.DataFrame()
    ratio_pob_inactiva.loc[:,'Ingreso > 0'] = df_inactivos_porcentaje.iloc[:,1]/df_inactivos_porcentaje.iloc[:,0] * 100
    ratio_pob_inactiva.loc[:,'Ingreso = 0'] = df_inactivos_porcentaje.iloc[:,3]/df_inactivos_porcentaje.iloc[:,2] * 100
    ratio_pob_inactiva.loc[:,'Ingreso = -9'] = df_inactivos_porcentaje.iloc[:,5]/df_inactivos_porcentaje.iloc[:,4] * 100
    ratio_pob_inactiva.drop(ratio_pob_inactiva.tail(1).index, inplace=True)

    return tabla_pob_inactiva, tabla_pob_inactiva_porcentaje, ratio_pob_inactiva


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
    grouped_df['ANO'] = custom_merged_df['ANO4']
    grouped_df['INCOME'] = custom_merged_df['ITF']
    grouped_df['INCOME_PONDERATOR'] = custom_merged_df['PONDIH_new']
    grouped_df['PROVINCIA'] = custom_merged_df['PROVINCIA']
    grouped_df['AGLOMERADO'] = custom_merged_df['AGLOMERADO']

def reshape_and_filter_data(grouped_df, df_CBT, dict_cod_provincia, dict_cod_aglomerado, map_aglomerado_region):
    melted_df_CBT = df_CBT.melt(id_vars='Trimestre', var_name='Region', value_name='CBT')
    grouped_df['Provincia'] = grouped_df['PROVINCIA'].map(dict_cod_provincia)
    grouped_df['Aglomerado'] = grouped_df['AGLOMERADO'].map(dict_cod_aglomerado)
    grouped_df['Region'] = grouped_df['Aglomerado'].map(map_aglomerado_region)
    
    grouped_df[grouped_df['ANO'] == 2021]
    filtered_melted_df_CBT_2021 = melted_df_CBT[melted_df_CBT['Trimestre'] == '3T2021']
    filtered_melted_df_CBT_2022 = melted_df_CBT[melted_df_CBT['Trimestre'] == '3T2022']

    map_region_CBT_2021 = filtered_melted_df_CBT_2021.set_index('Region')['CBT'].to_dict()
    map_region_CBT_2022 = filtered_melted_df_CBT_2022.set_index('Region')['CBT'].to_dict()
    
    grouped_df['CBT'] = pd.NaT
    grouped_df.loc[grouped_df['ANO'] == 2021, 'CBT'] = grouped_df.loc[grouped_df['ANO'] == 2021, 'Region'].map(map_region_CBT_2021)
    grouped_df.loc[grouped_df['ANO'] == 2022, 'CBT'] = grouped_df.loc[grouped_df['ANO'] == 2022, 'Region'].map(map_region_CBT_2022)

    grouped_df['THRESHOLD'] = grouped_df['CBT'] * grouped_df['EQUIVALENT_ADULTS']
    
def calculate_poverty_table(df_people, df_houses, df_CBT, df_adultos_equiv, dict_cod_provincia, map_provincia_region):
    grouped_df = df_people.groupby(['CODUSU', 'NRO_HOGAR']).agg({
        'CH03': list,
        'COMPONENTE': list,
        'CH04': list,
        'CH06': list,
        'P47T': list,
    })

    # grouped_df = grouped_df[grouped_df['COMPONENTE'].apply(lambda x: 2 not in x)]
    grouped_df['NUM_PEOPLE_IN_HOUSE'] = grouped_df['CH06'].apply(len)
    grouped_df['MEAN_AGE'] = grouped_df['CH06'].apply(lambda ages: sum(ages) / len(ages) if ages else None)
    grouped_df = grouped_df[grouped_df['CH06'].apply(lambda x: any(age < 25 for age in x))]
    grouped_df['GENDER_PERSON_IN_CHARGE'] = grouped_df.apply(lambda row: 'MALE' if 1 in row['CH03'] and row['CH04'][row['CH03'].index(1)] == 1 else 'FEMALE', axis=1)
    grouped_df['PERSON_IN_CHARGE_AGE'] = grouped_df.apply(lambda row: row['CH06'][row['CH03'].index(1)] if 1 in row['CH03'] else None, axis=1)
    grouped_df['GENDER_MAX_INCOME'] = grouped_df.apply(lambda row: 'MALE' if row['P47T'] and row['CH04'][row['P47T'].index(max(row['P47T']))] == 1 else 'FEMALE', axis=1)

    grouped_df = grouped_df.reset_index()

    calculate_equivalent_adults(grouped_df, df_adultos_equiv)
    merge_and_add_columns(grouped_df, df_houses)
    reshape_and_filter_data(grouped_df, df_CBT, dict_cod_provincia, dict_cod_aglomerado, map_aglomerado_region)

    return grouped_df


def ratio_no_pobreza(df_people_pool, df_houses_pool, df_CBT, df_adultos_equiv, dict_cod_provincia, map_provincia_region):

    var = 'GENDER_PERSON_IN_CHARGE'
    df_temp = df_houses_pool.loc[(df_houses_pool['IX_TOT']>1) & (df_houses_pool['REALIZADA']==1) & (~df_houses_pool['NRO_HOGAR'].isin([51, 71]))]

    grouped_df = calculate_poverty_table(df_people_pool, df_temp, df_CBT, df_adultos_equiv, dict_cod_provincia, map_provincia_region)
    grouped_df = grouped_df[~grouped_df['INCOME'].isna()]
    
    hogares_seleccionados = grouped_df[grouped_df['INCOME'] > grouped_df['THRESHOLD']]
    hogares_seleccionados_pob = hogares_seleccionados.groupby([var,'Provincia'])['INCOME_PONDERATOR'].sum().unstack(level=0)

    hogares_totales = grouped_df.groupby(['Provincia', var])['INCOME_PONDERATOR'].sum().unstack(level=1)
    fraccion =  hogares_seleccionados_pob / hogares_totales

    row_counts = grouped_df.groupby(['Provincia', var]).size().unstack()

    hogares_totales['ROW_COUNTS_FEMALE'] = row_counts['FEMALE']
    hogares_totales['ROW_COUNTS_MALE'] = row_counts['MALE']

    gender_ratios_pool = fraccion['FEMALE'] / fraccion['MALE']

    tasa = fraccion
    tasa = tasa.rename(columns= {'MALE':'Varon', 'FEMALE':'Mujer'})

    size = row_counts
    size = size.rename(columns= {'MALE':'N_v', 'FEMALE':'N_m'})

    n_pob = hogares_seleccionados_pob.copy()
    n_pob = n_pob.rename(columns= {'MALE':'N_pob_v', 'FEMALE':'N_pob_m'})

    size[['N_pob_v', 'N_pob_m']] = n_pob[['N_pob_v', 'N_pob_m']]
    size['N_pob_tot'] = n_pob['N_pob_v'] + n_pob['N_pob_m']
        
    ratio = gender_ratios_pool.to_frame()
    ratio.rename(columns={0: 'No Pobreza'}, inplace=True)
    error = pd.DataFrame(index=size.index, columns=['LI', 'LS', 'ME', 'ER'])

    hogares_no_pobres_jefatura = pd.concat([tasa*100, size, ratio*100, error*100], axis=1)

    return hogares_no_pobres_jefatura


def calcular_ratios_TNR(df, df_enut, dict_regiones_ENUT, dict_cod_aglomerado, map_aglomerado_provincia, map_provincia_region):
    
    # 1. Calcular el ratio a nivel regional
    def calcular_ratio_regional(df_enut, dict_regiones_ENUT):
        pondera = 'WPER'
        df_temp = df_enut[df_enut['EDAD_SEL'] >= 14]
        df_selected = df_temp[~df_temp['TSS_GRANGRUPO_TNR'].isna()][['TSS_GRANGRUPO_TNR', 'SEXO_SEL', 'REGION', pondera]]
        df_selected['Multiplication'] = df_selected['TSS_GRANGRUPO_TNR'] * df_selected[pondera]

        numerador = df_selected.groupby(['SEXO_SEL', 'REGION'])['Multiplication'].sum().unstack(level=0)
        denominador = df_selected.groupby(['SEXO_SEL', 'REGION'])[pondera].sum().unstack(level=0)
        tasa = numerador.div(denominador, fill_value=np.nan)
        tasa = tasa.rename(columns={2: 'Varon', 1: 'Mujer'})

        size = df_temp[~df_temp['TSS_GRANGRUPO_TNR'].isna()].groupby(['REGION', 'SEXO_SEL']).size().unstack(level=1)
        size = size.rename(columns={2: 'N_v', 1: 'N_m'})

        n_pob = df_temp[~df_temp['TSS_GRANGRUPO_TNR'].isna()].groupby(['REGION', 'SEXO_SEL'])[pondera].sum().unstack(level=1)
        n_pob = n_pob.rename(columns={2: 'N_pob_v', 1: 'N_pob_m'})

        size[['N_pob_v', 'N_pob_m']] = n_pob[['N_pob_v', 'N_pob_m']]
        size['N_pob_tot'] = n_pob['N_pob_v'] + n_pob['N_pob_m']

        ratio = tasa['Varon'] / tasa['Mujer']

        error = pd.DataFrame(index=size.index, columns=['LI', 'LS', 'ME', 'ER'])

        ratio_minutos_promedio_no_pago = pd.concat([tasa, size, ratio * 100, error], axis=1)
        ratio_minutos_promedio_no_pago.rename(columns={0: 'Tiempo TNR'}, inplace=True)

        ratio_minutos_promedio_no_pago.index = ratio_minutos_promedio_no_pago.index.map(dict_regiones_ENUT)

        return ratio_minutos_promedio_no_pago

    # 2. Calcular coeficientes para imputar a PBA
    def calcular_coeficientes_imputacion(df, dict_cod_aglomerado, map_aglomerado_provincia):
        df_temp = df[df['CH06'] > 14]
        df_pob = df_temp.groupby(['AGLOMERADO']).sum(['PONDERA_new'])
        df_pob.index = df_pob.index.map(dict_cod_aglomerado)
        df_pob['PROVINCIA'] = df_pob.index.map(map_aglomerado_provincia)
        pob_pba = df_pob[df_pob['PROVINCIA'] == 'Buenos Aires']['PONDERA_new'].sum()
        gba = df_pob[df_pob['PROVINCIA'] == 'Buenos Aires'].loc['Partidos del GBA', 'PONDERA_new']
        resto_pba = df_pob[df_pob['PROVINCIA'] == 'Buenos Aires'].loc[
            ['Gran La Plata', 'Bahía Blanca - Cerri', 'Mar del Plata', 'Resto Buenos Aires'], 'PONDERA_new'].sum()
        coef_gba = gba / pob_pba
        coef_resto_pba = resto_pba / pob_pba

        return coef_gba, coef_resto_pba

    # 3. Imputar de regiones a provincias
    def imputar_region_a_provincias(ratio_minutos_promedio_no_pago, map_provincia_region):
        map_provincia_region['Ciudad Autónoma de Buenos Aires'] = 'Gran Buenos Aires'
        provincias = list(map_provincia_region.keys())
        ratio_minutos_promedio_no_pago_provincial = pd.DataFrame(index=provincias, columns=['Mujer', 'Varon'])
        for provincia in provincias:
            region = map_provincia_region[provincia]
            ratio_minutos_promedio_no_pago_provincial.loc[provincia, :] = ratio_minutos_promedio_no_pago.loc[region, ['Mujer', 'Varon']]

        ratio_minutos_promedio_no_pago_provincial.index.name = 'Provincia'

        ## Corregir el caso de PBA
        ratio_minutos_promedio_no_pago_provincial.loc['Buenos Aires', ['Mujer', 'Varon']] = ratio_minutos_promedio_no_pago.loc[
                                                                                           'Gran Buenos Aires',
                                                                                           ['Mujer', 'Varon']] * coef_gba + \
                                                                                           ratio_minutos_promedio_no_pago.loc[
                                                                                           'Pampeana',
                                                                                           ['Mujer', 'Varon']] * coef_resto_pba

        # Calcular ratio
        ratio_minutos_promedio_no_pago_provincial['Tiempo TNR'] = ratio_minutos_promedio_no_pago_provincial[
                                                                     'Varon'] / ratio_minutos_promedio_no_pago_provincial[
                                                                     'Mujer'] * 100
        ratio_minutos_promedio_no_pago_provincial = ratio_minutos_promedio_no_pago_provincial.rename(
            index={'Ciudad Autónoma de Buenos Aires': 'CABA', 'Tierra del Fuego, Antártida e Islas del Atlántico Sur': 'TdF'})

        ratio_minutos_promedio_no_pago = ratio_minutos_promedio_no_pago.sort_index()
        ratio_minutos_promedio_no_pago_provincial = ratio_minutos_promedio_no_pago_provincial.sort_index()

        return ratio_minutos_promedio_no_pago, ratio_minutos_promedio_no_pago_provincial


    # Llamada a las funciones internas
    ratio_minutos_promedio_no_pago = calcular_ratio_regional(df_enut, dict_regiones_ENUT)
    coef_gba, coef_resto_pba = calcular_coeficientes_imputacion(df, dict_cod_aglomerado, map_aglomerado_provincia)

    ratio_minutos_promedio_no_pago, ratio_minutos_promedio_no_pago_provincial = imputar_region_a_provincias(
        ratio_minutos_promedio_no_pago, map_provincia_region)

    return ratio_minutos_promedio_no_pago, ratio_minutos_promedio_no_pago_provincial


def calcular_tabla_jefatura_con_conyuges(df_people_pool, df_houses_pool, df_CBT, df_adultos_equiv, dict_cod_provincia, map_provincia_region):

    """
    Calcula la tabla de .... 

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.

    Returns:
        tabla (DataFrame): DataFrame con la tabla_jefatura_con_conyuges

    """
    var = 'GENDER_PERSON_IN_CHARGE'
    df_temp = df_houses_pool.loc[(df_houses_pool['IX_TOT']>1) & (df_houses_pool['REALIZADA']==1) & (~df_houses_pool['NRO_HOGAR'].isin([51, 71]))]

    grouped_df = calculate_poverty_table(df_people_pool, df_houses_pool, df_CBT, df_adultos_equiv, dict_cod_provincia, map_provincia_region)
    grouped_df = grouped_df[~grouped_df['INCOME'].isna()]
    
    
    df_estado = df_temp[(df_temp['ESTADO'] == 1) & (df_temp['CAT_OCUP'] == 3)]
    
    resultados = df_estado.groupby(['CH04', 'PP07H', 'PP07I'])[pondera].sum().unstack(level='CH04', fill_value=0)
    resultados.columns = ['Varones', 'Mujeres']
    resultados = resultados.reset_index()
    
    resultados['Total Fila'] = resultados['Varones'] + resultados['Mujeres']
    resultados['Combinación'] = resultados.apply(lambda row: f"PP07H == {row['PP07H']} y PP07I == {row['PP07I']}", axis=1)
    
    resultados.loc['Total Col'] = resultados.sum(numeric_only=True, axis=0)
    
    resultados.loc[:, '% Varones'] = resultados['Varones'] / resultados.loc['Total Col', 'Varones'] * 100
    resultados.loc[:, '% Mujeres'] = resultados['Mujeres'] / resultados.loc['Total Col', 'Mujeres'] * 100
    
    resultados = resultados[['Combinación', 'Varones', '% Varones', 'Mujeres', '% Mujeres', 'Total Fila']]
    
    tabla_pob_ocupada_asalariada = resultados.style.format({
        'Varones': '{:,.0f}',
        '% Varones': '{:,.2f}',
        'Mujeres': '{:,.0f}',
        '% Mujeres': '{:,.2f}',
        'Total Fila': '{:,.0f}'
    })
    
    return resultados, tabla_pob_ocupada_asalariada


def indicadores_uso_del_tiempo_y_oportunidades(df_people, df_houses, df_enut, dict_regiones_ENUT, dict_cod_provincia, dict_cod_aglomerado, map_aglomerado_provincia, map_provincia_region, df_CBT, df_adultos_equiv, tipo='Urbano', base='Individual', confidence_level=0.95, pool=True):

    tasa, size, ratio, error = poblacion_inactiva_con_ingresos(df_people, tipo='Urbano', base='Individual', confidence_level=0.95, pool=True)
    inactivos_con_ingreso = pd.concat([tasa, size, ratio, error], axis=1)

    hogares_no_pobres_jefatura = ratio_no_pobreza(df_people, df_houses, df_CBT, df_adultos_equiv, dict_cod_provincia, map_provincia_region)

    jornada_no_paga, jornada_no_paga_provincial = calcular_ratios_TNR(df_people, df_enut, dict_regiones_ENUT, dict_cod_aglomerado, map_aglomerado_provincia, map_provincia_region)

    inactivos_con_ingreso.index = inactivos_con_ingreso.index.map(dict_cod_provincia)

    return inactivos_con_ingreso, hogares_no_pobres_jefatura, jornada_no_paga_provincial

def calcular_ratios_TNR(df, df_enut, dict_regiones_ENUT, dict_cod_aglomerado, map_aglomerado_provincia, map_provincia_region):
    
    # 1. Calcular el ratio a nivel regional
    def calcular_ratio_regional(df_enut, dict_regiones_ENUT):
        pondera = 'WPER'
        df_temp = df_enut[df_enut['EDAD_SEL'] >= 14]
        df_selected = df_temp[~df_temp['TSS_GRANGRUPO_TNR'].isna()][['TSS_GRANGRUPO_TNR', 'SEXO_SEL', 'REGION', pondera]]
        df_selected['Multiplication'] = df_selected['TSS_GRANGRUPO_TNR'] * df_selected[pondera]

        numerador = df_selected.groupby(['SEXO_SEL', 'REGION'])['Multiplication'].sum().unstack(level=0)
        denominador = df_selected.groupby(['SEXO_SEL', 'REGION'])[pondera].sum().unstack(level=0)
        tasa = numerador.div(denominador, fill_value=np.nan)
        tasa = tasa.rename(columns={2: 'Varon', 1: 'Mujer'})

        size = df_temp[~df_temp['TSS_GRANGRUPO_TNR'].isna()].groupby(['REGION', 'SEXO_SEL']).size().unstack(level=1)
        size = size.rename(columns={2: 'N_v', 1: 'N_m'})

        n_pob = df_temp[~df_temp['TSS_GRANGRUPO_TNR'].isna()].groupby(['REGION', 'SEXO_SEL'])[pondera].sum().unstack(level=1)
        n_pob = n_pob.rename(columns={2: 'N_pob_v', 1: 'N_pob_m'})

        size[['N_pob_v', 'N_pob_m']] = n_pob[['N_pob_v', 'N_pob_m']]
        size['N_pob_tot'] = n_pob['N_pob_v'] + n_pob['N_pob_m']

        ratio = tasa['Varon'] / tasa['Mujer']

        error = pd.DataFrame(index=size.index, columns=['LI', 'LS', 'ME', 'ER'])

        ratio_minutos_promedio_no_pago = pd.concat([tasa, size, ratio * 100, error], axis=1)
        ratio_minutos_promedio_no_pago.rename(columns={0: 'Tiempo TNR'}, inplace=True)

        ratio_minutos_promedio_no_pago.index = ratio_minutos_promedio_no_pago.index.map(dict_regiones_ENUT)

        return ratio_minutos_promedio_no_pago

    # 2. Calcular coeficientes para imputar a PBA
    def calcular_coeficientes_imputacion(df, dict_cod_aglomerado, map_aglomerado_provincia):
        df_temp = df[df['CH06'] > 14]
        df_pob = df_temp.groupby(['AGLOMERADO']).sum(['PONDERA_new'])
        df_pob.index = df_pob.index.map(dict_cod_aglomerado)
        df_pob['PROVINCIA'] = df_pob.index.map(map_aglomerado_provincia)
        pob_pba = df_pob[df_pob['PROVINCIA'] == 'Buenos Aires']['PONDERA_new'].sum()
        gba = df_pob[df_pob['PROVINCIA'] == 'Buenos Aires'].loc['Partidos del GBA', 'PONDERA_new']
        resto_pba = df_pob[df_pob['PROVINCIA'] == 'Buenos Aires'].loc[
            ['Gran La Plata', 'Bahía Blanca - Cerri', 'Mar del Plata', 'Resto Buenos Aires'], 'PONDERA_new'].sum()
        coef_gba = gba / pob_pba
        coef_resto_pba = resto_pba / pob_pba

        return coef_gba, coef_resto_pba

    # 3. Imputar de regiones a provincias
    def imputar_region_a_provincias(ratio_minutos_promedio_no_pago, map_provincia_region):
        map_provincia_region['Ciudad Autónoma de Buenos Aires'] = 'Gran Buenos Aires'
        provincias = list(map_provincia_region.keys())
        ratio_minutos_promedio_no_pago_provincial = pd.DataFrame(index=provincias, columns=['Mujer', 'Varon'])
        for provincia in provincias:
            region = map_provincia_region[provincia]
            ratio_minutos_promedio_no_pago_provincial.loc[provincia, :] = ratio_minutos_promedio_no_pago.loc[region, ['Mujer', 'Varon']]

        ratio_minutos_promedio_no_pago_provincial.index.name = 'Provincia'

        ## Corregir el caso de PBA
        ratio_minutos_promedio_no_pago_provincial.loc['Buenos Aires', ['Mujer', 'Varon']] = ratio_minutos_promedio_no_pago.loc[
                                                                                           'Gran Buenos Aires',
                                                                                           ['Mujer', 'Varon']] * coef_gba + \
                                                                                           ratio_minutos_promedio_no_pago.loc[
                                                                                           'Pampeana',
                                                                                           ['Mujer', 'Varon']] * coef_resto_pba

        # Calcular ratio
        ratio_minutos_promedio_no_pago_provincial['Tiempo TNR'] = ratio_minutos_promedio_no_pago_provincial[
                                                                     'Varon'] / ratio_minutos_promedio_no_pago_provincial[
                                                                     'Mujer'] * 100
        ratio_minutos_promedio_no_pago_provincial = ratio_minutos_promedio_no_pago_provincial.rename(
            index={'Ciudad Autónoma de Buenos Aires': 'CABA', 'Tierra del Fuego, Antártida e Islas del Atlántico Sur': 'TdF'})

        ratio_minutos_promedio_no_pago = ratio_minutos_promedio_no_pago.sort_index()
        ratio_minutos_promedio_no_pago_provincial = ratio_minutos_promedio_no_pago_provincial.sort_index()

        return ratio_minutos_promedio_no_pago, ratio_minutos_promedio_no_pago_provincial


    # Llamada a las funciones internas
    ratio_minutos_promedio_no_pago = calcular_ratio_regional(df_enut, dict_regiones_ENUT)
    coef_gba, coef_resto_pba = calcular_coeficientes_imputacion(df, dict_cod_aglomerado, map_aglomerado_provincia)

    ratio_minutos_promedio_no_pago, ratio_minutos_promedio_no_pago_provincial = imputar_region_a_provincias(
        ratio_minutos_promedio_no_pago, map_provincia_region)

    return ratio_minutos_promedio_no_pago, ratio_minutos_promedio_no_pago_provincial

