#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[4]:


def ratio_tasas_empleo(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio de tasas de empleo entre hombres y mujeres, junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las tasas de empleo (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los números de observaciones utilizados para calcular tasas de empleo (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de tasas de empleo (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Límite inferior, LS: Límite superior, ME: Margen de error, ER: Error relativo (CV)
    """


    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')

    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'

    numerador = df_temp[df_temp['ESTADO'] == 1].groupby(['CH04', var])[pondera].sum().unstack(level=0)
    denominador = df_temp.groupby(['CH04', var])[pondera].sum().unstack(level=0)

    tasa = numerador.div(denominador, fill_value=np.nan)
    tasa.columns = ['Varon', 'Mujer']

    size = df_temp[df_temp['ESTADO'] == 1].groupby(['CH04', var]).size().unstack(level=0)
    size.columns = ['N_v', 'N_m']

    p1 = tasa['Mujer']
    p2 = tasa['Varon']
    n1 = size['N_m']
    n2 = size['N_v']
    ratio = (p1 / p2).to_frame()
    ratio.rename(columns={0: 'Empleo'}, inplace=True)

    standard_error = np.sqrt(((1 / n1) * (p1 * (1 - p1))) + ((1 / n2) * (p2 * (1 - p2))))

    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z * standard_error
    lower_bound = p1 / p2 - margin_of_error
    upper_bound = p1 / p2 + margin_of_error

    relative_standard_error = margin_of_error / (p1 / p2)
    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa * 100, size, ratio * 100, error * 100


# In[6]:


def ratio_tasas_desempleo(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio de tasas de desempleo entre hombres y mujeres, junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las tasas de desempleo (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular tasas de desempleo (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de tasas de desempleo (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)


    """

    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
    
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'
        
    numerador = df_temp[df_temp['ESTADO'] == 2].groupby(['CH04', var])[pondera].sum().unstack(level=0)
    df_estado = df_temp[(df_temp['ESTADO'] == 1) | (df_temp['ESTADO'] == 2)]
    denominador = df_estado.groupby(['CH04', var])[pondera].sum().unstack(level=0)

    tasa = numerador.div(denominador, fill_value=np.nan)
    tasa.columns = ['Varon', 'Mujer']

    size = df_temp[df_temp['ESTADO'] == 2].groupby(['CH04', var]).size().unstack(level=0)
    size.columns = ['N_v', 'N_m']

    p1 = tasa['Varon']
    p2 = tasa['Mujer']
    n1 = size['N_v']
    n2 = size['N_m']
    ratio = (p1 / p2).to_frame()
    ratio.rename(columns={0: 'Desempleo'}, inplace=True)

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


# In[8]:


def formalidad(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio de tasas de formalidad (personas ocupadas asalariadas con descuento jubilatorio) entre hombres y mujeres junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las tasas de registro (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular tasas de registro (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de tasas de registro (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)

    """

    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
    
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'
        
    df_estado = df_temp[(df_temp['ESTADO'] == 1) & (df_temp['CAT_OCUP'] == 3)]          # personas ocupadas asalariadas
    numerador = df_estado[df_estado['PP07H']==1].groupby(['CH04', var])[pondera].sum().unstack(level=0)

    denominador = df_estado.groupby(['CH04', var])[pondera].sum().unstack(level=0)

    tasa = numerador.div(denominador, fill_value=np.nan)
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_estado[df_estado['PP07H']==1].groupby(['CH04', var]).size().unstack(level=0)
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
    ratio.rename(columns={0: 'Formalidad'}, inplace=True)

    standard_error = np.sqrt(((1 / n1) * (p1 * (1 - p1))) + ((1 / n2) * (p2 * (1 - p2))))

    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z * standard_error
    lower_bound = p1/p2 - margin_of_error
    upper_bound = p1/p2 + margin_of_error

    relative_standard_error = margin_of_error/(p1/p2)
    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa*100, size, ratio*100, error*100


# In[1]:


def ingreso_salarial_promedio(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio entre la tasa de horas promedio remuneradas de personas ocupadas entre hombres y mujeres junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las horas trabajadas remuneradas (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular las horas trabajadas remuneradas (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de las horas trabajadas remuneradas (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)


    """
    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
        
    if pool:
        pondera = 'PONDIIO_new'
    else:
        pondera = 'PONDIIO'

    if pool:
        p21 = 'P21_new'
    else:
        p21 = 'P21'

    df_estado = df_temp[(df_temp['ESTADO'] == 1)]          # personas ocupadas
    df_ocupados = df_temp[(df_temp[p21]!=-9) & (df_temp['ESTADO']==1) & (df_temp['CAT_OCUP'].isin([2,3]))][[var, pondera, p21, 'CH04']]
    df_ocupados = df_ocupados[df_ocupados[p21]>0]
    
    df_ocupados['Multiplication'] = df_ocupados[p21] * df_ocupados[pondera]
    numerador = df_ocupados.groupby(['CH04', var])['Multiplication'].sum().unstack(level=0)
    denominador = df_ocupados.groupby(['CH04', var])[pondera].sum().unstack(level=0)
    tasa = numerador.div(denominador, fill_value=np.nan)    
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_estado[(df_estado[p21]>0) & (df_estado[p21]!=np.nan)].groupby(['CH04', var]).size().unstack(level=0)
    size = size.rename(columns= {1:'N_v', 2:'N_m'})

    n_pob = df_ocupados.groupby(['CH04', var])[pondera].sum().unstack(level=0)
    n_pob = n_pob.rename(columns= {1:'N_pob_v', 2:'N_pob_m'})

    size[['N_pob_v', 'N_pob_m']] = n_pob[['N_pob_v', 'N_pob_m']]
    size['N_pob_tot'] = n_pob['N_pob_v'] + n_pob['N_pob_m']    

    df_ocupados['Deviation'] = df_ocupados[p21] - df_ocupados.groupby(['CH04', var])[p21].transform('mean')
    df_ocupados['Weighted_Deviation'] = df_ocupados[pondera] * df_ocupados['Deviation']**2
    weighted_variance = df_ocupados.groupby(['CH04', var])['Weighted_Deviation'].sum() / df_ocupados.groupby(['CH04', var])[pondera].sum()

    weighted_std = np.sqrt(weighted_variance.astype('float')).unstack(level=0)
    weighted_std = weighted_std.rename(columns= {1:'Varon', 2:'Mujer'})

    standard_error = np.sqrt((weighted_std['Mujer']**2 / size['N_m']) + (weighted_std['Varon']**2 / size['N_v']))

    degrees_of_freedom = size['N_m'] + size['N_v'] - 2
    t = stats.t.ppf((1 + confidence_level) / 2, df=degrees_of_freedom)
    margin_of_error = t * standard_error
    
    ratio = (tasa['Mujer'] / tasa['Varon']).to_frame()
    ratio.rename(columns={0: 'Ingreso laboral promedio'}, inplace=True)

    lower_bound = tasa['Mujer'] / tasa['Varon'] - margin_of_error
    upper_bound = tasa['Mujer'] / tasa['Varon'] + margin_of_error
    relative_standard_error = margin_of_error / (tasa['Mujer'] / tasa['Varon'])

    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']
    error[['LI', 'LS', 'ME', 'ER']] = pd.NaT

    return tasa, size, ratio*100, error


# In[ ]:


def jornada_laboral(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio entre la tasa de horas promedio remuneradas de personas ocupadas entre hombres y mujeres junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las horas trabajadas remuneradas (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular las horas trabajadas remuneradas (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de las horas trabajadas remuneradas (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)


    """
    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
        
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'

    df_estado = df_temp[(df_temp['ESTADO'] == 1)]          # personas ocupadas
    
    if df_estado['PP3F_TOT'].dtype == 'object':
        df_estado['PP3F_TOT'] = df_estado['PP3F_TOT'].str.replace(',', '.').astype(float)

    if df_estado['PP3E_TOT'].dtype == 'object':
        df_estado['PP3E_TOT'] = df_estado['PP3E_TOT'].str.replace(',', '.').astype(float)
        
    df_estado['Horas'] = pd.NaT

    mask_999 = (df_estado['PP3E_TOT'] == 999) & (df_estado['PP3F_TOT'] == 999)
    mask_pp3f_999 = (df_estado['PP3F_TOT'] == 999) & (df_estado['PP3E_TOT'] != 999)
    mask_pp3e_999 = (df_estado['PP3E_TOT'] == 999) & (df_estado['PP3F_TOT'] != 999)

    df_estado['Horas'] = np.where(mask_999, pd.NaT,
                                np.where(mask_pp3f_999, df_estado['PP3E_TOT'],
                                        np.where(mask_pp3e_999, df_estado['PP3F_TOT'],
                                                    np.maximum(df_estado['PP3E_TOT'], df_estado['PP3F_TOT'])
                                                )
                                        )
                                )

    df_selected = df_estado[(df_estado['Horas'] > 0) & (~df_estado['Horas'].isna())][['CH04', var, 'Horas', pondera]]
    df_selected['Multiplication'] = df_selected['Horas'] * df_selected[pondera]
    numerador = df_selected.groupby(['CH04', var])['Multiplication'].sum().unstack(level=0)
    denominador = df_selected.groupby(['CH04', var])[pondera].sum().unstack(level=0)
    tasa = numerador.div(denominador, fill_value=np.nan)    
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_estado[(df_estado['Horas']>0) & (df_estado['Horas']!=np.nan)].groupby(['CH04', var]).size().unstack(level=0)
    size = size.rename(columns= {1:'N_v', 2:'N_m'})
    
    n_pob = df_estado.groupby(['CH04', var])[pondera].sum().unstack(level=0)
    n_pob = n_pob.rename(columns= {1:'N_pob_v', 2:'N_pob_m'})

    size[['N_pob_v', 'N_pob_m']] = n_pob[['N_pob_v', 'N_pob_m']]
    size['N_pob_tot'] = n_pob['N_pob_v'] + n_pob['N_pob_m'] 
    
    df_selected['Deviation'] = df_selected['Horas'] - df_selected.groupby(['CH04', var])['Horas'].transform('mean')
    df_selected['Weighted_Deviation'] = df_selected[pondera] * df_selected['Deviation']**2
    weighted_variance = df_selected.groupby(['CH04', var])['Weighted_Deviation'].sum() / df_selected.groupby(['CH04', var])[pondera].sum()

    weighted_std = np.sqrt(weighted_variance.astype('float')).unstack(level=0)
    weighted_std.columns = ['Varon', 'Mujer']

    standard_error = np.sqrt((weighted_std['Mujer']**2 / size['N_m']) + (weighted_std['Varon']**2 / size['N_v']))

    degrees_of_freedom = size['N_m'] + size['N_v'] - 2
    t = stats.t.ppf((1 + confidence_level) / 2, df=degrees_of_freedom)
    margin_of_error = t * standard_error
    
    ratio = (tasa['Mujer'] / tasa['Varon']).to_frame()
    ratio.rename(columns={0: 'Horas promedio remuneradas'}, inplace=True)

    lower_bound = tasa['Mujer'] / tasa['Varon'] - margin_of_error
    upper_bound = tasa['Mujer'] / tasa['Varon'] + margin_of_error
    relative_standard_error = margin_of_error / (tasa['Mujer'] / tasa['Varon'])

    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa, size, ratio*100, error


# In[11]:


def horas_promedio_remuneradas_secundarias(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio entre la tasa de horas promedio remuneradas de personas ocupadas entre hombres y mujeres junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las horas trabajadas remuneradas (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular las horas trabajadas remuneradas (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de las horas trabajadas remuneradas (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)


    """
    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
        
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'

    df_estado = df_temp[(df_temp['ESTADO'] == 1)]          # personas ocupadas
    
    if df_estado['PP3F_TOT'].dtype == 'object':
        df_estado['PP3F_TOT'] = df_estado['PP3F_TOT'].str.replace(',', '.').astype(float)

    if df_estado['PP3E_TOT'].dtype == 'object':
        df_estado['PP3E_TOT'] = df_estado['PP3E_TOT'].str.replace(',', '.').astype(float)
        
    df_estado['Horas'] = pd.NaT

    mask_999 = (df_estado['PP3E_TOT'] == 999) & (df_estado['PP3F_TOT'] == 999)
    mask_pp3f_999 = (df_estado['PP3F_TOT'] == 999) & (df_estado['PP3E_TOT'] != 999)
    mask_pp3e_999 = (df_estado['PP3E_TOT'] == 999) & (df_estado['PP3F_TOT'] != 999)

    df_estado['Horas'] = np.where(mask_999, pd.NaT,
                                np.where(mask_pp3f_999, pd.NaT,
                                        np.where(mask_pp3e_999, pd.NaT,
                                                    np.minimum(df_estado['PP3E_TOT'], df_estado['PP3F_TOT'])
                                                )
                                        )
                                )

    df_selected = df_estado[(df_estado['Horas'] > 0) & (~df_estado['Horas'].isna())][['CH04', var, 'Horas', pondera]]
    df_selected['Multiplication'] = df_selected['Horas'] * df_selected[pondera]
    numerador = df_selected.groupby(['CH04', var])['Multiplication'].sum().unstack(level=0)
    denominador = df_selected.groupby(['CH04', var])[pondera].sum().unstack(level=0)
    tasa = numerador.div(denominador, fill_value=np.nan)    
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_estado[(df_estado['Horas']>0) & (df_estado['Horas']!=np.nan)].groupby(['CH04', var]).size().unstack(level=0)
    size = size.rename(columns= {1:'N_v', 2:'N_m'})
    
    n_pob = df_estado.groupby(['CH04', var])[pondera].sum().unstack(level=0)
    n_pob = n_pob.rename(columns= {1:'N_pob_v', 2:'N_pob_m'})

    size[['N_pob_v', 'N_pob_m']] = n_pob[['N_pob_v', 'N_pob_m']]
    size['N_pob_tot'] = n_pob['N_pob_v'] + n_pob['N_pob_m'] 
    
    df_selected['Deviation'] = df_selected['Horas'] - df_selected.groupby(['CH04', var])['Horas'].transform('mean')
    df_selected['Weighted_Deviation'] = df_selected[pondera] * df_selected['Deviation']**2
    weighted_variance = df_selected.groupby(['CH04', var])['Weighted_Deviation'].sum() / df_selected.groupby(['CH04', var])[pondera].sum()

    weighted_std = np.sqrt(weighted_variance.astype('float')).unstack(level=0)
    weighted_std.columns = ['Varon', 'Mujer']

    standard_error = np.sqrt((weighted_std['Mujer']**2 / size['N_m']) + (weighted_std['Varon']**2 / size['N_v']))

    degrees_of_freedom = size['N_m'] + size['N_v'] - 2
    t = stats.t.ppf((1 + confidence_level) / 2, df=degrees_of_freedom)
    margin_of_error = t * standard_error
    
    ratio = (tasa['Mujer'] / tasa['Varon']).to_frame()
    ratio.rename(columns={0: 'Horas promedio remuneradas'}, inplace=True)

    lower_bound = tasa['Mujer'] / tasa['Varon'] - margin_of_error
    upper_bound = tasa['Mujer'] / tasa['Varon'] + margin_of_error
    relative_standard_error = margin_of_error / (tasa['Mujer'] / tasa['Varon'])

    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa, size, ratio*100, error


# In[ ]:


def horas_mediana_remuneradas(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio entre la tasa de horas promedio remuneradas de personas ocupadas entre hombres y mujeres junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las horas trabajadas remuneradas (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular las horas trabajadas remuneradas (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de las horas trabajadas remuneradas (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)


    """
    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
        
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'

    df_estado = df_temp[(df_temp['ESTADO'] == 1)]          # personas ocupadas
    
    if df_estado['PP3F_TOT'].dtype == 'object':
        df_estado['PP3F_TOT'] = df_estado['PP3F_TOT'].str.replace(',', '.').astype(float)

    if df_estado['PP3E_TOT'].dtype == 'object':
        df_estado['PP3E_TOT'] = df_estado['PP3E_TOT'].str.replace(',', '.').astype(float)
    
    df_estado['Horas'] = pd.NaT

    mask_999 = (df_estado['PP3E_TOT'] == 999) & (df_estado['PP3F_TOT'] == 999)
    mask_pp3f_999 = (df_estado['PP3F_TOT'] == 999) & (df_estado['PP3E_TOT'] != 999)
    mask_pp3e_999 = (df_estado['PP3E_TOT'] == 999) & (df_estado['PP3F_TOT'] != 999)

    df_estado['Horas'] = np.where(mask_999, pd.NaT,
                                np.where(mask_pp3f_999, df_estado['PP3E_TOT'],
                                        np.where(mask_pp3e_999, df_estado['PP3F_TOT'],
                                                    np.maximum(df_estado['PP3E_TOT'], df_estado['PP3F_TOT'])
                                                )
                                        )
                                )

    df_selected = df_estado[(df_estado['Horas'] > 0) & (~df_estado['Horas'].isna())][['CH04', var, 'Horas', pondera]]
    
    def weighted_median(values, weights):
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        median_index = np.searchsorted(cumulative_weights, total_weight / 2.0)
        return sorted_values[median_index]

    def weighted_median_group(x):
        return weighted_median(x['Horas'].values, x[pondera].values)


    df_mediana = df_selected.groupby(['CH04', var]).apply(weighted_median_group)
    tasa = df_mediana.unstack(level=0)
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_estado[(df_estado['Horas']>0) & (df_estado['Horas']!=np.nan)].groupby(['CH04', var]).size().unstack(level=0)
    size = size.rename(columns= {1:'N_v', 2:'N_m'})
    
    n_pob = df_estado.groupby(['CH04', var])[pondera].sum().unstack(level=0)
    n_pob = n_pob.rename(columns= {1:'N_pob_v', 2:'N_pob_m'})

    size[['N_pob_v', 'N_pob_m']] = n_pob[['N_pob_v', 'N_pob_m']]
    size['N_pob_tot'] = n_pob['N_pob_v'] + n_pob['N_pob_m'] 
    
    df_selected['Deviation'] = df_selected['Horas'] - df_selected.groupby(['CH04', var])['Horas'].transform('mean')
    df_selected['Weighted_Deviation'] = df_selected[pondera] * df_selected['Deviation']**2
    weighted_variance = df_selected.groupby(['CH04', var])['Weighted_Deviation'].sum() / df_selected.groupby(['CH04', var])[pondera].sum()

    weighted_std = np.sqrt(weighted_variance.astype('float')).unstack(level=0)
    weighted_std.columns = ['Varon', 'Mujer']

    standard_error = np.sqrt((weighted_std['Mujer']**2 / size['N_m']) + (weighted_std['Varon']**2 / size['N_v']))

    degrees_of_freedom = size['N_m'] + size['N_v'] - 2
    t = stats.t.ppf((1 + confidence_level) / 2, df=degrees_of_freedom)
    margin_of_error = t * standard_error
    
    ratio = (tasa['Mujer'] / tasa['Varon']).to_frame()
    ratio.rename(columns={0: 'Horas mediana remuneradas'}, inplace=True)

    lower_bound = tasa['Mujer'] / tasa['Varon'] - margin_of_error
    upper_bound = tasa['Mujer'] / tasa['Varon'] + margin_of_error
    relative_standard_error = margin_of_error / (tasa['Mujer'] / tasa['Varon'])

    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa, size, ratio*100, error


# In[18]:


def ratio_tasa_inactividad(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio entre la tasa de inactividad entre hombres y mujeres junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las tasas de inactividad (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular las tasas de inactividad (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de las tasas de inactividad (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)


    """
    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
        
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'


    numerador = df_temp[df_temp['ESTADO'] == 3].groupby(['CH04', var])[pondera].sum().unstack(level=0)
    denominador = df_temp.groupby(['CH04', var])[pondera].sum().unstack(level=0)

    tasa = numerador.div(denominador, fill_value=np.nan)
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_temp[df_temp['ESTADO'] == 3].groupby(['CH04', var]).size().unstack(level=0)
    size = size.rename(columns= {1:'N_v', 2:'N_m'})

    n_pob = numerador.copy()
    n_pob = n_pob.rename(columns= {1:'N_pob_v', 2:'N_pob_m'})

    size[['N_pob_v', 'N_pob_m']] = n_pob[['N_pob_v', 'N_pob_m']]
    size['N_pob_tot'] = n_pob['N_pob_v'] + n_pob['N_pob_m']
    
    p1 = tasa['Varon']
    p2 = tasa['Mujer']
    n1 = size['N_v']
    n2 = size['N_m']
    ratio = (p1 / p2).to_frame()
    ratio.rename(columns={0: 'Inactividad'}, inplace=True)

    standard_error = np.sqrt(((1 / n1) * (p1 * (1 - p1))) + ((1 / n2) * (p2 * (1 - p2))))

    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z * standard_error
    lower_bound = p1/p2 - margin_of_error
    upper_bound = p1/p2 + margin_of_error

    relative_standard_error = margin_of_error/(p1/p2)
    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa*100, size, ratio*100, error*100


# In[ ]:


def ratio_tasa_actividad(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio entre la tasa de actividad entre hombres y mujeres junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las tasas de actividad (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular las tasas de actividad (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de las tasas de actividad (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)

    """

    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
        
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'
    
    numerador = df_temp[df_temp['ESTADO'].isin([1,2])].groupby(['CH04', var])[pondera].sum().unstack(level=0)
    denominador = df_temp.groupby(['CH04', var])[pondera].sum().unstack(level=0)

    tasa = numerador.div(denominador, fill_value=np.nan)
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_temp[df_temp['ESTADO'].isin([1,2])].groupby(['CH04', var]).size().unstack(level=0)
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
    ratio.rename(columns={0: 'Actividad'}, inplace=True)

    standard_error = np.sqrt(((1 / n1) * (p1 * (1 - p1))) + ((1 / n2) * (p2 * (1 - p2))))

    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z * standard_error
    lower_bound = p1/p2 - margin_of_error
    upper_bound = p1/p2 + margin_of_error

    relative_standard_error = margin_of_error/(p1/p2)
    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa*100, size, ratio*100, error*100


# In[ ]:


def acceso_aguinaldo(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio de tasas de personas ocupadas asalariadas con acceso a aguinaldo entre hombres y mujeres junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las tasas de acceso a aguinaldo (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular tasas de acceso a aguinaldo (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de tasas de acceso a aguinaldo (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)

    """
    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
        
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'
    
    df_estado = df_temp[(df_temp['ESTADO'] == 1) & (df_temp['CAT_OCUP'] == 3)]                          # personas ocupadas asalariadas
    numerador = df_estado[df_estado['PP07G2']==1].groupby(['CH04', var])[pondera].sum().unstack(level=0)

    denominador = df_estado.groupby(['CH04', var])[pondera].sum().unstack(level=0)

    tasa = numerador.div(denominador, fill_value=np.nan)
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_estado[df_estado['PP07G2']==1].groupby(['CH04', var]).size().unstack(level=0)
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
    ratio.rename(columns={0: 'Acceso a aguinaldo'}, inplace=True)

    standard_error = np.sqrt(((1 / n1) * (p1 * (1 - p1))) + ((1 / n2) * (p2 * (1 - p2))))

    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z * standard_error
    lower_bound = p1/p2 - margin_of_error
    upper_bound = p1/p2 + margin_of_error

    relative_standard_error = margin_of_error/(p1/p2)
    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa*100, size, ratio*100, error*100


# In[21]:


def acceso_obra_social(df, tipo='Aglomerado', base='Individual', confidence_level=0.95, pool=False):
    
    """
    Calcula el ratio de tasas de personas ocupadas asalariadas con acceso a obra social entre hombres y mujeres junto con los errores asociados.

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.
        pool (bool): Indica si se utiliza una base de muestras ("pool") o no. Por defecto: False.

    Returns:
        tasa (DataFrame): DataFrame con las tasas de acceso a obra social (M/V) desglosadas por Aglomerado o Provincia. Expresado en % [0-100]
        size (DataFrame): DataFrame con los numeros de observaciones utilizados para calcular tasas de acceso a obra social (M/V). Expresado en valores absolutos
        ratio (DataFrame): DataFrame con las proporciones de tasas de acceso a obra social (M/V). Expresado en % [0-100]
        error (DataFrame): DataFrame con los errores asociados a las proporciones. Expresado en % [0-100]. 
                            LI: Limite inferior, LS: Limite superior, ME: Margen de error, ER: Error relativo (CV)

    """
    
    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
        
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'
    
    df_estado = df_temp[(df_temp['ESTADO'] == 1) & (df_temp['CAT_OCUP'] == 3)]          # personas ocupadas asalariadas
    numerador = df_estado[df_estado['PP07G4']==1].groupby(['CH04', var])[pondera].sum().unstack(level=0)

    denominador = df_estado.groupby(['CH04', var])[pondera].sum().unstack(level=0)

    tasa = numerador.div(denominador, fill_value=np.nan)
    tasa = tasa.rename(columns= {1:'Varon', 2:'Mujer'})

    size = df_estado[df_estado['PP07G4']==1].groupby(['CH04', var]).size().unstack(level=0)
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
    ratio.rename(columns={0: 'Acceso a OS'}, inplace=True)

    standard_error = np.sqrt(((1 / n1) * (p1 * (1 - p1))) + ((1 / n2) * (p2 * (1 - p2))))

    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z * standard_error
    lower_bound = p1/p2 - margin_of_error
    upper_bound = p1/p2 + margin_of_error

    relative_standard_error = margin_of_error/(p1/p2)
    error = pd.concat([lower_bound, upper_bound, margin_of_error, relative_standard_error], axis=1)
    error.columns = ['LI', 'LS', 'ME', 'ER']

    return tasa*100, size, ratio*100, error*100


# In[ ]:


def calcular_tabla_poblacion_ocupada_asalariada(df, tipo='Aglomerado', base='Individual', pool=False):

    """
    Calcula la tabla de .... 

    Args:
        df (DataFrame): DataFrame de entrada que contiene los datos de la EPH.
        tipo (str): Tipo de encuesta de la EPH, 'Aglomerado' o 'Urbano'. Por defecto: 'Aglomerado'.
        base (str): Tipo de base de la EPH, 'Individual' o 'Hogar'. Por defecto: 'Individual'.
        confidence_level (float): Nivel de confianza para el cálculo de errores. Por defecto: 0.95.

    Returns:
        tabla (DataFrame): DataFrame con la tabla_pob_ocupada_asalariada

    """
    if tipo == 'Aglomerado':
        var = 'AGLOMERADO'
    elif tipo == 'Urbano':
        var = 'PROVINCIA'

    df_temp = df.query('CH06 >= 14')
        
    if pool:
        pondera = 'PONDERA_new'
    else:
        pondera = 'PONDERA'
    
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


# In[ ]:


def indicadores_insercion_laboral(df, dict_cod_provincia, tipo='Urbano', base='Individual', confidence_level=0.95, pool=False):

    tasa, size, ratio, error = ratio_tasa_actividad(df, tipo=tipo, base=base, confidence_level=confidence_level, pool=pool)
    ratio_actividad = pd.concat([tasa, size, ratio, error], axis=1)

    tasa, size, ratio, error = formalidad(df, tipo=tipo, base=base, confidence_level=confidence_level, pool=pool)
    empleo_asalariado_con_descuento = pd.concat([tasa, size, ratio, error], axis=1)

    tasa, size, ratio, error = jornada_laboral(df, tipo=tipo, base=base, confidence_level=confidence_level, pool=pool)
    horas_remuneradas_media = pd.concat([tasa, size, ratio, error], axis=1)

    tasa, size, ratio, error = ingreso_salarial_promedio(df, tipo=tipo, base=base, confidence_level=confidence_level, pool=pool)
    ingreso_salarial_media = pd.concat([tasa, size, ratio, error], axis=1)

    if horas_remuneradas_media['Horas promedio remuneradas'].dtype == 'object':
        horas_remuneradas_media['Horas promedio remuneradas'] = horas_remuneradas_media['Horas promedio remuneradas'].astype(float)
    
    variables = [ratio_actividad, empleo_asalariado_con_descuento, horas_remuneradas_media, ingreso_salarial_media]
    for variable in variables:
        variable.index = variable.index.map(dict_cod_provincia)

    return ratio_actividad, empleo_asalariado_con_descuento, horas_remuneradas_media, ingreso_salarial_media


# In[ ]:


def calcular_insercion_laboral(df, dict_cod_provincia, tipo='Urbano', base='Individual', confidence_level=0.95, pool=False):

    tasa, size, ratio, error = ratio_tasa_actividad(df, tipo=tipo, base=base, confidence_level=confidence_level, pool=pool)
    ratio_actividad = pd.concat([tasa, size, ratio, error], axis=1)

    tasa, size, ratio, error = formalidad(df, tipo=tipo, base=base, confidence_level=confidence_level, pool=pool)
    empleo_asalariado_con_descuento = pd.concat([tasa, size, ratio, error], axis=1)

    tasa, size, ratio, error = jornada_laboral(df, tipo=tipo, base=base, confidence_level=confidence_level, pool=pool)
    horas_remuneradas_media = pd.concat([tasa, size, ratio, error], axis=1)

    tasa, size, ratio, error = ingreso_salarial_promedio(df, tipo=tipo, base=base, confidence_level=confidence_level, pool=pool)
    ingreso_salarial_media = pd.concat([tasa, size, ratio, error], axis=1)

    if horas_remuneradas_media['Horas promedio remuneradas'].dtype == 'object':
        horas_remuneradas_media['Horas promedio remuneradas'] = horas_remuneradas_media['Horas promedio remuneradas'].astype(float)

    insercion_laboral = (ratio_actividad['Actividad'] + empleo_asalariado_con_descuento['Formalidad'] +                          horas_remuneradas_media['Horas promedio remuneradas'] + ingreso_salarial_media['Ingreso laboral promedio']) / 4
    insercion_laboral.name = 'Insercion laboral'

    variables = [ratio_actividad, empleo_asalariado_con_descuento, horas_remuneradas_media, ingreso_salarial_media, insercion_laboral]
    for variable in variables:
        variable.index = variable.index.map(dict_cod_provincia)

    df_insercion = pd.concat([ratio_actividad['Actividad'],                                 empleo_asalariado_con_descuento['Formalidad'],                                 horas_remuneradas_media['Horas promedio remuneradas'],                                 ingreso_salarial_media['Ingreso laboral promedio'],                                 insercion_laboral], axis=1)
    
    return df_insercion


# In[ ]:


## pluriempleo

# df_temp = df_people_pool.query('CH06 >= 16 & CH06 < 65')
# df_estado = df_temp[(df_temp['ESTADO'] == 1) & (df_temp['CAT_OCUP'] != 1)& (df_temp['CAT_OCUP'] != 4)]          # personas ocupadas cuentapropistas, obreros o empleados

# # Filtrar y contar las combinaciones por género
# resultados = df_estado.groupby(['CH04', 'CAT_OCUP', 'PP03C'])['PONDERA_new'].sum().unstack(level='CH04', fill_value=0)

# resultados.columns = ['Varones', 'Mujeres']
# resultados = resultados.reset_index()

# resultados['Total Fila'] = resultados['Varones'] + resultados['Mujeres']
# resultados['Combinación'] = resultados.apply(lambda row: f"CAT_OCUP == {row['CAT_OCUP']}, PP03C == {row['PP03C']}", axis=1)

# resultados.loc['Total Col'] = resultados.sum(numeric_only=True, axis=0)
# resultados.loc[:, '% Varones'] = resultados['Varones']/resultados.loc['Total Col', 'Varones'] * 100
# resultados.loc[:, '% Mujeres'] = resultados['Mujeres']/resultados.loc['Total Col', 'Mujeres'] * 100

# resultados = resultados[['Combinación', 'Varones', '% Varones', 'Mujeres', '% Mujeres', 'Total Fila']]

# # Aplicar formato a los valores en la tabla
# tabla_pob_ocupada = resultados.style.format({
#     'Varones': '{:,.0f}',
#     '% Varones': '{:,.2f}',
#     'Mujeres': '{:,.0f}',
#     '% Mujeres': '{:,.2f}',
#     'Total Fila': '{:,.0f}'
# })

# tabla_pob_ocupada

# [P21 mean M y V, TOT_P12 mean M y V]

