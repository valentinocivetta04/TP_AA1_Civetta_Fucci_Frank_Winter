"""
Análisis de coherencia de outliers (CRIM, ZN, B) - TP1 Regresión

Este script asume que ya tenés cargado tu DataFrame como 'df' (el CSV crudo)
y que ya generaste 'df_limpio' con el dropna de MEDV, tal como en tu notebook:

    df = pd.read_csv('house-prices-tp.csv')
    df_limpio = df.dropna(subset=['MEDV'])

La idea: la regla de IQR sola marca demasiados "outliers" en variables muy
asimétricas (CRIM, ZN, B) que en realidad son la cola larga de una
distribución real, no errores. Este análisis agrega un segundo filtro:
para cada outlier, chequea si su relación con el target (MEDV) y con otras
variables correlacionadas por dominio va en la dirección que el propio EDA
(matriz de correlación) predice. Si rompe esa relación en 2 o más variables
al mismo tiempo, se marca como "incoherente" y es candidato a revisión
manual (no automáticamente eliminado).
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def detectar_outliers_iqr(df, columna):
    """
    Marca los outliers de 'columna' con la regla clásica de IQR
    (Q1 - 1.5*IQR, Q3 + 1.5*IQR) y guarda de qué lado cae cada uno
    ('alto' o 'bajo'), que se usa después para juzgar coherencia.
    """
    serie = df[columna].dropna()
    Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
    IQR = Q3 - Q1
    lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    df_outliers = df[(df[columna] < lim_inf) | (df[columna] > lim_sup)].copy()
    df_outliers['lado'] = np.where(df_outliers[columna] > lim_sup, 'alto', 'bajo')
    return df_outliers, lim_inf, lim_sup


def chequear_coherencia(df, columna, direccion, relaciones=None, target='MEDV'):
    """
    Para los outliers de 'columna', evalúa si el valor extremo es coherente
    con la relación esperada respecto al target y a otras variables
    correlacionadas por dominio (según la matriz de correlación del EDA).

    Parámetros
    ----------
    df : DataFrame completo (uso df_limpio, sin nulos en MEDV)
    columna : variable a analizar (ej: 'CRIM', 'ZN', 'B')
    direccion : +1 si 'columna' correlaciona POSITIVO con el target,
                -1 si correlaciona NEGATIVO (esto sale de tu matriz de
                correlación, no es un supuesto arbitrario)
    relaciones : dict opcional {nombre_var: signo_esperado} con otras
                 variables relacionadas por dominio
                 (ej: {'LSTAT': 1, 'RM': -1} para CRIM)
    target : variable objetivo, default 'MEDV'

    Devuelve
    --------
    df_outliers : todos los outliers de 'columna', con su puntaje de
                  incoherencia (cuántas relaciones esperadas rompe)
    df_incoherentes : subconjunto que rompe 2 o más relaciones esperadas
                      -> son los candidatos a revisar, no a eliminar
                      automáticamente
    """
    df_outliers, lim_inf, lim_sup = detectar_outliers_iqr(df, columna)
    signo_lado = np.where(df_outliers['lado'] == 'alto', 1, -1)

    def marca_rompe(var, signo_esperado):
        # Compara cada outlier contra la mediana GLOBAL de 'var' (no la de
        # los outliers), para juzgarlo contra el comportamiento típico de
        # todo el dataset.
        mediana_var = df[var].median(skipna=True)
        diff = df_outliers[var] - mediana_var
        signo_fila = np.sign(diff)
        signo_esperado_fila = signo_lado * signo_esperado
        # Las filas con NaN en 'var' no se cuentan como incoherentes:
        # no hay evidencia suficiente para juzgarlas.
        valido = diff.notna() & (signo_fila != 0)
        return valido & (np.sign(signo_esperado_fila) != signo_fila)

    puntaje = marca_rompe(target, direccion).astype(int)
    df_outliers[f'rompe_{target}'] = marca_rompe(target, direccion)

    relaciones = relaciones or {}
    for var, signo_var in relaciones.items():
        rompe = marca_rompe(var, signo_var)
        df_outliers[f'rompe_{var}'] = rompe
        puntaje = puntaje + rompe.astype(int)

    df_outliers['puntaje_incoherencia'] = puntaje
    df_incoherentes = (
        df_outliers[df_outliers['puntaje_incoherencia'] >= 2]
        .sort_values('puntaje_incoherencia', ascending=False)
    )

    print(f"--- {columna} ---")
    print(f"  límites IQR: [{lim_inf:.2f}, {lim_sup:.2f}]  -> {len(df_outliers)} outliers totales")
    print(f"  incoherentes (rompen >=2 relaciones esperadas): {len(df_incoherentes)}")

    return df_outliers, df_incoherentes


def graficar_coherencia(df, columna, df_outliers, df_incoherentes, target='MEDV'):
    """
    Extiende el scatterplot 'Impacto de X en el valor de la propiedad' que
    ya tenías, distinguiendo tres grupos: datos normales, outliers coherentes
    (se conservan) y outliers incoherentes (candidatos a revisión manual).
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=columna, y=target, color='lightgrey', alpha=0.6,
                     label='Datos normales')

    coherentes = df_outliers.drop(df_incoherentes.index)
    sns.scatterplot(data=coherentes, x=columna, y=target, color='orange', edgecolor='black',
                     label='Outlier coherente (conservar)')

    sns.scatterplot(data=df_incoherentes, x=columna, y=target, color='red', edgecolor='black',
                     s=90, label='Outlier incoherente (revisar)')

    plt.title(f'Coherencia de outliers de {columna} respecto a {target}')
    plt.xlabel(columna)
    plt.ylabel(target)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('house-prices-tp.csv')
    # Igual que en el resto del pipeline: nunca se imputa MEDV, se dropean
    # sus nulos primero, antes de analizar cualquier otra cosa.
    df_limpio = df.dropna(subset=['MEDV']).copy()

    # Signos esperados según la matriz de correlación del EDA (Sección
    # "¿Cuál es la correlación entre las variables?" de tu propio análisis).
    # direccion: signo esperado de la correlación de la columna con MEDV.
    # relaciones: otras variables correlacionadas por dominio con la columna.
    config = {
        'CRIM': dict(direccion=-1, relaciones={'RM': -1, 'LSTAT': 1}),
        'ZN':   dict(direccion=+1, relaciones={'INDUS': -1, 'LSTAT': -1}),
        'B':    dict(direccion=+1, relaciones={'LSTAT': -1}),
    }

    resultados = {}
    for col, cfg in config.items():
        outliers, incoherentes = chequear_coherencia(df_limpio, col, **cfg)
        resultados[col] = (outliers, incoherentes)
        graficar_coherencia(df_limpio, col, outliers, incoherentes)

    # Triangulación: una fila que rompe el patrón en MÁS DE UNA variable
    # independiente es un candidato mucho más fuerte a revisión que una
    # fila que solo lo rompe en una. No es automático que se elimine: es
    # el punto de partida para decidir con criterio (diagnóstico de
    # influencia, regresión robusta, o exclusión justificada y comparada).
    print()
    print("--- Triangulación: filas incoherentes en MÁS DE UNA variable ---")
    indices = [set(incoh.index) for _, incoh in resultados.values()]
    interseccion = set.union(*[
        indices[i] & indices[j]
        for i in range(len(indices)) for j in range(i + 1, len(indices))
    ])
    print(f"Filas candidatas a revisión prioritaria: {sorted(interseccion)}")
    cols_mostrar = ['CRIM', 'ZN', 'B', 'RM', 'LSTAT', 'INDUS', 'MEDV']
    print(df_limpio.loc[sorted(interseccion), cols_mostrar].to_string())
