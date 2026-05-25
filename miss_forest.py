import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import ParameterGrid

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)



dados = pd.read_csv(
    r"C:\Users\joaop\OneDrive\Área de Trabalho\TP1 Aprendizado de Máquina\dados_20mil.csv"
)



dados_numericos = dados.select_dtypes(include=["number"]).copy()

if "ano" in dados_numericos.columns:
    dados_numericos = dados_numericos.drop(columns=["ano"])



np.random.seed(42)

dados_original = dados_numericos.copy()

mask = (
    np.random.rand(*dados_original.shape) < 0.40
) & (~dados_original.isna())

dados_com_missing = dados_original.copy()

dados_com_missing[mask] = np.nan



def calcular_mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = np.abs(y_true) > 0.1

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100



param_grid = {
    "max_iter": [5],
    "n_estimators": [50],
    "max_depth": [10]
}

grid = list(ParameterGrid(param_grid))



resultados = []

for i, params in enumerate(grid):

    print(f"\nTestando combinação {i+1}/{len(grid)}")
    print(params)

    rf = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=17,
        n_jobs=-1
    )

    imputador = IterativeImputer(
        estimator=rf,
        max_iter=params["max_iter"],
        n_nearest_features=20,
        random_state=17
    )

    # imputação
    dados_imputados = imputador.fit_transform(
        dados_com_missing
    )

    dados_imputados = pd.DataFrame(
        dados_imputados,
        columns=dados_original.columns
    )


    y_true = dados_original.values[mask]

    y_pred = dados_imputados.values[mask]

    # remover NaNs
    validos = (
        ~np.isnan(y_true)
    ) & (
        ~np.isnan(y_pred)
    )

    y_true = y_true[validos]

    y_pred = y_pred[validos]

 

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mape = calcular_mape(y_true, y_pred)

    resultados.append({
        "max_iter": params["max_iter"],
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    })



resultados_df = pd.DataFrame(resultados)

resultados_df = resultados_df.sort_values(by="RMSE")

print("\n=========== MELHORES RESULTADOS ===========\n")

print(resultados_df.head(10))

# salvar resultados
resultados_df.to_csv(
    r"C:\Users\joaop\OneDrive\Área de Trabalho\TP1 Aprendizado de Máquina\resultado_missforest.csv",
    index=False
)


melhor = resultados_df.iloc[0]

print("\nMelhores hiperparâmetros:")
print(melhor)

rf_final = RandomForestRegressor(
    n_estimators=int(melhor["n_estimators"]),
    max_depth=None if pd.isna(melhor["max_depth"]) else int(melhor["max_depth"]),
    random_state=17,
    n_jobs=-1
)

imputador_final = IterativeImputer(
    estimator=rf_final,
    max_iter=int(melhor["max_iter"]),
    random_state=17
)

# imputar base completa
dados_imputados_finais = imputador_final.fit_transform(
    dados_numericos
)

dados_imputados_finais = pd.DataFrame(
    dados_imputados_finais,
    columns=dados_numericos.columns
)

# juntar categóricas
dados_categoricos = dados.select_dtypes(exclude=["number"])

dados_final = pd.concat(
    [dados_categoricos, dados_imputados_finais],
    axis=1
)

dados_final.to_csv(
    r"C:\Users\joaop\OneDrive\Área de Trabalho\TP1 Aprendizado de Máquina\base_imputada_rf.csv",
    index=False
)

print("\nBase imputada salva com sucesso!")
