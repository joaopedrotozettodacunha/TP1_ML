import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

dados = pd.read_csv(r"C:\Users\joaop\OneDrive\Área de Trabalho\TP1 Aprendizado de Máquina\dados_20mil.csv")

dados_numericos = dados.select_dtypes(include = ["number"])
dados_numericos = dados_numericos.drop(columns = ["ano"])
dados_categoricos = dados.select_dtypes(exclude = ["number"])


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

    # evita divisão por zero
    mask = y_true != 0

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100



param_grid = {
    "n_neighbors": [10],
    "weights": ["uniform"]
}

grid = list(ParameterGrid(param_grid))

# Imputador
resultados = []

for i, params in enumerate(grid):

    print(f"\nTestando combinação {i+1}/{len(grid)}")
    print(params)

    imputador = KNNImputer(
        **params
    )

    # imputação
    dados_imputados = imputador.fit_transform(dados_com_missing)

    dados_imputados = pd.DataFrame(
        dados_imputados,
        columns=dados_original.columns
    )


   

    y_true = dados_original.values[mask]

    y_pred = dados_imputados.values[mask]

    

    valores_validos = (
        ~np.isnan(y_true)
    ) & (
        ~np.isnan(y_pred)
    )

    y_true = y_true[valores_validos]

    y_pred = y_pred[valores_validos]

    

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mape = calcular_mape(y_true, y_pred)

    resultados.append({
    "n_neighbors": params["n_neighbors"],
    "weights": params["weights"],
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "MAPE": mape
})



resultados_df = pd.DataFrame(resultados)

# ordenar pelo menor RMSE
resultados_df = resultados_df.sort_values(by="RMSE")

print("\n================ MELHORES RESULTADOS ================\n")

print(resultados_df.head(10))


resultados_df.to_csv(
    r"C:\Users\joaop\OneDrive\Área de Trabalho\TP1 Aprendizado de Máquina\resultado_gridsearch_knn.csv",
    index=False
)



melhores_params = resultados_df.iloc[0]

print("\nMelhores hiperparâmetros:")
print(melhores_params)

imputador_final = KNNImputer(
    random_state=17,
    max_iter=int(melhores_params["max_iter"]),
    tol=melhores_params["tol"],
    initial_strategy=melhores_params["initial_strategy"],
    imputation_order=melhores_params["imputation_order"]
)


dados_imputados_finais = imputador_final.fit_transform(dados_numericos)

dados_imputados_finais = pd.DataFrame(
    dados_imputados_finais,
    columns=dados_numericos.columns
)


dados_categoricos = dados.select_dtypes(exclude=["number"])

dados_final = pd.concat(
    [dados_categoricos, dados_imputados_finais],
    axis=1
)


dados_final.to_csv(
    r"C:\Users\joaop\OneDrive\Área de Trabalho\TP1 Aprendizado de Máquina\base_inputada_knn_melhor.csv",
    index=False
)
