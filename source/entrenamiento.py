"""Entrenamiento de modelos de clasificación de vinos."""
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eda import TARGET, cargar_datos, limpiar_datos

RANDOM_STATE = 42
MODELO_PATH = Path("modelos") / "mejor_modelo.joblib"


def dividir_datos(df, test_size=0.2):
    """Divide en entrenamiento y prueba (misma semilla en todos los scripts)."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def calcular_metricas(y_real, y_pred):
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_real, y_pred, average="macro", zero_division=0),
    }


def obtener_modelos():
    return {
        "Regresión logística": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def main():
    df = limpiar_datos(cargar_datos())
    X_train, X_test, y_train, y_test = dividir_datos(df)

    resultados = {}
    for nombre, modelo in obtener_modelos().items():
        modelo.fit(X_train, y_train)
        metricas = calcular_metricas(y_test, modelo.predict(X_test))
        resultados[nombre] = (modelo, metricas)
        print(f"\n{nombre}")
        for k, v in metricas.items():
            print(f"  {k}: {v:.4f}")

    mejor = max(resultados, key=lambda n: resultados[n][1]["f1"])
    MODELO_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(resultados[mejor][0], MODELO_PATH)
    print(f"\nMejor modelo: {mejor} (guardado en {MODELO_PATH})")


if __name__ == "__main__":
    main()
