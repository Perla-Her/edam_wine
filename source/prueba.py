"""Evaluación del modelo guardado sobre los datos de prueba."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from eda import cargar_datos, limpiar_datos
from entrenamiento import MODELO_PATH, calcular_metricas, dividir_datos


def main():
    if not MODELO_PATH.exists():
        raise SystemExit("No hay modelo guardado. Ejecuta primero src/entrenamiento.py")

    modelo = joblib.load(MODELO_PATH)
    df = limpiar_datos(cargar_datos())
    _, X_test, _, y_test = dividir_datos(df)
    y_pred = modelo.predict(X_test)

    print("\nMétricas finales:")
    for k, v in calcular_metricas(y_test, y_pred).items():
        print(f"  {k}: {v:.4f}")
    print("\nReporte por clase:\n", classification_report(y_test, y_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Matriz de confusión")
    plt.tight_layout()
    Path("figuras").mkdir(exist_ok=True)
    plt.savefig("figuras/matriz_confusion.png")
    plt.show()


if __name__ == "__main__":
    main()
