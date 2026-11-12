"""EDA del dataset de vinos: carga, estadísticas, gráficas y limpieza."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET = "quality"       
DATA_PATH = Path("data") / "WineQT.csv"
FIGURAS_DIR = Path("figuras")


def cargar_datos(ruta_csv=DATA_PATH):
    """Carga el dataset WineQT desde un CSV."""
    return pd.read_csv(ruta_csv)


def mostrar_estadisticas(df):
    print("Dimensiones:", df.shape)
    print("\nPrimeras filas:\n", df.head())
    print("\nEstadísticas descriptivas:\n", df.describe().T)
    print("\nValores nulos por columna:\n", df.isnull().sum())
    print("\nDistribución de clases:\n", df[TARGET].value_counts())


def limpiar_datos(df):
    """Quita la columna Id (solo es un identificador), duplicados y nulos."""
    antes = len(df)
    df = df.drop(columns=["Id"], errors="ignore")
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    print(f"\nLimpieza: {antes - len(df)} filas eliminadas, quedan {len(df)}.")
    return df


def graficar_histogramas(df):
    df.drop(columns=TARGET).hist(bins=20, figsize=(14, 10))
    plt.suptitle("Distribución de las variables")
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "histogramas.png")


def graficar_boxplots(df):
    columnas = df.drop(columns=TARGET).columns
    fig, ejes = plt.subplots(3, 5, figsize=(16, 9))
    for eje, col in zip(ejes.flatten(), columnas):
        sns.boxplot(y=df[col], ax=eje)
    for eje in ejes.flatten()[len(columnas):]:
        eje.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "boxplots.png")


def graficar_correlacion(df):
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "correlacion.png")


def main():
    FIGURAS_DIR.mkdir(exist_ok=True)
    df = cargar_datos()
    mostrar_estadisticas(df)
    df = limpiar_datos(df)
    graficar_histogramas(df)
    graficar_boxplots(df)
    graficar_correlacion(df)
    print(f"\nGráficas guardadas en '{FIGURAS_DIR}/'.")
    plt.show()


if __name__ == "__main__":
    main()
