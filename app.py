import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ==============================
# Variables globales para modelo
# ==============================
modelo = None
tokenizer = None
CLASES = ["NORMAL", "REGULAR", "TÓXICO"]
MAX_LEN = 10

def cargar_modelo():
    """Carga el modelo y tokenizer solo una vez bajo demanda"""
    global modelo, tokenizer
    if modelo is None or tokenizer is None:
        import tensorflow as tf
        import pickle

        modelo = tf.keras.models.load_model("modelo.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

# ==============================
# Ruta principal
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

# ==============================
# Ruta para subir archivo
# ==============================
@app.route("/upload", methods=["POST"])
def upload():
    cargar_modelo()  # Cargar modelo solo al primer request

    if "file" not in request.files:
        return "No se encontró archivo"

    file = request.files["file"]
    if file.filename == "":
        return "Archivo vacío"

    try:
        df = pd.read_excel(file)
        if "texto" not in df.columns:
            return "El Excel debe tener una columna llamada 'texto'"

        textos = df["texto"].astype(str).tolist()

        # Tokenización
        secuencias = tokenizer.texts_to_sequences(textos)
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            secuencias,
            maxlen=MAX_LEN,
            padding="post"
        )

        # Predicción
        predicciones = modelo.predict(padded)
        clases_predichas = np.argmax(predicciones, axis=1)

        resultados = pd.Series([CLASES[i] for i in clases_predichas]).value_counts()
        resultados = resultados.reindex(CLASES, fill_value=0)

        # ==============================
        # Generar gráfica
        # ==============================
        plt.figure(figsize=(6, 4))
        colores = ["#22c55e", "#eab308", "#ef4444"]
        plt.bar(resultados.index, resultados.values, color=colores)
        plt.title("Resultados del Análisis", fontsize=14)
        plt.xlabel("Clase")
        plt.ylabel("Cantidad")
        plt.xticks(rotation=0)
        for i, v in enumerate(resultados.values):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.tight_layout()

        # Crear carpeta static si no existe
        if not os.path.exists("static"):
            os.makedirs("static")

        nombre_imagen = f"grafica_{uuid.uuid4().hex}.png"
        ruta = os.path.join("static", nombre_imagen)
        plt.savefig(ruta)
        plt.close()

        return render_template("index.html", grafica=ruta)

    except Exception as e:
        return f"Error interno: {str(e)}"

# ==============================
# Ejecutar servidor localmente
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
