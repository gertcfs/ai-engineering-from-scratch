# app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Bienvenido al contenedor de IA", "status": "ok"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict")
def predict():
    return jsonify({"prediction": "ejemplo", "model": "no-cargado-aún"})

if __name__ == "__main__":
    # host="0.0.0.0" es OBLIGATORIO en Docker: así el servidor escucha
    # en todas las interfaces y el mapeo de puertos (-p) funciona.
    app.run(host="0.0.0.0", port=5000)
