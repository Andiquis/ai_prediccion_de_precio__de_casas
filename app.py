from flask import Flask, render_template, request
import numpy as np
import joblib

# Cargar los modelos entrenados
linear_model = joblib.load(r'D:\ai_prediccion_de_precio__de_casas\linear_model.pkl')
poly_model = joblib.load(r'D:\ai_prediccion_de_precio__de_casas\poly_model.pkl')
poly_transform = joblib.load(r'D:\ai_prediccion_de_precio__de_casas\poly_transform.pkl')
tree_model = joblib.load(r'D:\ai_prediccion_de_precio__de_casas\tree_model.pkl')
rf_model = joblib.load(r'D:\ai_prediccion_de_precio__de_casas\rf_model.pkl')
xg_model = joblib.load(r'D:\ai_prediccion_de_precio__de_casas\xg_model.pkl')

# Inicializar la aplicación Flask
app = Flask(__name__)

# Ruta principal que carga la interfaz web
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para recibir los datos y hacer la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    model_name = request.form['model']
    house_size = float(request.form['house_size'])
    num_bedrooms = int(request.form['num_bedrooms'])
    num_bathrooms = int(request.form['num_bathrooms'])
    garage_spaces = int(request.form['garage_spaces'])
    age = int(request.form['age'])
    location_score = float(request.form['location_score'])

    # Crear el array de entrada
    input_data = np.array([[house_size, num_bedrooms, num_bathrooms, garage_spaces, age, location_score]])

    # Seleccionar el modelo y predecir
    if model_name == 'Regresión Lineal':
        prediction = linear_model.predict(input_data)
    elif model_name == 'Regresión Polinómica':
        prediction = poly_model.predict(poly_transform.transform(input_data))
    elif model_name == 'Árbol de Decisión':
        prediction = tree_model.predict(input_data)
    elif model_name == 'Random Forest':
        prediction = rf_model.predict(input_data)
    elif model_name == 'XGBoost':
        prediction = xg_model.predict(input_data)
    
    # Retornar el resultado
    return render_template('index.html', prediction_text=f'El precio predicho es: ${prediction[0]:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
