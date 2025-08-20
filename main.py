import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("Dashboard de Entrenamiento de Regresión")

# Generar datos sintéticos 30x7
np.random.seed(42)
X = np.random.rand(30, 6)  # 6 features
y = np.random.rand(30)     # target

df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(6)])
df["Target"] = y

st.subheader("Datos de Entrenamiento (30x7)")
st.dataframe(df)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.subheader("Resultados del Modelo")
st.write(f"Mean Squared Error en test: {mse:.4f}")

# Mostrar coeficientes
st.subheader("Coeficientes del Modelo")
coef_df = pd.DataFrame({
    "Feature": [f"Feature_{i+1}" for i in range(6)],
    "Coeficiente": model.coef_
})

st.dataframe(coef_df)

# Gráfico de dispersión de las predicciones vs valores reales
st.subheader("Predicciones vs Valores Reales")
pred_df = pd.DataFrame({"Valores Reales": y_test, "Predicciones": y_pred})
st.line_chart(pred_df)

# Gráfico de los coeficientes
st.subheader("Gráfico de Coeficientes")
st.bar_chart(coef_df.set_index("Feature")["Coeficiente"].abs())

# Ajuste de hiperparámetros (ejemplo simple con alpha de Ridge)
from sklearn.linear_model import Ridge
alpha = st.slider("Selecciona el valor de alpha para Ridge:", 0.01, 10.0, 0.1)
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train, y_train)
y_ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_ridge_pred)

st.subheader("Resultados del Modelo Ridge")
st.write(f"Mean Squared Error en test: {ridge_mse:.4f}")

# Comparativa de errores
st.subheader("Comparativa de Errores")
error_df = pd.DataFrame({
    "Modelo": ["Linear", "Ridge"],
    "Mean Squared Error": [mse, ridge_mse]
})

st.bar_chart(error_df.set_index("Modelo"))

# Predicción de nuevos datos
st.subheader("Predicción de Nuevos Datos")
new_data = st.text_input("Ingresa los valores de las features separadas por comas:")
if new_data:
    new_data_np = np.array([[float(i) for i in new_data.split(",")]])
    prediccion_nueva = model.predict(new_data_np)
    st.write(f"Predicción: {prediccion_nueva[0]:.4f}")