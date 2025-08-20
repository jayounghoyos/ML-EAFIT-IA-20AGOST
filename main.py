import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor  # <-- Importar Decision Tree

st.set_page_config(
    page_title="Entrenamiento de Regresión",
    page_icon="📈",
    layout="wide"
)

st.title("Dashboard de Entrenamiento de Regresión")

# Leer datos desde el CSV
df = pd.read_csv("data.csv")

# Separar features y target
X = df.drop("Target", axis=1).values
y = df["Target"].values

# --- EDA Creativo ---
st.header("Análisis Exploratorio de Datos (EDA)")

# Estadísticas descriptivas
st.subheader("Estadísticas Descriptivas")
st.dataframe(df.describe().T)

# Histograma de cada feature
st.subheader("Distribución de las Features")
selected_feature = st.selectbox("Selecciona una feature para ver su histograma:", df.columns[:-1])
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(df[selected_feature], bins=10, color="skyblue", edgecolor="black")
ax_hist.set_title(f"Histograma de {selected_feature}")
st.pyplot(fig_hist)

# Matriz de correlación
st.subheader("Matriz de Correlación")
corr = df.corr()
st.dataframe(corr)
st.write("Mapa de calor de correlación:")
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
st.pyplot(fig)

# Pairplot (muestra solo si el usuario lo solicita)
if st.checkbox("Mostrar pairplot (puede tardar unos segundos)"):
    fig2 = sns.pairplot(df, diag_kind="hist", corner=True, palette="viridis")
    st.pyplot(fig2.figure)

# Boxplot de las features
st.subheader("Boxplot de las Features")
fig3, ax3 = plt.subplots()
df.iloc[:, :-1].boxplot(ax=ax3)
st.pyplot(fig3)

# Gráfico de dispersión entre dos features seleccionables
st.subheader("Gráfico de dispersión entre dos features")
feature_x = st.selectbox("Eje X", df.columns[:-1], key="x")
feature_y = st.selectbox("Eje Y", df.columns[:-1], key="y")
fig_scatter, ax_scatter = plt.subplots()
ax_scatter.scatter(df[feature_x], df[feature_y], color="purple")
ax_scatter.set_xlabel(feature_x)
ax_scatter.set_ylabel(feature_y)
ax_scatter.set_title(f"Scatter: {feature_x} vs {feature_y}")
st.pyplot(fig_scatter)

# Datos de entrenamiento
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
    "Feature": df.columns[:-1],
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
alpha = st.slider("Selecciona el valor de alpha para Ridge:", 0.01, 10.0, 0.1)
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train, y_train)
y_ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_ridge_pred)

st.subheader("Resultados del Modelo Ridge")
st.write(f"Mean Squared Error en test: {ridge_mse:.4f}")

# Entrenamiento del modelo Decision Tree
st.subheader("Modelo Decision Tree")
st.markdown(
    """
    **¿Qué necesitas para usar un árbol de decisión?**
    - Tener instalado `scikit-learn` (ya está en requirements.txt).
    - Importar `DecisionTreeRegressor` de `sklearn.tree`.
    - Ajustar hiperparámetros como `max_depth` para evitar sobreajuste.
    - Los árboles de decisión pueden modelar relaciones no lineales y son fáciles de interpretar.
    """
)
max_depth = st.slider("Selecciona la profundidad máxima del árbol:", 1, 10, 3)
dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
dt_model.fit(X_train, y_train)
y_dt_pred = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, y_dt_pred)
st.write(f"Mean Squared Error en test (Decision Tree): {dt_mse:.4f}")

# Comparativa de errores (ahora incluye Decision Tree)
st.subheader("Comparativa de Errores")
error_df = pd.DataFrame({
    "Modelo": ["Linear", "Ridge", "Decision Tree"],
    "Mean Squared Error": [mse, ridge_mse, dt_mse]
})
st.bar_chart(error_df.set_index("Modelo"))

# Predicción de nuevos datos con Decision Tree
st.subheader("Predicción de Nuevos Datos (Decision Tree)")
st.caption("Ejemplo: 0.1,0.2,0.3,0.4,0.5,0.6")
new_data_dt = st.text_input("Ingresa los valores de las features separadas por comas (Decision Tree):")
if new_data_dt:
    try:
        new_data_np_dt = np.array([[float(i) for i in new_data_dt.split(",")]])
        prediccion_nueva_dt = dt_model.predict(new_data_np_dt)
        st.write(f"Predicción (Decision Tree): {prediccion_nueva_dt[0]:.4f}")
    except Exception as e:
        st.error(f"Error en la entrada: {e}")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)