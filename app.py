# app.py

import streamlit as st
import pandas as pd
from prophet import Prophet
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# --- CONFIGURACIÓN DE LA APP Y CREDENCIALES
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Predictor SEO", layout="wide")
st.title("📈 Predictor de Métricas SEO con Prophet")

# --- CONFIGURACIÓN DE OAUTH (MODIFICADO PARA LEER DESDE ST.SECRETS) ---
CLIENT_ID = st.secrets["google_credentials"]["CLIENT_ID"]
CLIENT_SECRET = st.secrets["google_credentials"]["CLIENT_SECRET"]
REDIRECT_URI = "https://predictor-seo-prophet.streamlit.app"
SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]


# -----------------------------------------------------------------------------
# --- FUNCIONES DE LÓGICA (Sin cambios)
# -----------------------------------------------------------------------------

def autenticar_google():
    """
    Gestiona el flujo de autenticación de OAuth 2.0 con Google.
    Devuelve las credenciales si el login es exitoso, de lo contrario None.
    """
    flow = Flow.from_client_config(
        client_config={
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI],
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

    query_params = st.query_params
    code = query_params.get("code")

    if 'credentials' not in st.session_state and not code:
        auth_url, _ = flow.authorization_url(prompt="consent")
        st.link_button("Login con Google", auth_url, use_container_width=True)
        st.info("Necesitas iniciar sesión con tu cuenta de Google para usar la aplicación.")
        return None

    elif not 'credentials' in st.session_state and code:
        try:
            flow.fetch_token(code=code)
            st.session_state['credentials'] = flow.credentials
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error durante el login: {e}")
            return None

    return st.session_state.get('credentials')

def listar_sitios_gsc(credentials):
    """Obtiene la lista de sitios de GSC a los que el usuario tiene acceso."""
    try:
        service = build("searchconsole", "v1", credentials=credentials)
        site_list = service.sites().list().execute()
        return [s["siteUrl"] for s in site_list.get("siteEntry", [])]
    except Exception as e:
        st.error(f"No se pudo obtener la lista de sitios: {e}")
        return []

def obtener_metricas(credentials, site_url, start_date, end_date):
    """Obtiene los datos de GSC para una propiedad y rango de fechas."""
    try:
        service = build("searchconsole", "v1", credentials=credentials)
        request = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": ["date"],
            "rowLimit": 25000
        }
        response = service.searchanalytics().query(siteUrl=site_url, body=request).execute()
        
        rows = response.get("rows", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Expande la columna 'keys' que contiene la fecha 'ds'
        df_keys = pd.DataFrame(df['keys'].tolist(), columns=['ds'])
        df = pd.concat([df.drop('keys', axis=1), df_keys], axis=1)
        df["ds"] = pd.to_datetime(df["ds"])
        
        # Asegurarse que las métricas existan
        for metric in ['clicks', 'impressions', 'ctr', 'position']:
            if metric not in df.columns:
                df[metric] = 0
                
        return df

    except Exception as e:
        st.error(f"Error obteniendo métricas de GSC: {e}")
        return pd.DataFrame()


def hacer_prediccion(df, metrica, dias):
    """Ejecuta el modelo Prophet y devuelve el forecast y los gráficos."""
    datos = df[["ds", metrica]].rename(columns={metrica: "y"}).copy()
    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    model.fit(datos)

    future = model.make_future_dataframe(periods=dias)
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    ax1 = fig1.gca()
    ax1.set_title(f"Predicción de {metrica.capitalize()} a {dias} días", size=16)
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel(metrica.capitalize())

    fig2 = model.plot_components(forecast)

    return forecast, fig1, fig2

@st.cache_data
def convert_df_to_csv(df):
    """Función para cachear la conversión de DataFrame a CSV."""
    return df.to_csv(index=False).encode('utf-8')

# -----------------------------------------------------------------------------
# --- FLUJO PRINCIPAL DE LA APLICACIÓN
# -----------------------------------------------------------------------------

credentials = autenticar_google()

if credentials:
    # --- Comprueba si la lista de sitios ya fue cargada ---
    if 'sites' not in st.session_state:
        # Si no está cargada, la pide a Google, la ordena y la guarda
        with st.spinner("Cargando propiedades de GSC..."):
            sites_list = listar_sitios_gsc(credentials)
            st.session_state.sites = sorted(sites_list) # La ordenamos para consistencia

    # Usa la lista guardada en la sesión
    sites = st.session_state.get('sites', [])

    if not sites:
        st.warning("No se encontraron propiedades en tu cuenta de Google Search Console.")
    else:
        st.sidebar.success("¡Login exitoso!")
        # --- WIDGETS DE LA BARRA LATERAL ---
        st.sidebar.header("Configuración del Análisis")
        selected_site = st.sidebar.selectbox("Selecciona una propiedad:", options=sites)
        horizonte = st.sidebar.selectbox("Horizonte de predicción:", options=[30, 60, 90], format_func=lambda x: f'{x} días')

        # --- BOTÓN DE ANÁLISIS ---
        if st.sidebar.button("🚀 Generar Predicción", type="primary", use_container_width=True):
            with st.spinner(f"Obteniendo datos de los últimos 16 meses para **{selected_site}**..."):
                hoy = datetime.today()
                start_date = (hoy - timedelta(days=16 * 30)).strftime("%Y-%m-%d")
                end_date = hoy.strftime("%Y-%m-%d")
                df_historico = obtener_metricas(credentials, selected_site, start_date, end_date)

            if df_historico.empty:
                st.error("No se encontraron datos para el período seleccionado. Prueba con otra propiedad.")
            else:
                st.header(f"Resultados para {horizonte} días", divider='rainbow')

                # --- PREDICCIONES Y GRÁFICOS ---
                metricas_a_predecir = ["clicks", "impressions"]
                predicciones = {}

                for metrica in metricas_a_predecir:
                    st.subheader(f"Análisis de {metrica.capitalize()}")
                    with st.spinner(f"Entrenando modelo para {metrica.capitalize()}..."):
                        forecast, fig_pred, fig_comp = hacer_prediccion(df_historico, metrica, horizonte)
                        predicciones[metrica] = forecast[['ds', 'yhat']].rename(columns={'yhat': f'{metrica}_pred'})

                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig_pred)
                    with col2:
                        st.pyplot(fig_comp)
                
                # --- TABLA DE RESULTADOS ---
                st.subheader("Tabla de Predicciones Combinadas")
                df_final = pd.merge(predicciones['clicks'], predicciones['impressions'], on='ds')
                df_final['ctr_predicho'] = (df_final['clicks_pred'] / df_final['impressions_pred']).fillna(0)
                
                # Formatear para mejor visualización
                df_final_display = df_final.tail(horizonte).copy()
                df_final_display['clicks_pred'] = df_final_display['clicks_pred'].round(0).astype(int)
                df_final_display['impressions_pred'] = df_final_display['impressions_pred'].round(0).astype(int)
                df_final_display['ctr_predicho'] = (df_final_display['ctr_predicho'] * 100).round(2).astype(str) + '%'
                
                st.dataframe(df_final_display, use_container_width=True)

                csv = convert_df_to_csv(df_final.tail(horizonte))
                st.download_button(
                    label="📥 Descargar predicciones en CSV",
                    data=csv,
                    file_name=f'prediccion_{horizonte}d.csv',
                    mime='text/csv',
                )