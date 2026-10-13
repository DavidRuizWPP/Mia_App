# .\venv\Scripts\Activate.ps1
# streamlit run Dash_MiA-V03.py

import io
import os
import re
import base64
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.interpolate import interp1d
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACIÓN Y ESTILOS CORPORATIVOS WPP MEDIA
# ==============================================================================
st.set_page_config(
    page_title="WPP Media - Media Impact Analysis (MiA)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta WPP Oficial[cite: 2, 3]
COLOR_NAVY = "#000050"       # Primario: WPP Navy[cite: 2]
COLOR_LIME = "#B0F467"       # Primario: Lime Green[cite: 2]
COLOR_CYAN = "#93DFE3"       # Primario: WPP Pantone 629[cite: 2]
COLOR_CORNFLOWER = "#5465FF" # Secundario: Cornflower Blue[cite: 3]
COLOR_PERIWINKLE = "#788BFF" # Secundario: Periwinkle[cite: 3]
COLOR_TEAL = "#00DBEE"       # Secundario: Teal[cite: 3]
COLOR_YELLOW = "#FCFE67"     # Secundario: Yellow[cite: 3]

PALETA_WPP = [
    COLOR_CORNFLOWER, COLOR_TEAL, COLOR_NAVY, COLOR_PERIWINKLE,
    "#10b981", COLOR_LIME, "#f59e0b", COLOR_CYAN, COLOR_YELLOW,
    "#ec4899", "#6366f1", "#14b8a6", "#84cc16", "#e11d48"
]

st.markdown(f"""
    <style>
    .brand-header {{
        background: linear-gradient(135deg, {COLOR_NAVY} 0%, #000035 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 0.85rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 80, 0.15);
        border-bottom: 4px solid {COLOR_LIME};
    }}
    .kpi-container {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 125px;
    }}
    .kpi-title {{ font-size: 0.68rem; font-weight: 800; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 0.35rem; }}
    .kpi-value {{ font-size: 1.45rem; font-weight: 900; line-height: 1.2; }}
    .kpi-delta {{ font-size: 0.75rem; font-weight: 700; margin-top: 0.35rem; }}
    
    .section-card {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.85rem;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }}
    .section-badge {{
        display: inline-block;
        background-color: {COLOR_LIME};
        color: {COLOR_NAVY};
        font-size: 0.72rem;
        font-weight: 800;
        padding: 0.2rem 0.8rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.92rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        padding: 0.75rem 0.25rem;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLOR_NAVY} !important;
        border-bottom: 4px solid {COLOR_NAVY} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }}
    </style>
""", unsafe_allow_html=True)

def humanizar(col):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(col))
    return s.replace('_', ' ').replace('-', ' ').strip()

def color_idx(idx):
    return PALETA_WPP[idx % len(PALETA_WPP)]

def cargar_df(archivo_o_ruta):
    if archivo_o_ruta is None:
        return None
    if hasattr(archivo_o_ruta, 'read'):
        archivo_o_ruta.seek(0)
        try:
            return pd.read_csv(archivo_o_ruta, encoding='utf-8')
        except (UnicodeDecodeError, Exception):
            archivo_o_ruta.seek(0)
            return pd.read_csv(archivo_o_ruta, encoding='latin1')
    if isinstance(archivo_o_ruta, str) and os.path.exists(archivo_o_ruta):
        try:
            return pd.read_csv(archivo_o_ruta, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(archivo_o_ruta, encoding='latin1')
    return None

COLUMNAS_SISTEMA = {
    'cy', 'date', 'dateweekclose', 'fecha', 'period', 'periodo',
    'total', 'totalmedia', 'total_media', 'unnamed: 0'
}

def extraer_variables(df):
    if df is None:
        return []
    return [c for c in df.columns if c.lower() not in COLUMNAS_SISTEMA and np.issubdtype(df[c].dtype, np.number)]

def obtener_imagen_base64(ruta_imagen):
    if os.path.exists(ruta_imagen):
        with open(ruta_imagen, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

# ==============================================================================
# SIDEBAR: FUENTES DE DATOS
# ==============================================================================
st.sidebar.title("WPP Media | Fuentes")
st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Cargar archivos CSV del modelo", 
    type=["csv"], 
    accept_multiple_files=True,
    help="Arrastra aquí los archivos correspondientes al modelo."
)

files_dict = {f.name: f for f in uploaded_files} if uploaded_files else {}
locales = [f for f in os.listdir('.') if f.endswith('.csv')] if not files_dict else []
opciones_archivos = ["(No asignado)"] + list(files_dict.keys()) if files_dict else ["(No asignado)"] + locales

def auto_detectar_archivo(patron_exacto, terminos_generales, opciones):
    for op in opciones:
        if op == "(No asignado)": continue
        nl = op.lower()
        if any(p in nl for p in patron_exacto): return op
    for op in opciones:
        if op == "(No asignado)": continue
        nl = op.lower()
        if any(t in nl for t in terminos_generales): return op
    return "(No asignado)"

st.sidebar.subheader("📂 1. Fuentes Data Input")
sel_spend = st.sidebar.selectbox("Media Spend:", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["mediaspend", "media_spend"], ["spend", "inversion"], opciones_archivos)))
sel_imp = st.sidebar.selectbox("Media Impacts:", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["mediaimpression", "media_impression"], ["impression", "impresion", "impact"], opciones_archivos)))
sel_sales = st.sidebar.selectbox("Sales:", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["sales_", "sales."], ["sale", "venta"], opciones_archivos)))
sel_brand = st.sidebar.selectbox("Brand Health (Opcional):", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["brandhealth", "brand_health"], ["brand", "salud"], opciones_archivos)))
sel_ext = st.sidebar.selectbox("Factores Externos (Opcional):", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["factoresext", "factores_ext"], ["externo", "clima", "weather"], opciones_archivos)))
sel_ctrl = st.sidebar.selectbox("Variables Control (Opcional):", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["control", "varianles", "variablesdecontrol"], ["dummy", "event"], opciones_archivos)))

st.sidebar.subheader("📂 2. Fuentes Resultados & Optimizador")
sel_contrib = st.sidebar.selectbox("Model Contribution:", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["modelcontribution", "model_contribution", "contribution"], ["contrib"], opciones_archivos)))
sel_rev = st.sidebar.selectbox("Media Revenue:", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["mediarevenue", "media_revenue"], ["revenue"], opciones_archivos)))
sel_curv = st.sidebar.selectbox("Curvas Sat. / Touchpoints:", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["curvas", "curves"], ["curva", "saturation"], opciones_archivos)))

# ------------------------------------------------------------------------------
# 📂 3. PIPELINE FUNNEL DATA (TEMPLATE CSV / EXCEL)[cite: 8]
# ------------------------------------------------------------------------------
st.sidebar.subheader("📂 3. Pipeline Funnel (Template)")
archivo_template_pipeline = st.sidebar.file_uploader(
    "Cargar Template Pipeline (.xlsx o .csv):",
    type=["xlsx", "csv"],
    help="Sube aquí el archivo Template_Pipeline_MiA_2.xlsx o similar."
)

ruta_pipeline_defecto = "Template_Pipeline_MiA_2.xlsx" if os.path.exists("Template_Pipeline_MiA_2.xlsx") else (
    "Template_Pipeline_MiA.xlsx" if os.path.exists("Template_Pipeline_MiA.xlsx") else (
        "Template_Pipeline_MiA.csv" if os.path.exists("Template_Pipeline_MiA.csv") else None
    )
)

df_pipeline_raw = None
pipeline_hash = "default"

if archivo_template_pipeline is not None:
    archivo_template_pipeline.seek(0)
    bytes_p = archivo_template_pipeline.getvalue()
    pipeline_hash = hashlib.md5(bytes_p).hexdigest()[:8]
    if archivo_template_pipeline.name.endswith(".csv"):
        df_pipeline_raw = pd.read_csv(archivo_template_pipeline)
    else:
        df_pipeline_raw = pd.read_excel(archivo_template_pipeline)
    st.sidebar.caption(f"✅ Template cargado: `{archivo_template_pipeline.name}`")
elif ruta_pipeline_defecto is not None:
    with open(ruta_pipeline_defecto, "rb") as f:
        bytes_p = f.read()
    pipeline_hash = hashlib.md5(bytes_p).hexdigest()[:8]
    if ruta_pipeline_defecto.endswith(".csv"):
        df_pipeline_raw = pd.read_csv(ruta_pipeline_defecto)
    else:
        df_pipeline_raw = pd.read_excel(ruta_pipeline_defecto)
    st.sidebar.caption(f"✅ Template local: `{ruta_pipeline_defecto}`")

# Botón para descargar el Template
template_ejemplo = pd.DataFrame({
    'DataType': ['Actual']*3 + ['Predective']*3,
    'DateWeekClose': ['05/01/2025', '12/01/2025', '19/01/2025', '05/10/2025', '12/10/2025', '19/10/2025'],
    'Circulo_PaidMedia': [1850000, 1360000, 1250000, np.nan, np.nan, np.nan],
    'Circulo_MentalAvailability': [12400, 126000, 123000, np.nan, np.nan, np.nan],
    'Circulo_ConsumoFrecuente': [24700, 252000, 245000, np.nan, np.nan, np.nan],
    'Circulo_ActivacionDigital': [40900, 418000, 407000, np.nan, np.nan, np.nan],
    'Circulo_AWOS': [118000, 1200000, 1170000, np.nan, np.nan, np.nan],
    'Flecha_PM_a_AWOs': [np.nan, np.nan, np.nan, 260.1, 216.4, 151.9],
    'Flecha_PM_a_MA': [np.nan, np.nan, np.nan, 24.6, 20.9, 14.8],
    'Flecha_PM_a_CF': [np.nan, np.nan, np.nan, 39.8, 35.7, 25.6],
    'Flecha_PM_a_AD': [np.nan, np.nan, np.nan, 153.7, 116.5, 79.2],
    'Flecha_MA_a_CF': [np.nan, np.nan, np.nan, 39.8, 35.7, 25.6],
    'Flecha_MA_a_AD': [np.nan, np.nan, np.nan, 153.7, 116.5, 79.2]
})

buf_template = io.BytesIO()
with pd.ExcelWriter(buf_template, engine='openpyxl') as writer:
    template_ejemplo.to_excel(writer, index=False, sheet_name='Template')

st.sidebar.download_button(
    label="📥 Descargar Plantilla Template (.xlsx)",
    data=buf_template.getvalue(),
    file_name="Template_Pipeline_MiA.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

def resolver_fuente(sel):
    if sel == "(No asignado)": return None
    return files_dict.get(sel, sel)

spend_df = cargar_df(resolver_fuente(sel_spend))
imp_df = cargar_df(resolver_fuente(sel_imp))
sales_df = cargar_df(resolver_fuente(sel_sales))
brand_df = cargar_df(resolver_fuente(sel_brand))
ext_df = cargar_df(resolver_fuente(sel_ext))
ctrl_df = cargar_df(resolver_fuente(sel_ctrl))
contrib_df = cargar_df(resolver_fuente(sel_contrib))
rev_df = cargar_df(resolver_fuente(sel_rev))
curv_df = cargar_df(resolver_fuente(sel_curv))

# Normalizar columnas de fecha
for df in [spend_df, imp_df, sales_df, brand_df, ext_df, ctrl_df, contrib_df, rev_df]:
    if df is not None:
        for c in ['DateWeekClose', 'date', 'fecha', 'Fecha', 'Period', 'date_week_close']:
            if c in df.columns:
                df.rename(columns={c: 'Date'}, inplace=True)
                break

if sales_df is None or spend_df is None:
    st.warning("⚠️ Carga al menos los archivos de **Sales** y **Media Spend** para iniciar el análisis.")
    st.stop()

canales_detectados = extraer_variables(spend_df)
canales_meta = {ch: {'name': humanizar(ch), 'color': color_idx(i)} for i, ch in enumerate(canales_detectados)}

vars_sales = extraer_variables(sales_df)
vars_brand = extraer_variables(brand_df)
vars_ext = extraer_variables(ext_df)
vars_ctrl = extraer_variables(ctrl_df)

st.sidebar.subheader("🎯 Filtros Globales")
cys_disponibles = sorted(list(sales_df['CY'].unique())) if 'CY' in sales_df.columns else []
selected_cys = st.sidebar.multiselect("Período (CY):", options=cys_disponibles, default=cys_disponibles)

nombres_a_ids = {meta['name']: ch for ch, meta in canales_meta.items()}
selected_nombres = st.sidebar.multiselect(
    f"Canales de Medios ({len(canales_detectados)} activos):",
    options=list(nombres_a_ids.keys()),
    default=list(nombres_a_ids.keys())
)
selected_media_ids = [nombres_a_ids[nom] for nom in selected_nombres]

active_cys = selected_cys if selected_cys else cys_disponibles

def filtrar_cy(df):
    if df is not None and 'CY' in df.columns and active_cys:
        return df[df['CY'].isin(active_cys)].copy()
    return df.copy() if df is not None else None

spend_f = filtrar_cy(spend_df)
imp_f = filtrar_cy(imp_df)
sales_f = filtrar_cy(sales_df)
brand_f = filtrar_cy(brand_df)
ext_f = filtrar_cy(ext_df)
ctrl_f = filtrar_cy(ctrl_df)
contrib_f = filtrar_cy(contrib_df)
rev_f = filtrar_cy(rev_df)

# ==============================================================================
# HEADER PRINCIPAL
# ==============================================================================
nombre_logo = "Logo_WPP+ALSEA.png"
logo_base64 = obtener_imagen_base64(nombre_logo)
if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="WPP + ALSEA" style="max-height: 48px; max-width: 220px; object-fit: contain;">'
else:
    logo_html = f'<div style="font-weight: 800; font-size: 1.15rem; color: #ffffff; letter-spacing: 0.05em;">WPP <span style="color: {COLOR_LIME};">+</span> ALSEA</div>'

st.markdown(f"""
    <div class="brand-header" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #ffffff !important; letter-spacing: -0.025em;">
                WPP Media | Media Impact Analysis (MiA)
            </h1>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; color: {COLOR_CYAN}; font-weight: 500;">
                Reporte Ejecutivo Dinámico | MMM Optimization & Data Intelligence
            </p>
        </div>
        <div style="display: flex; align-items: center; justify-content: flex-end;">
            {logo_html}
        </div>
    </div>
""", unsafe_allow_html=True)

prev_cys = []
if len(selected_cys) == 1 and len(cys_disponibles) > 1:
    idx_actual = cys_disponibles.index(selected_cys[0])
    if idx_actual > 0:
        prev_cys = [cys_disponibles[idx_actual - 1]]

def get_delta_badge(curr, prev):
    if prev is None or prev == 0:
        return '<span style="color:#94a3b8;font-weight:400;">Sin ant.</span>'
    pct = ((curr - prev) / prev) * 100
    color = "#16a34a" if pct > 0 else "#dc2626"
    return f'<span style="color:{color};">{"+" if pct > 0 else ""}{pct:.1f}% vs ant.</span>'

# ==============================================================================
# MOTOR MATEMÁTICO DE OPTIMIZACIÓN (SLSQP CON S-CURVES)
# ==============================================================================
def preparar_y_obtener_limites_engine(df_curvas, budget_anual, meses_con_actividad=12):
    df = df_curvas.copy()
    cols_norm = {}
    for c in df.columns:
        cl = c.strip().replace(' ', '')
        if 'inversi' in c.lower(): cols_norm[c] = 'Inversión'
        elif 'heavy' in c.lower(): cols_norm[c] = 'HeavyUp'
        elif 'sustain' in c.lower(): cols_norm[c] = 'Sustain'
        elif 'mroi' in c.lower(): cols_norm[c] = 'mROI'
        elif 'contrib' in c.lower(): cols_norm[c] = 'Contribución'
        else: cols_norm[c] = cl
    df.rename(columns=cols_norm, inplace=True)
    
    presupuesto_mensual = budget_anual / meses_con_actividad
    canales = df['Medio'].unique()
    
    inversion_historica_total = sum(df[df['Sustain'] > 0]['Inversión'].values)
    if inversion_historica_total <= 0:
        inversion_historica_total = df['Inversión'].mean() * len(canales)
        
    factor_escala = presupuesto_mensual / (inversion_historica_total / len(canales)) if inversion_historica_total > 0 else 1.0
    factor_escala = max(0.2, min(factor_escala, 5.0))

    limites_sugeridos = {}
    for canal in canales:
        sub_canal = df[df['Medio'] == canal]
        sub_sustain = sub_canal[sub_canal['Sustain'] > 0]
        min_base = sub_sustain['Inversión'].values[0] if len(sub_sustain) > 0 else sub_canal['Inversión'].min()
        sub_heavy = sub_canal[sub_canal['HeavyUp'] > 0]
        max_base = sub_heavy['Inversión'].values[0] if len(sub_heavy) > 0 else sub_canal['Inversión'].max()
        
        min_dinero = min_base * (factor_escala ** 0.5)
        max_dinero = max_base * factor_escala
        limites_sugeridos[canal] = {'min': float(min_dinero), 'max': float(max_dinero)}
        
    return limites_sugeridos

def ejecutar_optimizacion_engine(df_curvas, budget_anual, meses_con_actividad=12, limites_personalizados=None):
    df = df_curvas.copy()
    cols_norm = {}
    for c in df.columns:
        cl = c.strip().replace(' ', '')
        if 'inversi' in c.lower(): cols_norm[c] = 'Inversión'
        elif 'heavy' in c.lower(): cols_norm[c] = 'HeavyUp'
        elif 'sustain' in c.lower(): cols_norm[c] = 'Sustain'
        elif 'mroi' in c.lower(): cols_norm[c] = 'mROI'
        elif 'contrib' in c.lower(): cols_norm[c] = 'Contribución'
        else: cols_norm[c] = cl
    df.rename(columns=cols_norm, inplace=True)
    
    presupuesto_mensual = budget_anual / meses_con_actividad
    canales = df['Medio'].unique()
    
    inversion_historica_total = sum(df[df['Sustain'] > 0]['Inversión'].values)
    if inversion_historica_total <= 0:
        inversion_historica_total = df['Inversión'].mean() * len(canales)
        
    factor_escala = presupuesto_mensual / (inversion_historica_total / len(canales)) if inversion_historica_total > 0 else 1.0
    factor_escala = max(0.2, min(factor_escala, 5.0))
    
    if limites_personalizados is None:
        limites_calc = {}
        for canal in canales:
            sub_canal = df[df['Medio'] == canal]
            sub_sustain = sub_canal[sub_canal['Sustain'] > 0]
            min_base = sub_sustain['Inversión'].values[0] if len(sub_sustain) > 0 else sub_canal['Inversión'].min()
            sub_heavy = sub_canal[sub_canal['HeavyUp'] > 0]
            max_base = sub_heavy['Inversión'].values[0] if len(sub_heavy) > 0 else sub_canal['Inversión'].max()
            min_dinero = min_base * (factor_escala ** 0.5)
            max_dinero = max_base * factor_escala
            limites_calc[canal] = {'min': min_dinero, 'max': max_dinero}
    else:
        limites_calc = limites_personalizados
        
    interpoladores_roi = {}
    limites_historicos = {}
    for canal in canales:
        sub = df[df['Medio'] == canal].sort_values('Inversión')
        x = sub['Inversión'].values
        y_roi = sub['mROI'].values
        interpoladores_roi[canal] = interp1d(
            x, y_roi, kind='cubic', bounds_error=False, fill_value=(y_roi[0], y_roi[-1])
        )
        limites_historicos[canal] = (x.min(), x.max() * max(1.0, factor_escala))
        
    bounds = []
    suma_minimos = 0
    for canal in canales:
        h_min, h_max = limites_historicos[canal]
        if canal in limites_calc:
            min_d = limites_calc[canal].get('min', h_min)
            max_d = limites_calc[canal].get('max', h_max)
            min_b = max(h_min, min_d)
            max_b = max(min_b * 1.05, max_d)
            if min_b > max_b: min_b = h_min
        else:
            min_b, max_b = h_min, h_max
        bounds.append((min_b, max_b))
        suma_minimos += min_b
        
    if suma_minimos > presupuesto_mensual:
        factor_ajuste = presupuesto_mensual / suma_minimos
        bounds = [(b[0] * factor_ajuste, b[1]) for b in bounds]
        
    def funcion_objetivo(inversiones):
        ganancia = 0
        for i, canal in enumerate(canales):
            inv = inversiones[i]
            roi = interpoladores_roi[canal](inv)
            ganancia += inv * roi
        return -ganancia
        
    restricciones = ({
        'type': 'eq',
        'fun': lambda inv: np.sum(inv) - presupuesto_mensual
    })
    
    x0 = [presupuesto_mensual / len(canales)] * len(canales)
    for i in range(len(canales)):
        x0[i] = max(bounds[i][0], min(x0[i], bounds[i][1]))
        
    res = minimize(
        funcion_objetivo, x0, method='SLSQP', bounds=bounds,
        constraints=restricciones, options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    inversiones_mensuales = res.x
    suma_m = np.sum(inversiones_mensuales)
    if abs(suma_m - presupuesto_mensual) > 1e-4:
        inversiones_mensuales = inversiones_mensuales * (presupuesto_mensual / suma_m)
        
    resultados_lista = []
    suma_rev_anual = 0
    for i, canal in enumerate(canales):
        inv_m = inversiones_mensuales[i]
        inv_a = inv_m * meses_con_actividad
        roi = float(interpoladores_roi[canal](inv_m))
        rev_m = inv_m * roi
        rev_a = rev_m * meses_con_actividad
        suma_rev_anual += rev_a
        resultados_lista.append({
            'Medio': canal,
            'Inversión Óptima Anual': inv_a,
            'ROI Estimado': roi,
            'Media Revenue Anual': rev_a
        })
    df_res = pd.DataFrame(resultados_lista)
    df_res['Digital Mix'] = df_res['Inversión Óptima Anual'] / df_res['Inversión Óptima Anual'].sum()
    return df_res, suma_rev_anual, (suma_rev_anual / budget_anual) if budget_anual > 0 else 0.0

# ==============================================================================
# TABS DE NAVEGACIÓN (4 PESTAÑAS INTEGRANDO EL PIPELINE)
# ==============================================================================
tab_results, tab_input, tab_opt, tab_pipeline = st.tabs([
    "🏆 Resultados MiA (Contribución y ROI)", 
    "📁 Información Input (Data de Entrada)",
    "⚡ Media Optimization (Optimizador S-Curve)",
    "🍕 Pipeline Funnel (Multi-MMM Simulator)"
])

# ##############################################################################
# TAB 1: RESULTADOS MIA (CONTRIBUCIÓN Y ROI)[cite: 6]
# ##############################################################################
with tab_results:
    if contrib_df is None or rev_df is None:
        st.warning("⚠️ Para visualizar los **Resultados MiA**, asigna los archivos de **Model Contribution** y **Media Revenue** en la barra lateral.")
    else:
        var_vol = vars_sales[0] if vars_sales else 'Litros'
        total_sales_vol = sales_f[var_vol].sum() if var_vol in sales_f.columns else 0.0

        medios_contrib = [m for m in selected_media_ids if m in contrib_f.columns]
        medios_rev = [m for m in selected_media_ids if m in rev_f.columns]
        medios_spend = [m for m in selected_media_ids if m in spend_f.columns]

        curr_contrib_vol = contrib_f[medios_contrib].sum().sum() if medios_contrib else 0.0
        curr_spend_res = spend_f[medios_spend].sum().sum() if medios_spend else 0.0
        curr_rev_res = rev_f[medios_rev].sum().sum() if medios_rev else 0.0

        pct_media_contrib = (curr_contrib_vol / total_sales_vol) if total_sales_vol > 0 else 0.0
        total_roi_media = (curr_rev_res / curr_spend_res) if curr_spend_res > 0 else 0.0

        cols_no_brand = set(canales_detectados + [var_vol, 'Baseline', 'CY', 'Date', 'Litros Totales', 'TotalMedia'])
        cols_brand_contrib = [c for c in contrib_f.columns if c not in cols_no_brand and np.issubdtype(contrib_f[c].dtype, np.number) and 'bardahl' not in c.lower()]
        curr_brand_contrib_vol = contrib_f[cols_brand_contrib].sum().sum() if cols_brand_contrib else 0.0
        pct_brand_contrib = (curr_brand_contrib_vol / total_sales_vol) if total_sales_vol > 0 else 0.0

        p_pct_media, p_roi, p_pct_brand = None, None, None
        if prev_cys:
            p_sales_rows = sales_df[sales_df['CY'].isin(prev_cys)]
            p_contrib_rows = contrib_df[contrib_df['CY'].isin(prev_cys)]
            p_spend_rows = spend_df[spend_df['CY'].isin(prev_cys)]
            p_rev_rows = rev_df[rev_df['CY'].isin(prev_cys)]

            p_total_vol = p_sales_rows[var_vol].sum() if var_vol in p_sales_rows.columns else 0.0
            p_contrib_vol = p_contrib_rows[medios_contrib].sum().sum() if medios_contrib else 0.0
            p_spend_val = p_spend_rows[medios_spend].sum().sum() if medios_spend else 0.0
            p_rev_val = p_rev_rows[medios_rev].sum().sum() if medios_rev else 0.0
            p_brand_vol = p_contrib_rows[cols_brand_contrib].sum().sum() if cols_brand_contrib else 0.0

            p_pct_media = (p_contrib_vol / p_total_vol) if p_total_vol > 0 else 0.0
            p_roi = (p_rev_val / p_spend_val) if p_spend_val > 0 else 0.0
            p_pct_brand = (p_brand_vol / p_total_vol) if p_total_vol > 0 else 0.0

        res_kpi_1, res_kpi_2, res_kpi_3 = st.columns(3)
        with res_kpi_1:
            st.markdown(f"""
                <div class="kpi-container" style="min-height: 120px;">
                    <div>
                        <div class="kpi-title">Contribución Total Media (%)</div>
                        <div class="kpi-value" style="color:{COLOR_NAVY};">{pct_media_contrib*100:.1f}%</div>
                    </div>
                    <div class="kpi-delta">{get_delta_badge(pct_media_contrib, p_pct_media)}</div>
                </div>
            """, unsafe_allow_html=True)
        with res_kpi_2:
            st.markdown(f"""
                <div class="kpi-container" style="min-height: 120px;">
                    <div>
                        <div class="kpi-title">ROI Total Media (Revenue / Spend)</div>
                        <div class="kpi-value" style="color:{COLOR_CORNFLOWER};">{total_roi_media:.2f}x</div>
                    </div>
                    <div class="kpi-delta">{get_delta_badge(total_roi_media, p_roi)}</div>
                </div>
            """, unsafe_allow_html=True)
        with res_kpi_3:
            st.markdown(f"""
                <div class="kpi-container" style="min-height: 120px;">
                    <div>
                        <div class="kpi-title">Contribución Brand Health (%)</div>
                        <div class="kpi-value" style="color:{COLOR_TEAL};">{pct_brand_contrib*100:.1f}%</div>
                    </div>
                    <div class="kpi-delta">{get_delta_badge(pct_brand_contrib, p_pct_brand)}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Model Construction
        date_col_c = 'Date' if 'Date' in contrib_f.columns else contrib_f.columns[1]
        fig_decomp = go.Figure()

        if 'Baseline' in contrib_f.columns:
            fig_decomp.add_trace(go.Scatter(
                x=contrib_f[date_col_c], y=contrib_f['Baseline'],
                name='Baseline', mode='lines', stackgroup='one',
                line=dict(width=0.5, color='#94a3b8'), fillcolor='rgba(148, 163, 184, 0.45)'
            ))

        if cols_brand_contrib:
            sum_brand = contrib_f[cols_brand_contrib].sum(axis=1)
            fig_decomp.add_trace(go.Scatter(
                x=contrib_f[date_col_c], y=sum_brand,
                name='Brand Health', mode='lines', stackgroup='one',
                line=dict(width=0.5, color=COLOR_TEAL), fillcolor='rgba(0, 219, 238, 0.45)'
            ))

        for idx, m_id in enumerate(medios_contrib):
            fig_decomp.add_trace(go.Scatter(
                x=contrib_f[date_col_c], y=contrib_f[m_id],
                name=canales_meta[m_id]['name'], mode='lines', stackgroup='one',
                line=dict(width=0.5, color=canales_meta[m_id]['color']),
                fillcolor=canales_meta[m_id]['color']
            ))

        if var_vol in sales_f.columns:
            fig_decomp.add_trace(go.Scatter(
                x=sales_f[date_col_c], y=sales_f[var_vol],
                name=f'Ventas Reales ({var_vol})', mode='lines+markers',
                line=dict(color=COLOR_NAVY, width=2.5), marker=dict(size=3, color=COLOR_NAVY)
            ))

        fig_decomp.update_layout(
            height=380, margin=dict(l=20, r=20, t=20, b=50),
            yaxis=dict(title=f"Volumen ({var_vol})", gridcolor="#f1f5f9"),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5, font=dict(size=9)),
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_decomp, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Waterfall
        if len(cys_disponibles) >= 2:
            wf_start_cy = cys_disponibles[-2]
            wf_end_cy = cys_disponibles[-1]

            s_sales = sales_df[sales_df['CY'] == wf_start_cy][var_vol].sum()
            e_sales = sales_df[sales_df['CY'] == wf_end_cy][var_vol].sum()
            net_growth = ((e_sales - s_sales) / s_sales) * 100 if s_sales > 0 else 0

            s_contrib = contrib_df[contrib_df['CY'] == wf_start_cy]
            e_contrib = contrib_df[contrib_df['CY'] == wf_end_cy]

            diff_baseline = (e_contrib['Baseline'].sum() - s_contrib['Baseline'].sum()) if 'Baseline' in contrib_df.columns else 0.0
            diff_media = e_contrib[medios_contrib].sum().sum() - s_contrib[medios_contrib].sum().sum()
            
            wf_drivers_x = [wf_start_cy, "Baseline", "Media Execution"]
            wf_drivers_y = [s_sales, diff_baseline, diff_media]
            wf_measures = ["absolute", "relative", "relative"]

            for c in cols_brand_contrib:
                d_val = e_contrib[c].sum() - s_contrib[c].sum()
                wf_drivers_x.append(humanizar(c))
                wf_drivers_y.append(d_val)
                wf_measures.append("relative")

            cols_comp = [c for c in contrib_df.columns if 'bardahl' in c.lower() or 'comp' in c.lower()]
            for c in cols_comp:
                d_val = e_contrib[c].sum() - s_contrib[c].sum()
                wf_drivers_x.append(humanizar(c))
                wf_drivers_y.append(d_val)
                wf_measures.append("relative")

            wf_drivers_x.append(wf_end_cy)
            wf_drivers_y.append(e_sales)
            wf_measures.append("total")

            fig_wf = go.Figure(go.Waterfall(
                orientation="v", measure=wf_measures, x=wf_drivers_x, y=wf_drivers_y,
                connector=dict(line=dict(color="#cbd5e1")),
                increasing=dict(marker=dict(color=COLOR_LIME)),
                decreasing=dict(marker=dict(color="#ef4444")),
                totals=dict(marker=dict(color=COLOR_NAVY))
            ))
            fig_wf.update_layout(
                title=f"Net YoY Growth ({wf_start_cy} vs {wf_end_cy}): {net_growth:+.1f}%",
                height=360, margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(gridcolor="#f1f5f9"), plot_bgcolor='white'
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3 y 4. Contribución y ROI
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            fig_pct = go.Figure()
            for idx, cy in enumerate(active_cys):
                cy_s = sales_df[sales_df['CY'] == cy][var_vol].sum()
                cy_c = contrib_df[contrib_df['CY'] == cy]
                pct_vals = [(cy_c[m].sum() / cy_s * 100) if (cy_s > 0 and m in cy_c.columns) else 0.0 for m in selected_media_ids]
                fig_pct.add_trace(go.Bar(
                    x=[canales_meta[m]['name'] for m in selected_media_ids],
                    y=pct_vals, name=cy, marker_color=color_idx(idx)
                ))
            fig_pct.update_layout(
                title="Contribución de Medios (%) por Canal",
                barmode='group', height=330, margin=dict(l=20, r=20, t=35, b=40),
                yaxis=dict(title="% Contribución", gridcolor="#f1f5f9", ticksuffix="%"),
                xaxis=dict(tickangle=-25), plot_bgcolor='white'
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        with c_m2:
            fig_roi = go.Figure()
            for idx, cy in enumerate(active_cys):
                cy_sp = spend_df[spend_df['CY'] == cy]
                cy_rv = rev_df[rev_df['CY'] == cy]
                roi_vals = [(cy_rv[m].sum() / cy_sp[m].sum()) if (m in cy_sp.columns and cy_sp[m].sum() > 0 and m in cy_rv.columns) else 0.0 for m in selected_media_ids]
                fig_roi.add_trace(go.Bar(
                    x=[canales_meta[m]['name'] for m in selected_media_ids],
                    y=roi_vals, name=cy, marker_color=color_idx(idx + 2)
                ))
            fig_roi.update_layout(
                title="Retorno de Inversión (ROI) por Canal",
                barmode='group', height=330, margin=dict(l=20, r=20, t=35, b=40),
                yaxis=dict(title="ROI (Revenue / Spend)", gridcolor="#f1f5f9", ticksuffix="x"),
                xaxis=dict(tickangle=-25), plot_bgcolor='white'
            )
            st.plotly_chart(fig_roi, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 5. Dos Anillos
        c_don1, c_don2 = st.columns(2)
        pie_labels = [canales_meta[m]['name'] for m in selected_media_ids]
        pie_colors = [canales_meta[m]['color'] for m in selected_media_ids]
        contrib_sums = [contrib_f[m].sum() if m in contrib_f.columns else 0.0 for m in selected_media_ids]
        spend_sums = [spend_f[m].sum() if m in spend_f.columns else 0.0 for m in selected_media_ids]

        with c_don1:
            fig_d_contrib = go.Figure(data=[go.Pie(
                labels=pie_labels, values=contrib_sums, hole=0.55,
                marker=dict(colors=pie_colors), textinfo='percent'
            )])
            fig_d_contrib.update_layout(title="Contribution Mix (% Share - Solo Medios)", height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_d_contrib, use_container_width=True)

        with c_don2:
            fig_d_spend = go.Figure(data=[go.Pie(
                labels=pie_labels, values=spend_sums, hole=0.55,
                marker=dict(colors=pie_colors), textinfo='percent'
            )])
            fig_d_spend.update_layout(title="Media Spend Mix (% Share Inversión)", height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_d_spend, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 6. Circana Dynamics
        labels_dyn = ["Total Media"] + pie_labels
        for met_name, suf, df_source, is_roi in [
            ("Media Spend ($ MXN)", "$", spend_df, False),
            ("Media Incremental Net Revenue ($ MXN)", "$", rev_df, False),
            ("Media Net Revenue ROI ($ Revenue / $ Spend)", "x", None, True)
        ]:
            fig_dyn = go.Figure()
            for idx, cy in enumerate(active_cys):
                vals_dyn = []
                if not is_roi:
                    t_val = df_source[df_source['CY'] == cy][selected_media_ids].sum().sum()
                else:
                    sp_t = spend_df[spend_df['CY'] == cy][selected_media_ids].sum().sum()
                    rv_t = rev_df[rev_df['CY'] == cy][selected_media_ids].sum().sum()
                    t_val = (rv_t / sp_t) if sp_t > 0 else 0.0
                vals_dyn.append(t_val)

                for m in selected_media_ids:
                    if not is_roi:
                        v = df_source[df_source['CY'] == cy][m].sum() if m in df_source.columns else 0.0
                    else:
                        sp_m = spend_df[spend_df['CY'] == cy][m].sum() if m in spend_df.columns else 0.0
                        rv_m = rev_df[rev_df['CY'] == cy][m].sum() if m in rev_df.columns else 0.0
                        v = (rv_m / sp_m) if sp_m > 0 else 0.0
                    vals_dyn.append(v)

                fig_dyn.add_trace(go.Bar(x=labels_dyn, y=vals_dyn, name=cy, marker_color=color_idx(idx + 1)))

            fig_dyn.update_layout(
                title=met_name, barmode='group', height=280, margin=dict(l=20, r=20, t=35, b=20),
                yaxis=dict(gridcolor="#f1f5f9", ticksuffix=suf), xaxis=dict(tickangle=-20), plot_bgcolor='white'
            )
            st.plotly_chart(fig_dyn, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 7. Curvas S & Benchmarks[cite: 7]
        if curv_df is not None:
            st.markdown("""
                <div class="section-card">
                    <span class="section-badge">📉 7. Optimización de Presupuesto & Curvas de Respuesta</span>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Saturation & Response S-Curve por Medio</h3>
                    <p style="margin: 0.15rem 0 0.5rem 0; font-size: 0.8rem; color: #64748b;">Curva de respuesta continua y seguimiento de la ejecución histórica vs. umbrales guía.</p>
                </div>
            """, unsafe_allow_html=True)

            medios_curvas = sorted(list(curv_df['Medio'].unique()))
            sel_curv_media = st.selectbox("Seleccionar Medio para evaluar Curva y Ejecución:", medios_curvas, key="sel_curva_dinamica")
            c_rows = curv_df[curv_df['Medio'] == sel_curv_media].copy()

            col_inv = next((c for c in c_rows.columns if 'inversi' in c.lower() or 'spend' in c.lower()), c_rows.columns[1])
            col_cont = next((c for c in c_rows.columns if 'contrib' in c.lower()), c_rows.columns[2])

            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=c_rows[col_inv], 
                y=c_rows[col_cont] * 100 if c_rows[col_cont].mean() <= 1.0 else c_rows[col_cont],
                mode='lines', name='Curva de Respuesta', line=dict(color=COLOR_NAVY, width=3)
            ))

            umbrales_spend = {}
            for punto, col_pt, sym in [("Sustain", "#f59e0b", "diamond"), ("Optimal", "#10b981", "circle"), ("Heavy Up", "#ef4444", "square")]:
                col_found = next((c for c in c_rows.columns if punto.lower().replace(" ", "") in c.lower().replace(" ", "")), None)
                if col_found:
                    pt_val = c_rows[col_found].max()
                    if pt_val > 0:
                        pt_row = c_rows[c_rows[col_found] == pt_val].iloc[0]
                        inv_val = float(pt_row[col_inv])
                        umbrales_spend[punto] = inv_val
                        fig_curve.add_trace(go.Scatter(
                            x=[inv_val], 
                            y=[pt_row[col_cont] * 100 if pt_row[col_cont] <= 1.0 else pt_row[col_cont]],
                            mode='markers', name=f"Punto {punto}", marker=dict(size=11, color=col_pt, symbol=sym)
                        ))

            fig_curve.update_layout(
                title=f"Curva S: Contribución vs. Inversión ({sel_curv_media})",
                height=360, margin=dict(l=20, r=20, t=35, b=20),
                xaxis=dict(title="Nivel de Inversión ($ MXN)", gridcolor="#f1f5f9"),
                yaxis=dict(title="% Contribución", gridcolor="#f1f5f9", ticksuffix="%"),
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            # Inversión Histórica vs Benchmarks[cite: 7]
            st.markdown("""
                <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1.25rem; margin-top: 1rem;">
                    <h4 style="margin: 0; font-size: 1rem; font-weight: 800; color: #000050; text-transform: uppercase;">
                        INVERSIÓN HISTÓRICA MENSUAL VS. NIVELES GUÍA RECOMENDADOS ($)
                    </h4>
                    <p style="margin: 0.2rem 0 0.8rem 0; font-size: 0.78rem; color: #64748b;">
                        Seguimiento de la ejecución presupuestaria frente a los umbrales de Sustain (Naranja), Optimal (Verde) y Heavy Up (Rojo).
                    </p>
                </div>
            """, unsafe_allow_html=True)

            col_spend_match = next((c for c in spend_f.columns if c.lower() == sel_curv_media.lower() or sel_curv_media.lower() in c.lower()), None)
            if col_spend_match and col_spend_match in spend_f.columns:
                date_labels = spend_f['CY'].astype(str) + " - " + spend_f[date_col_c].astype(str)
                hist_spend_vals = spend_f[col_spend_match]

                fig_hist_bench = go.Figure()
                fig_hist_bench.add_trace(go.Scatter(
                    x=date_labels, y=hist_spend_vals,
                    name="Inversión Histórica Real", mode="lines+markers",
                    line=dict(color="#2563eb", width=2.5), marker=dict(size=6, color="#2563eb"),
                    hovertemplate="%{x}<br>Gasto Real: $%{y:,.0f}<extra></extra>"
                ))

                if "Sustain" in umbrales_spend:
                    fig_hist_bench.add_hline(y=umbrales_spend["Sustain"], line_dash="dash", line_color="#f59e0b", line_width=1.8, annotation_text="Nivel Sustain (Mínimo)", annotation_position="bottom right")
                if "Optimal" in umbrales_spend:
                    fig_hist_bench.add_hline(y=umbrales_spend["Optimal"], line_dash="solid", line_color="#10b981", line_width=2, annotation_text="Nivel Optimal (Óptimo)", annotation_position="top right")
                if "Heavy Up" in umbrales_spend:
                    fig_hist_bench.add_hline(y=umbrales_spend["Heavy Up"], line_dash="dot", line_color="#ef4444", line_width=1.8, annotation_text="Nivel Heavy Up (Máximo)", annotation_position="top right")

                fig_hist_bench.update_layout(
                    height=380, margin=dict(l=20, r=20, t=25, b=65),
                    yaxis=dict(title="Inversión ($)", gridcolor="#f1f5f9", tickprefix="$", tickformat=",.0f"),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=8, color="#64748b")),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.55, xanchor="center", x=0.5, font=dict(size=9)),
                    plot_bgcolor="white"
                )
                st.plotly_chart(fig_hist_bench, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 8. Tabla Detallada
        st.markdown("""
            <div class="section-card">
                <span class="section-badge">📋 8. Tabla Detallada</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Detalle de Contribución y Desempeño por Driver</h3>
            </div>
        """, unsafe_allow_html=True)

        tbl_data = []
        for m in selected_media_ids:
            sp = spend_f[m].sum() if m in spend_f.columns else 0.0
            rv = rev_f[m].sum() if m in rev_f.columns else 0.0
            cb = contrib_f[m].sum() if m in contrib_f.columns else 0.0
            pct = (cb / total_sales_vol * 100) if total_sales_vol > 0 else 0.0
            roi = (rv / sp) if sp > 0 else 0.0
            tbl_data.append({
                "Driver / Canal": canales_meta[m]['name'],
                "Inversión (Spend)": f"${sp:,.0f}",
                "Ingresos (Revenue)": f"${rv:,.0f}",
                f"Contribución ({var_vol})": f"{cb:,.0f}",
                "% Contribución": f"{pct:.2f}%",
                "ROI": f"{roi:.2f}x"
            })
        st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

# ##############################################################################
# TAB 2: INFORMACIÓN INPUT (DATA DE ENTRADA)
# ##############################################################################
with tab_input:
    def sincronizar_var_sesion(key_prefix, lista_vars):
        key = f"kpi_sel_{key_prefix}"
        if not lista_vars: return None
        if key in st.session_state and st.session_state[key] not in lista_vars:
            st.session_state[key] = lista_vars[0]
        return st.session_state.get(key, lista_vars[0])

    grupos_activos = [("spend", "Inversión Medios")]
    if vars_sales: grupos_activos.append(("sales", "Ventas"))
    if vars_brand: grupos_activos.append(("brand", "Brand Health"))
    if vars_ext:   grupos_activos.append(("ext", "Factores Externos"))
    if vars_ctrl:  grupos_activos.append(("ctrl", "Variables Control"))

    kpi_cols = st.columns(len(grupos_activos))

    def render_kpi_card(col_ui, key_prefix, titulo_grupo, opciones, valor_formateado, delta_html, color_texto):
        with col_ui:
            st.markdown(f'<div class="kpi-title">{titulo_grupo}</div>', unsafe_allow_html=True)
            st.selectbox(f"Sel. {titulo_grupo}", options=opciones, key=f"kpi_sel_{key_prefix}", label_visibility="collapsed")
            st.markdown(f"""
                <div class="kpi-container" style="border-top:none; margin-top: -15px;">
                    <div class="kpi-value" style="color:{color_texto};">{valor_formateado}</div>
                    <div class="kpi-delta">{delta_html}</div>
                </div>
            """, unsafe_allow_html=True)

    total_spend_inp = spend_f[selected_media_ids].sum().sum() if selected_media_ids else 0.0
    prev_spend_inp = None
    if prev_cys:
        p_df = spend_df[spend_df['CY'].isin(prev_cys)]
        prev_spend_inp = p_df[selected_media_ids].sum().sum() if selected_media_ids else 0.0

    render_kpi_card(kpi_cols[0], "spend", "Inversión Medios", ["Inversión Total"], f"${total_spend_inp:,.0f}", get_delta_badge(total_spend_inp, prev_spend_inp), COLOR_NAVY)

    def calcular_valor_kpi(df_actual, df_historico, var_sel):
        if df_actual is None or var_sel not in df_actual.columns:
            return "N/A", get_delta_badge(0, None)
        es_acum = any(k in var_sel.lower() for k in ["litros", "venta", "sales", "valor", "awo"])
        curr_val = df_actual[var_sel].sum() if es_acum else df_actual[var_sel].mean()
        prev_val = None
        if prev_cys and df_historico is not None and 'CY' in df_historico.columns:
            p_rows = df_historico[df_historico['CY'].isin(prev_cys)]
            if not p_rows.empty and var_sel in p_rows.columns:
                prev_val = p_rows[var_sel].sum() if es_acum else p_rows[var_sel].mean()
        es_pct = df_actual[var_sel].mean() <= 1.0 and df_actual[var_sel].mean() >= 0.0
        val_str = f"{curr_val*100:.1f}%" if es_pct else (f"{curr_val:,.0f}" if curr_val > 100 else f"{curr_val:,.2f}")
        return val_str, get_delta_badge(curr_val, prev_val)

    idx_col = 1
    if vars_sales:
        sel_var_sales = sincronizar_var_sesion("sales", vars_sales)
        v_str, d_badge = calcular_valor_kpi(sales_f, sales_df, sel_var_sales)
        render_kpi_card(kpi_cols[idx_col], "sales", "Ventas", vars_sales, v_str, d_badge, COLOR_CORNFLOWER)
        idx_col += 1

    if vars_brand:
        sel_var_brand = sincronizar_var_sesion("brand", vars_brand)
        v_str, d_badge = calcular_valor_kpi(brand_f, brand_df, sel_var_brand)
        render_kpi_card(kpi_cols[idx_col], "brand", "Brand Health", vars_brand, v_str, d_badge, COLOR_NAVY)
        idx_col += 1

    if vars_ext:
        sel_var_ext = sincronizar_var_sesion("ext", vars_ext)
        v_str, d_badge = calcular_valor_kpi(ext_f, ext_df, sel_var_ext)
        render_kpi_card(kpi_cols[idx_col], "ext", "Factores Externos", vars_ext, v_str, d_badge, COLOR_TEAL)
        idx_col += 1

    if vars_ctrl:
        sel_var_ctrl = sincronizar_var_sesion("ctrl", vars_ctrl)
        v_str, d_badge = calcular_valor_kpi(ctrl_f, ctrl_df, sel_var_ctrl)
        render_kpi_card(kpi_cols[idx_col], "ctrl", "Variables Control", vars_ctrl, v_str, d_badge, "#7c3aed")

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráficos Inputs
    c1, c2 = st.columns([2, 1])
    ch_l = [canales_meta[ch]['name'] for ch in selected_media_ids]
    with c1:
        fig_sb = go.Figure()
        for idx, cy in enumerate(active_cys):
            cy_rows = spend_df[spend_df['CY'] == cy]
            cy_v = [cy_rows[ch].sum() if ch in cy_rows.columns else 0.0 for ch in selected_media_ids]
            fig_sb.add_trace(go.Bar(x=ch_l, y=cy_v, name=cy, marker_color=color_idx(idx)))
        fig_sb.update_layout(barmode='group', height=340, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(title="Inversión ($)", gridcolor="#f1f5f9"), xaxis=dict(tickangle=-25), plot_bgcolor='white')
        st.plotly_chart(fig_sb, use_container_width=True)

    with c2:
        t_by_ch = [spend_f[ch].sum() if ch in spend_f.columns else 0.0 for ch in selected_media_ids]
        c_pie = [canales_meta[ch]['color'] for ch in selected_media_ids]
        fig_d = go.Figure(data=[go.Pie(labels=ch_l, values=t_by_ch, hole=0.55, marker=dict(colors=c_pie), textinfo='percent')])
        fig_d.update_layout(title="Mix de Inversión (% Share)", height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sales vs Spend Apilado
    var_v_inp = st.session_state.get("kpi_sel_sales", vars_sales[0] if vars_sales else None)
    if var_v_inp not in vars_sales: var_v_inp = vars_sales[0] if vars_sales else None

    cs_1, cs_2 = st.columns([1, 2])
    with cs_1:
        cy_st = [sales_df[sales_df['CY'] == cy][var_v_inp].sum() for cy in active_cys]
        fig_cys = go.Figure(data=[go.Bar(x=active_cys, y=cy_st, marker_color=COLOR_NAVY)])
        fig_cys.update_layout(title=f"Total {var_v_inp} por CY", height=360, margin=dict(l=20, r=20, t=35, b=20), yaxis=dict(gridcolor="#f1f5f9"), plot_bgcolor='white')
        st.plotly_chart(fig_cys, use_container_width=True)

    with cs_2:
        fig_stk = go.Figure()
        d_col = 'Date' if 'Date' in spend_f.columns else spend_f.columns[1]
        for ch in selected_media_ids:
            if ch in spend_f.columns:
                fig_stk.add_trace(go.Bar(x=spend_f[d_col], y=spend_f[ch], name=canales_meta[ch]['name'], marker_color=canales_meta[ch]['color'], yaxis='y1'))
        if var_v_inp and var_v_inp in sales_f.columns:
            fig_stk.add_trace(go.Scatter(x=sales_f[d_col], y=sales_f[var_v_inp], name=f"{var_v_inp}", mode='lines+markers', line=dict(color=COLOR_CORNFLOWER, width=3), marker=dict(size=4, color=COLOR_NAVY), yaxis='y2'))
        fig_stk.update_layout(
            title=dict(text="Media Spend vs Ventas", font=dict(size=12, color="#475569")),
            barmode='stack', height=360, margin=dict(l=20, r=20, t=35, b=65),
            yaxis=dict(title="Inversión ($)", side='left', showgrid=True, gridcolor="#f1f5f9"),
            yaxis2=dict(title=var_v_inp, side='right', overlaying='y', showgrid=False),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8, color="#64748b")),
            legend=dict(orientation="h", yanchor="bottom", y=-0.55, xanchor="center", x=0.5, font=dict(size=9)),
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_stk, use_container_width=True)

# ##############################################################################
# TAB 3: MEDIA OPTIMIZATION (OPTIMIZADOR S-CURVE)[cite: 7]
# ##############################################################################
with tab_opt:
    if curv_df is None:
        st.warning("⚠️ Para utilizar el optimizador de medios, por favor asigna el archivo de **Curvas Sat. / Touchpoints** en la barra lateral.")
    else:
        st.markdown(f"""
            <div class="section-card">
                <span class="section-badge">⚡ Algoritmo SLSQP / S-Curves</span>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #000050;">Media Mix & Budget Optimization Engine</h3>
                <p style="margin: 0.2rem 0 0 0; font-size: 0.85rem; color: #64748b;">
                    Optimización matemática con retornos decrecientes para maximizar el Revenue Incremental anual en función de curvas de respuesta.
                </p>
            </div>
        """, unsafe_allow_html=True)

        col_opt_cfg1, col_opt_cfg2, col_opt_cfg3 = st.columns([2, 1, 1.5])
        with col_opt_cfg1:
            budget_anual_input = st.number_input(
                "Presupuesto Anual Total a Optimizar ($ MXN):",
                min_value=10000.0,
                max_value=500000000.0,
                value=float(spend_f[selected_media_ids].sum().sum()) if selected_media_ids and spend_f[selected_media_ids].sum().sum() > 0 else 5000000.0,
                step=50000.0,
                format="%.2f"
            )
        with col_opt_cfg2:
            meses_actividad = st.number_input(
                "Meses con Actividad:",
                min_value=1,
                max_value=12,
                value=12,
                step=1
            )
        with col_opt_cfg3:
            tipo_optimizacion = st.radio(
                "Modalidad de Optimización:",
                options=["1. Libre (Auto-Scale Curvas S)", "2. Con Restricciones (Límites Min/Max)"],
                index=0
            )

        limites_finales = None

        if "2. Con Restricciones" in tipo_optimizacion:
            limites_sugeridos = preparar_y_obtener_limites_engine(curv_df, budget_anual_input, meses_actividad)
            df_limites_editor = pd.DataFrame([
                {
                    "Medio": canal,
                    "Mínimo Mensual ($)": float(vals['min']),
                    "Máximo Mensual ($)": float(vals['max'])
                }
                for canal, vals in limites_sugeridos.items()
            ])

            edited_df = st.data_editor(
                df_limites_editor,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Medio": st.column_config.TextColumn(disabled=True),
                    "Mínimo Mensual ($)": st.column_config.NumberColumn(format="$%.2f", min_value=0.0),
                    "Máximo Mensual ($)": st.column_config.NumberColumn(format="$%.2f", min_value=0.0),
                }
            )

            limites_finales = {
                row["Medio"]: {"min": float(row["Mínimo Mensual ($)"]), "max": float(row["Máximo Mensual ($)"])}
                for _, row in edited_df.iterrows()
            }

        btn_optimizar = st.button("🚀 Ejecutar Optimización de Presupuesto", type="primary", use_container_width=True)

        if btn_optimizar or "df_opt_cache" in st.session_state:
            if btn_optimizar:
                with st.spinner("Ejecutando algoritmo SLSQP con curvas de respuesta..."):
                    df_res_opt, sum_rev_opt, roi_glob_opt = ejecutar_optimizacion_engine(
                        curv_df, 
                        budget_anual=budget_anual_input, 
                        meses_con_actividad=meses_actividad, 
                        limites_personalizados=limites_finales
                    )
                    st.session_state["df_opt_cache"] = df_res_opt
                    st.session_state["rev_opt_cache"] = sum_rev_opt
                    st.session_state["roi_opt_cache"] = roi_glob_opt
                    st.session_state["budget_opt_cache"] = budget_anual_input

            df_opt = st.session_state.get("df_opt_cache")
            rev_opt = st.session_state.get("rev_opt_cache", 0.0)
            roi_opt = st.session_state.get("roi_opt_cache", 0.0)

            if df_opt is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                opt_kpi1, opt_kpi2, opt_kpi3 = st.columns(3)
                with opt_kpi1:
                    st.markdown(f"""
                        <div class="kpi-container" style="min-height: 110px;">
                            <div>
                                <div class="kpi-title">Presupuesto Asignado Verificado</div>
                                <div class="kpi-value" style="color:{COLOR_NAVY};">${df_opt['Inversión Óptima Anual'].sum():,.0f}</div>
                            </div>
                            <div class="kpi-delta" style="color:#16a34a;">100.0% Ejecutado</div>
                        </div>
                    """, unsafe_allow_html=True)
                with opt_kpi2:
                    st.markdown(f"""
                        <div class="kpi-container" style="min-height: 110px;">
                            <div>
                                <div class="kpi-title">Media Revenue Proyectado</div>
                                <div class="kpi-value" style="color:{COLOR_CORNFLOWER};">${rev_opt:,.0f}</div>
                            </div>
                            <div class="kpi-delta" style="color:{COLOR_CORNFLOWER};">Retorno Total Estimado</div>
                        </div>
                    """, unsafe_allow_html=True)
                with opt_kpi3:
                    st.markdown(f"""
                        <div class="kpi-container" style="min-height: 110px;">
                            <div>
                                <div class="kpi-title">ROI Global Proyectado</div>
                                <div class="kpi-value" style="color:{COLOR_TEAL};">{roi_opt:.4f}x</div>
                            </div>
                            <div class="kpi-delta" style="color:#16a34a;">Eficiencia Ponderada</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_graf_opt1, col_graf_opt2 = st.columns([1.2, 1.8])

                with col_graf_opt1:
                    df_plot_mix = df_opt.sort_values('Digital Mix', ascending=True).reset_index(drop=True)
                    fig_mix100 = go.Figure()
                    for i, r_mix in df_plot_mix.iterrows():
                        pct_val = r_mix['Digital Mix'] * 100
                        texto_adentro = f"{pct_val:.1f}%" if pct_val >= 3.0 else ""
                        fig_mix100.add_trace(go.Bar(
                            x=["Media Mix"], y=[pct_val], name=f"{r_mix['Medio']} ({pct_val:.1f}%)",
                            marker_color=color_idx(i), text=texto_adentro, textposition='inside',
                            textfont=dict(size=15, color='white', family='Arial Black'), insidetextanchor='middle',
                            hovertemplate=f"<b>{r_mix['Medio']}</b><br>Share: {pct_val:.2f}%<br>Monto: ${r_mix['Inversión Óptima Anual']:,.0f}<extra></extra>"
                        ))
                    fig_mix100.update_layout(
                        barmode='stack', height=420, margin=dict(l=20, r=20, t=25, b=20),
                        yaxis=dict(title="% Participación", range=[0, 100], ticksuffix="%", gridcolor="#f1f5f9"),
                        xaxis=dict(showticklabels=False), legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="center", x=0.5, font=dict(size=11, color='#1e293b')),
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_mix100, use_container_width=True)

                with col_graf_opt2:
                    fig_comp_opt = go.Figure()
                    fig_comp_opt.add_trace(go.Bar(
                        x=df_opt['Medio'], y=df_opt['Inversión Óptima Anual'],
                        name="Inversión Óptima Anual ($)", marker_color=COLOR_NAVY,
                        hovertemplate="<b>%{x}</b><br>Presupuesto: $%{y:,.0f}<extra></extra>"
                    ))
                    fig_comp_opt.add_trace(go.Bar(
                        x=df_opt['Medio'], y=df_opt['Media Revenue Anual'],
                        name="Media Revenue Anual ($)", marker_color=COLOR_CORNFLOWER,
                        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
                    ))
                    fig_comp_opt.update_layout(
                        barmode='group', height=420, margin=dict(l=20, r=20, t=25, b=20),
                        yaxis=dict(title="Monto ($ MXN)", gridcolor="#f1f5f9", tickprefix="$", tickformat=",.0f"),
                        xaxis=dict(tickangle=-25), legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(size=9)),
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_comp_opt, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                df_opt_display = df_opt.copy()
                df_opt_display['Inversión Óptima ($)'] = df_opt_display['Inversión Óptima Anual'].apply(lambda x: f"${x:,.2f}")
                df_opt_display['ROI Marginal Estimado'] = df_opt_display['ROI Estimado'].apply(lambda x: f"{x:.4f}")
                df_opt_display['Revenue Estimado ($)'] = df_opt_display['Media Revenue Anual'].apply(lambda x: f"${x:,.2f}")
                df_opt_display['Media Mix (%)'] = df_opt_display['Digital Mix'].apply(lambda x: f"{x*100:.2f}%")

                st.dataframe(
                    df_opt_display[['Medio', 'Inversión Óptima ($)', 'ROI Marginal Estimado', 'Revenue Estimado ($)', 'Media Mix (%)']],
                    use_container_width=True, hide_index=True
                )

# ##############################################################################
# TAB 4: PIPELINE FUNNEL (SIMULADOR CROSS MEDIA + LIFT AWOS)[cite: 8]
# ##############################################################################
with tab_pipeline:
    st.markdown(f"""
        <div class="section-card">
            <span class="section-badge">🍕 Multi-MMM Funnel Simulator</span>
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #000050;">Pipeline de Medición Cross Media en el Funnel</h3>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.85rem; color: #64748b;">
                Simulación Dinámica de Propagación de Medios a través de Intermediarios de Marketing (Base T0 + Aportes Predictivos por Período).
            </p>
        </div>
    """, unsafe_allow_html=True)

    if df_pipeline_raw is None:
        st.warning("⚠️ Carga el archivo **Template_Pipeline_MiA_2.xlsx** o **.csv** en la sección **📂 3. Pipeline Funnel** de la barra lateral para activar este módulo.")
    else:
        df_pipe = df_pipeline_raw.copy()
        df_pipe['DataType_Clean'] = df_pipe['DataType'].astype(str).str.strip().str.lower()
        mask_actual = df_pipe['DataType_Clean'] == 'actual'
        mask_pred = df_pipe['DataType_Clean'].str.contains('pred')

        df_act = df_pipe[mask_actual]
        df_pr = df_pipe[mask_pred] if not df_pipe[mask_pred].empty else df_pipe

        if df_act.empty:
            st.error("El archivo cargado debe contener filas con DataType 'Actual' para calcular los promedios en tiempo cero (T0).")
        else:
            # 1. Promedios base en Tiempo Cero (T0)
            t0_pm = float(pd.to_numeric(df_act['Circulo_PaidMedia'], errors='coerce').mean())
            t0_ma = float(pd.to_numeric(df_act['Circulo_MentalAvailability'], errors='coerce').mean())
            t0_cf = float(pd.to_numeric(df_act['Circulo_ConsumoFrecuente'], errors='coerce').mean())
            t0_ad = float(pd.to_numeric(df_act['Circulo_ActivacionDigital'], errors='coerce').mean())
            t0_awos = float(pd.to_numeric(df_act['Circulo_AWOS'], errors='coerce').mean())

            # 2. Procesar semanas predictivas
            semanas_data = []
            for idx_p, (_, row_p) in enumerate(df_pr.iterrows(), start=1):
                fecha_dt = pd.to_datetime(row_p.get('DateWeekClose'))
                fecha_fmt = fecha_dt.strftime("%d/%m/%Y") if pd.notna(fecha_dt) else str(row_p.get('DateWeekClose'))
                
                # Aportes puntuales en las flechas
                fl_pm_awos = float(pd.to_numeric(row_p.get('Flecha_PM_a_AWOs', 0), errors='coerce'))
                fl_pm_ma = float(pd.to_numeric(row_p.get('Flecha_PM_a_MA', 0), errors='coerce'))
                fl_pm_cf = float(pd.to_numeric(row_p.get('Flecha_PM_a_CF', 0), errors='coerce'))
                fl_pm_ad = float(pd.to_numeric(row_p.get('Flecha_PM_a_AD', 0), errors='coerce'))
                fl_ma_cf = float(pd.to_numeric(row_p.get('Flecha_MA_a_CF', 0), errors='coerce'))
                fl_ma_ad = float(pd.to_numeric(row_p.get('Flecha_MA_a_AD', 0), errors='coerce'))
                
                # Aportes totales recibidos en la semana t
                recibe_pm = fl_pm_awos + fl_pm_ma + fl_pm_cf + fl_pm_ad
                recibe_ma = fl_pm_ma
                recibe_cf = fl_pm_cf + fl_ma_cf
                recibe_ad = fl_pm_ad + fl_ma_ad
                recibe_awos = fl_pm_awos

                # Cálculo dinámico de los círculos (Promedio T0 + Aporte)
                val_c_pm = pd.to_numeric(row_p.get('Circulo_PaidMedia'), errors='coerce')
                c_pm_final = float(val_c_pm) if pd.notna(val_c_pm) and val_c_pm > 0 else ((t0_pm + recibe_pm) / 2.0)

                val_c_ma = pd.to_numeric(row_p.get('Circulo_MentalAvailability'), errors='coerce')
                c_ma_final = float(val_c_ma) if pd.notna(val_c_ma) and val_c_ma > 0 else ((t0_ma + recibe_ma) / 2.0)

                val_c_cf = pd.to_numeric(row_p.get('Circulo_ConsumoFrecuente'), errors='coerce')
                c_cf_final = float(val_c_cf) if pd.notna(val_c_cf) and val_c_cf > 0 else ((t0_cf + recibe_cf) / 2.0)

                val_c_ad = pd.to_numeric(row_p.get('Circulo_ActivacionDigital'), errors='coerce')
                c_ad_final = float(val_c_ad) if pd.notna(val_c_ad) and val_c_ad > 0 else ((t0_ad + recibe_ad) / 2.0)

                val_c_awos = pd.to_numeric(row_p.get('Circulo_AWOS'), errors='coerce')
                c_awos_final = float(val_c_awos) if pd.notna(val_c_awos) and val_c_awos > 0 else ((t0_awos + recibe_awos) / 2.0)

                indir_val = fl_pm_ma + fl_pm_cf + fl_pm_ad
                tot_val = fl_pm_awos + indir_val
                mult = (indir_val / fl_pm_awos * 100) if fl_pm_awos > 0 else 0.0

                semanas_data.append({
                    "semana": idx_p,
                    "fecha": fecha_fmt,
                    "c_pm": f"{c_pm_final:,.1f}",
                    "c_ma": f"{c_ma_final:,.1f}",
                    "c_cf": f"{c_cf_final:,.1f}",
                    "c_ad": f"{c_ad_final:,.1f}",
                    "c_awos": f"{c_awos_final:,.1f}",
                    "f_pm_awos": f"+{fl_pm_awos:,.1f}",
                    "f_pm_ma": f"+{fl_pm_ma:,.1f}",
                    "f_pm_cf": f"+{fl_pm_cf:,.1f}",
                    "f_pm_ad": f"+{fl_pm_ad:,.1f}",
                    "f_ma_cf": f"+{fl_ma_cf:,.1f}",
                    "f_ma_ad": f"+{fl_ma_ad:,.1f}",
                    "res_dir": f"{fl_pm_awos:,.1f}",
                    "res_indir": f"{indir_val:,.1f}",
                    "res_tot": f"{tot_val:,.1f}",
                    "res_mult": f"+{mult:.1f}%"
                })

            semanas_json = json.dumps(semanas_data)
            total_steps = len(semanas_data)

            # 3. HTML / SVG ANIMADO REACTIVO (CON textContent)
            html_pipeline = f"""<!-- pipeline_{pipeline_hash}_{total_steps} -->
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <style>
                    body {{ margin: 0; padding: 8px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f8fafc; color: #1e293b; }}
                    .funnel-container {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }}
                    .control-bar {{ display: flex; align-items: center; justify-content: space-between; background: #000050; padding: 10px 16px; border-radius: 8px; color: white; margin-bottom: 12px; }}
                    .btn {{ background: #B0F467; color: #000050; border: none; padding: 7px 14px; border-radius: 6px; font-weight: 800; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px; }}
                    .btn:hover {{ background: #9be253; transform: scale(1.02); }}
                    .btn-secondary {{ background: rgba(255, 255, 255, 0.15); color: white; margin-left: 8px; }}
                    .slider-control {{ flex: 1; margin: 0 16px; accent-color: #B0F467; }}
                    .flow-line-direct {{ stroke-dasharray: 8; animation: flow-forward 1.2s linear infinite; }}
                    .flow-line-indirect {{ stroke-dasharray: 6; animation: flow-forward 1.8s linear infinite; }}
                    @keyframes flow-forward {{ from {{ stroke-dashoffset: 40; }} to {{ stroke-dashoffset: 0; }} }}
                    .kpi-card {{ background: white; border: 2px solid #000050; border-radius: 10px; padding: 12px 16px; box-shadow: 0 8px 16px -4px rgba(0, 0, 80, 0.12); }}
                </style>
            </head>
            <body>

            <div class="funnel-container">
                <div class="control-bar">
                    <div style="display: flex; align-items: center;">
                        <button id="btnPlay" class="btn" onclick="togglePlay()"><i class="fa-solid fa-play"></i> Iniciar Recorrido</button>
                        <button class="btn btn-secondary" onclick="resetSim()"><i class="fa-solid fa-rotate-left"></i> Reiniciar</button>
                    </div>
                    <input type="range" id="weekSlider" min="1" max="{total_steps}" value="1" class="slider-control" oninput="manualSlider(this.value)">
                    <div style="font-weight: 800; font-size: 0.95rem; min-width: 220px; text-align: right;">
                        <span style="color: #93DFE3;">Período:</span> <span id="lblWeek" style="color: #B0F467;">Semana 1</span> | <span id="lblDate" style="color: #ffffff; font-size: 0.85rem;">-</span>
                    </div>
                </div>

                <svg id="funnelSvg" viewBox="0 0 1050 560" style="width: 100%; height: auto;">
                    <defs>
                        <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#000050" /></marker>
                        <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" /></marker>
                    </defs>

                    <!-- LEYENDA -->
                    <g transform="translate(30, 20)">
                        <text x="0" y="0" font-size="12" font-weight="bold" fill="#64748b">Contribución del Funnel por:</text>
                        <circle cx="6" cy="18" r="4" fill="#000050" />
                        <text x="18" y="22" font-size="11" font-weight="bold" fill="#000050">Paid Media (Flujo Directo)</text>
                        <circle cx="6" cy="36" r="4" fill="#dc2626" />
                        <text x="18" y="40" font-size="11" font-weight="bold" fill="#dc2626">Intermediarios de Marketing (Flujo Indirecto)</text>
                    </g>

                    <!-- FLECHAS / APORTES -->
                    <!-- PM -> AWOS -->
                    <line x1="205" y1="290" x2="680" y2="290" stroke="#000050" stroke-width="4" marker-end="url(#arrow-blue)" class="flow-line-direct" />
                    <rect x="290" y="275" width="60" height="22" rx="4" fill="#ffffff" stroke="#000050" stroke-width="1.5"/>
                    <text id="f_pm_awos" x="320" y="290" font-size="11" font-weight="bold" fill="#000050" text-anchor="middle">+0</text>

                    <!-- PM -> MA -->
                    <path d="M 170 235 L 170 120 L 325 120" fill="none" stroke="#000050" stroke-width="3" marker-end="url(#arrow-blue)" class="flow-line-direct" />
                    <rect x="235" y="108" width="55" height="20" rx="4" fill="#ffffff" stroke="#000050" stroke-width="1.5"/>
                    <text id="f_pm_ma" x="262" y="122" font-size="10" font-weight="bold" fill="#000050" text-anchor="middle">+0</text>

                    <!-- PM -> AD -->
                    <path d="M 170 345 L 170 460 L 440 460" fill="none" stroke="#000050" stroke-width="3" marker-end="url(#arrow-blue)" class="flow-line-direct" />
                    <rect x="265" y="448" width="55" height="20" rx="4" fill="#ffffff" stroke="#000050" stroke-width="1.5"/>
                    <text id="f_pm_ad" x="292" y="462" font-size="10" font-weight="bold" fill="#000050" text-anchor="middle">+0</text>

                    <!-- PM -> CF -->
                    <path d="M 210 280 L 490 230" fill="none" stroke="#000050" stroke-width="2.5" marker-end="url(#arrow-blue)" class="flow-line-direct" />
                    <rect x="330" y="240" width="55" height="20" rx="4" fill="#ffffff" stroke="#000050" stroke-width="1.5"/>
                    <text id="f_pm_cf" x="357" y="254" font-size="10" font-weight="bold" fill="#000050" text-anchor="middle">+0</text>

                    <!-- MA -> CF -->
                    <path d="M 380 172 L 380 215 L 490 215" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5" marker-end="url(#arrow-red)" class="flow-line-indirect" />
                    <rect x="388" y="198" width="58" height="18" rx="3" fill="#ffffff" stroke="#dc2626" stroke-width="1"/>
                    <text id="f_ma_cf" x="417" y="211" font-size="10" font-weight="bold" fill="#dc2626" text-anchor="middle">+0</text>

                    <!-- MA -> AD -->
                    <path d="M 370 172 L 370 460 L 440 460" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5" marker-end="url(#arrow-red)" class="flow-line-indirect" />
                    <rect x="340" y="315" width="58" height="18" rx="3" fill="#ffffff" stroke="#dc2626" stroke-width="1"/>
                    <text id="f_ma_ad" x="369" y="328" font-size="10" font-weight="bold" fill="#dc2626" text-anchor="middle">+0</text>

                    <!-- Brand Image de soporte -->
                    <g transform="translate(830, 190)">
                        <circle cx="0" cy="0" r="46" fill="#000050" stroke="#788BFF" stroke-width="3"/>
                        <text x="0" y="-4" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Brand</text>
                        <text x="0" y="12" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Image</text>
                    </g>
                    <path d="M 830 240 L 830 290 L 790 290" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5" marker-end="url(#arrow-red)" class="flow-line-indirect" />
                    <text x="815" y="275" font-size="11" font-weight="bold" fill="#dc2626">8%</text>

                    <!-- ==================== CÍRCULOS ==================== -->
                    <!-- Paid Media -->
                    <g transform="translate(150, 290)">
                        <circle cx="0" cy="0" r="55" fill="#000050" stroke="#B0F467" stroke-width="4"/>
                        <text x="0" y="-12" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">Paid Media</text>
                        <text id="c_pm" x="0" y="8" font-size="14" font-weight="900" fill="#B0F467" text-anchor="middle">0.0</text>
                        <text x="0" y="24" font-size="7.5" fill="#93DFE3" text-anchor="middle">Base T0: {t0_pm/1e6:.2f}M</text>
                    </g>

                    <!-- Mental Availability -->
                    <g transform="translate(380, 120)">
                        <circle cx="0" cy="0" r="52" fill="#93DFE3" stroke="#000050" stroke-width="3"/>
                        <text x="0" y="-12" font-size="11" font-weight="bold" fill="#000050" text-anchor="middle">Mental Avail.</text>
                        <text id="c_ma" x="0" y="8" font-size="14" font-weight="900" fill="#dc2626" text-anchor="middle">0.0</text>
                        <text x="0" y="24" font-size="7.5" fill="#64748b" text-anchor="middle">Base T0: {t0_ma/1e6:.2f}M</text>
                    </g>

                    <!-- Activación Digital -->
                    <g transform="translate(500, 460)">
                        <circle cx="0" cy="0" r="52" fill="#788BFF" stroke="#000050" stroke-width="3"/>
                        <text x="0" y="-12" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">Activación Dig.</text>
                        <text id="c_ad" x="0" y="8" font-size="14" font-weight="900" fill="#B0F467" text-anchor="middle">0.0</text>
                        <text x="0" y="24" font-size="7.5" fill="#e2e8f0" text-anchor="middle">Base T0: {t0_ad/1e6:.2f}M</text>
                    </g>

                    <!-- Consumo Frecuente -->
                    <g transform="translate(550, 220)">
                        <circle cx="0" cy="0" r="52" fill="#00DBEE" stroke="#000050" stroke-width="3"/>
                        <text x="0" y="-12" font-size="11" font-weight="bold" fill="#000050" text-anchor="middle">Consumo Frec.</text>
                        <text id="c_cf" x="0" y="8" font-size="14" font-weight="900" fill="#000050" text-anchor="middle">0.0</text>
                        <text x="0" y="24" font-size="7.5" fill="#64748b" text-anchor="middle">Base T0: {t0_cf/1e6:.2f}M</text>
                    </g>

                    <!-- AWOS -->
                    <g transform="translate(735, 290)">
                        <circle cx="0" cy="0" r="55" fill="#000050" stroke="#dc2626" stroke-width="4"/>
                        <text x="0" y="-12" font-size="14" font-weight="900" fill="#ffffff" text-anchor="middle">AWOS</text>
                        <text id="c_awos" x="0" y="8" font-size="14" font-weight="900" fill="#B0F467" text-anchor="middle">0.0</text>
                        <text x="0" y="25" font-size="7.5" fill="#93DFE3" text-anchor="middle">Base T0: {t0_awos/1e6:.2f}M</text>
                    </g>
                </svg>

                <!-- RESUMEN -->
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 15px; border-top: 1px solid #e2e8f0; padding-top: 15px;">
                    <div style="font-size: 0.8rem; color: #64748b;">
                        <b>Ajuste del Modelo:</b> R² Adj: <span style="background: #B0F467; color: #000050; padding: 2px 6px; border-radius: 4px; font-weight: bold;">0.7214</span> | F-stat: 14.2 (p < 0.001)<br>
                        <i>Digital Activation: Descargas de App y Visitas Web | Mental Availability: TOM</i>
                    </div>

                    <div class="kpi-card" style="min-width: 310px;">
                        <div style="font-size: 0.95rem; font-weight: 800; color: #000050; margin-bottom: 6px;">Aportación de Medios en la Semana</div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                            <span>• Directo a AWOS:</span> <b id="res_dir" style="color: #000050;">0</b>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;">
                            <span>• Intermediarios Marketing:</span> <b id="res_indir" style="color: #dc2626;">0</b>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 1.15rem; font-weight: 900; border-top: 2px solid #000050; padding-top: 6px;">
                            <span>Total Generado:</span> <span id="res_tot" style="color: #000050;">0</span>
                        </div>
                        <div style="font-size: 0.78rem; color: #059669; font-weight: 800; text-align: right; margin-top: 4px;">
                            Multiplicador Indirecto: <span id="res_mult">+0.0%</span>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                const simData = {semanas_json};
                let currentIdx = 0;
                let timer = null;
                let isPlaying = false;
                const speedMs = 1200;

                function setText(id, value) {{
                    const el = document.getElementById(id);
                    if (el) {{
                        el.textContent = value;
                    }}
                }}

                function updateView(idx) {{
                    if (!simData || simData.length === 0) return;
                    const row = simData[idx];
                    
                    setText('lblWeek', 'Semana ' + row.semana);
                    setText('lblDate', row.fecha);
                    const slider = document.getElementById('weekSlider');
                    if (slider) slider.value = row.semana;
                    
                    // Círculos SVG
                    setText('c_pm', row.c_pm);
                    setText('c_ma', row.c_ma);
                    setText('c_cf', row.c_cf);
                    setText('c_ad', row.c_ad);
                    setText('c_awos', row.c_awos);
                    
                    // Flechas SVG
                    setText('f_pm_awos', row.f_pm_awos);
                    setText('f_pm_ma', row.f_pm_ma);
                    setText('f_pm_cf', row.f_pm_cf);
                    setText('f_pm_ad', row.f_pm_ad);
                    setText('f_ma_cf', row.f_ma_cf);
                    setText('f_ma_ad', row.f_ma_ad);
                    
                    // Resumen
                    setText('res_dir', row.res_dir);
                    setText('res_indir', row.res_indir);
                    setText('res_tot', row.res_tot);
                    setText('res_mult', row.res_mult);
                }}

                function togglePlay() {{
                    const btn = document.getElementById('btnPlay');
                    if (isPlaying) {{
                        clearInterval(timer);
                        isPlaying = false;
                        btn.innerHTML = '<i class="fa-solid fa-play"></i> Continuar';
                    }} else {{
                        isPlaying = true;
                        btn.innerHTML = '<i class="fa-solid fa-pause"></i> Pausar';
                        timer = setInterval(() => {{
                            currentIdx++;
                            if (currentIdx >= simData.length) currentIdx = 0;
                            updateView(currentIdx);
                        }}, speedMs);
                    }}
                }}

                function resetSim() {{
                    if (isPlaying) togglePlay();
                    currentIdx = 0;
                    updateView(0);
                }}

                function manualSlider(val) {{
                    if (isPlaying) togglePlay();
                    currentIdx = parseInt(val) - 1;
                    updateView(currentIdx);
                }}

                updateView(0);
            </script>
            </body>
            </html>
            """

            container_canvas = st.empty()
            with container_canvas:
                components.html(html_pipeline, height=720, scrolling=False)

            st.markdown("### 📋 Vista Previa de las Semanas Predictivas Procesadas")
            st.dataframe(pd.DataFrame(semanas_data), use_container_width=True, hide_index=True)

            # ==================================================================
            # SECCIÓN INSPIRADA EN STARBUCKS: SIMULADOR DE LIFT EN AWOS CON CURVAS S
            # ==================================================================
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="section-card" style="border: 2px solid #000050;">
                    <span class="section-badge">⚖️ Elasticity & Lift Model</span>
                    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #000050;">
                        Simulador Interactivo de Inversión y Lift Esperado en Transacciones (AWOS)
                    </h3>
                    <p style="margin: 0.2rem 0 0 0; font-size: 0.85rem; color: #64748b;">
                        Ajusta el presupuesto total para recalcular el impacto esperado en órdenes por tienda (AWOS) y transacciones totales utilizando las Curvas S del modelo.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # 1. Parámetros interactivos del simulador
            col_sim1, col_sim2, col_sim3 = st.columns([2, 1.2, 1.2])

            with col_sim1:
                budget_base_referencia = float(t0_pm) if t0_pm > 10000 else 5000000.0
                presupuesto_sim_lift = st.slider(
                    "Presupuesto Anual de Medios a Simular ($ MXN):",
                    min_value=float(max(100000.0, budget_base_referencia * 0.3)),
                    max_value=float(budget_base_referencia * 3.0),
                    value=float(budget_base_referencia),
                    step=50000.0,
                    format="$%d"
                )

            with col_sim2:
                tiendas_activas = st.number_input(
                    "Número de Tiendas / Unidades:",
                    min_value=1,
                    max_value=5000,
                    value=859,
                    step=10
                )

            with col_sim3:
                awos_base_sim = st.number_input(
                    "AWOS Base Histórico (Órdenes/Sem):",
                    min_value=10.0,
                    max_value=100000.0,
                    value=float(t0_awos) if t0_awos > 0 else 2733.2,
                    step=25.0
                )

            # 2. Evaluación matemática con Curvas S (curv_df)
            if curv_df is not None:
                df_opt_sim, rev_opt_sim, roi_opt_sim = ejecutar_optimizacion_engine(
                    curv_df, 
                    budget_anual=presupuesto_sim_lift, 
                    meses_con_actividad=12
                )
                lift_incremental_anual = rev_opt_sim
                lift_incremental_semanal_total = lift_incremental_anual / 52.0
                lift_awos_por_tienda = lift_incremental_semanal_total / tiendas_activas
                awos_lift_pct = (lift_awos_por_tienda / awos_base_sim) * 100 if awos_base_sim > 0 else 0.0
                awos_proyectado = awos_base_sim + lift_awos_por_tienda
                transacciones_totales_semanales = int(awos_proyectado * tiendas_activas)
                mix_digital_estimado = df_opt_sim['Digital Mix'].sum() * 100 if 'Digital Mix' in df_opt_sim.columns else 38.5
            else:
                delta_budget = (presupuesto_sim_lift - budget_base_referencia) / budget_base_referencia if budget_base_referencia > 0 else 0
                awos_lift_pct = 1.80 + (delta_budget * 3.5)
                awos_proyectado = awos_base_sim * (1 + (awos_lift_pct / 100.0))
                transacciones_totales_semanales = int(awos_proyectado * tiendas_activas)
                mix_digital_estimado = 38.59
                roi_opt_sim = 1.45
                df_opt_sim = pd.DataFrame({
                    'Medio': ['Social Media', 'Online Video (OLV)', 'Search', 'OOH', 'Streaming'],
                    'Inversión Óptima Anual': [presupuesto_sim_lift * 0.35, presupuesto_sim_lift * 0.25, presupuesto_sim_lift * 0.15, presupuesto_sim_lift * 0.15, presupuesto_sim_lift * 0.10],
                    'Digital Mix': [0.35, 0.25, 0.15, 0.15, 0.10],
                    'Media Revenue Anual': [presupuesto_sim_lift * 0.35 * 1.6, presupuesto_sim_lift * 0.25 * 1.5, presupuesto_sim_lift * 0.15 * 1.8, presupuesto_sim_lift * 0.15 * 1.1, presupuesto_sim_lift * 0.10 * 1.2]
                })

            # 3. Tarjetas Ejecutivas de Lift en AWOS
            st.markdown("<br>", unsafe_allow_html=True)
            kpi_aw1, kpi_aw2, kpi_aw3 = st.columns(3)

            with kpi_aw1:
                st.markdown(f"""
                    <div style="background-color: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 0.75rem; padding: 1.1rem 1.25rem;">
                        <span style="font-size: 0.7rem; font-weight: 800; color: #047857; text-transform: uppercase; letter-spacing: 0.05em; display: block;">
                            AWOS Proyectado Optimizado
                        </span>
                        <div style="font-size: 1.65rem; font-weight: 900; color: #064e3b; margin-top: 0.25rem;">
                            {awos_proyectado:,.1f} órdenes/sem
                        </div>
                        <span style="font-size: 0.75rem; font-weight: 700; color: #059669; display: block; margin-top: 0.35rem;">
                            {"+" if awos_lift_pct >= 0 else ""}{awos_lift_pct:.2f}% Lift Estimado por Tienda
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            with kpi_aw2:
                st.markdown(f"""
                    <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 0.75rem; padding: 1.1rem 1.25rem;">
                        <span style="font-size: 0.7rem; font-weight: 800; color: #1d4ed8; text-transform: uppercase; letter-spacing: 0.05em; display: block;">
                            Transacciones Semanales Totales
                        </span>
                        <div style="font-size: 1.65rem; font-weight: 900; color: #172554; margin-top: 0.25rem;">
                            {transacciones_totales_semanales:,.0f}
                        </div>
                        <span style="font-size: 0.75rem; font-weight: 700; color: #2563eb; display: block; margin-top: 0.35rem;">
                            Red de {tiendas_activas:,} Tiendas Activas
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            with kpi_aw3:
                st.markdown(f"""
                    <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-radius: 0.75rem; padding: 1.1rem 1.25rem;">
                        <span style="font-size: 0.7rem; font-weight: 800; color: #b45309; text-transform: uppercase; letter-spacing: 0.05em; display: block;">
                            ROI Marginal & Mix Digital
                        </span>
                        <div style="font-size: 1.65rem; font-weight: 900; color: #451a03; margin-top: 0.25rem;">
                            {roi_opt_sim:.2f}x | {mix_digital_estimado:.1f}% Mix
                        </div>
                        <span style="font-size: 0.75rem; font-weight: 700; color: #d97706; display: block; margin-top: 0.35rem;">
                            Impulsado por Curvas S y Performance
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            # 4. Comparativa Side-by-Side (Plan Base vs Plan Optimizado)
            st.markdown("<br>", unsafe_allow_html=True)
            col_side_base, col_side_opt = st.columns(2)

            with col_side_base:
                st.markdown(f"""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem 1.25rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #cbd5e1; padding-bottom: 0.5rem; margin-bottom: 0.75rem;">
                            <span style="font-size: 0.8rem; font-weight: 800; color: #334155; text-transform: uppercase;">Plan Inercial / Base</span>
                            <span style="font-size: 0.85rem; font-weight: 800; color: #64748b;">${presupuesto_sim_lift:,.0f}</span>
                        </div>
                """, unsafe_allow_html=True)

                # Desglose base proporcional
                n_canales = len(df_opt_sim)
                for idx_m, r_m in df_opt_sim.iterrows():
                    share_base = 1.0 / n_canales
                    monto_base = presupuesto_sim_lift * share_base
                    st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #475569; padding: 0.3rem 0; border-bottom: 1px dashed #e2e8f0;">
                            <span>{r_m['Medio']}</span>
                            <span><b>${monto_base:,.0f}</b> ({share_base*100:.1f}%)</span>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_side_opt:
                st.markdown(f"""
                    <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 0.75rem; padding: 1rem 1.25rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #22c55e; padding-bottom: 0.5rem; margin-bottom: 0.75rem;">
                            <span style="font-size: 0.8rem; font-weight: 800; color: #15803d; text-transform: uppercase;">Plan Optimizado (Curvas S) ✨</span>
                            <span style="font-size: 0.85rem; font-weight: 900; color: #166534;">${df_opt_sim['Inversión Óptima Anual'].sum():,.0f}</span>
                        </div>
                """, unsafe_allow_html=True)

                for idx_m, r_m in df_opt_sim.iterrows():
                    inv_opt_c = r_m['Inversión Óptima Anual']
                    pct_opt_c = (inv_opt_c / presupuesto_sim_lift) * 100 if presupuesto_sim_lift > 0 else 0
                    st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #1e293b; padding: 0.3rem 0; border-bottom: 1px dashed #bbf7d0;">
                            <span style="font-weight: 600;">{r_m['Medio']}</span>
                            <span style="color: #15803d; font-weight: 800;">${inv_opt_c:,.0f} ({pct_opt_c:.1f}%)</span>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
