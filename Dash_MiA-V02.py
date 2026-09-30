# .\venv\Scripts\Activate.ps1
# streamlit run Dash_MiA-V02.py

import io
import os
import re
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# CONFIGURACIÓN Y ESTILOS CORPORATIVOS WPP MEDIA
# ==============================================================================
st.set_page_config(
    page_title="WPP Media - Media Impact Analysis (MiA)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta WPP Oficial
COLOR_NAVY = "#000050"       # Primario: WPP Navy
COLOR_LIME = "#B0F467"       # Primario: Lime Green
COLOR_CYAN = "#93DFE3"       # Primario: WPP Pantone 629
COLOR_CORNFLOWER = "#5465FF" # Secundario: Cornflower Blue
COLOR_PERIWINKLE = "#788BFF" # Secundario: Periwinkle
COLOR_TEAL = "#00DBEE"       # Secundario: Teal
COLOR_YELLOW = "#FCFE67"     # Secundario: Yellow

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
        min-height: 145px;
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
    
    /* Pestañas estilizadas */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.95rem;
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
# SIDEBAR: FUENTES DE DATOS (INPUTS + MODEL RESULTS)
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

st.sidebar.subheader("📂 2. Fuentes Resultados MiA")
sel_contrib = st.sidebar.selectbox("Model Contribution:", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["modelcontribution", "model_contribution", "contribution"], ["contrib"], opciones_archivos)))
sel_rev = st.sidebar.selectbox("Media Revenue:", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["mediarevenue", "media_revenue"], ["revenue"], opciones_archivos)))
sel_curv = st.sidebar.selectbox("Curvas Sat. (Opcional):", opciones_archivos, index=opciones_archivos.index(auto_detectar_archivo(["curvas", "curves"], ["curva", "saturation"], opciones_archivos)))

def resolver_fuente(sel):
    if sel == "(No asignado)": return None
    return files_dict.get(sel, sel)

# Cargar DataFrames
spend_df = cargar_df(resolver_fuente(sel_spend))
imp_df = cargar_df(resolver_fuente(sel_imp))
sales_df = cargar_df(resolver_fuente(sel_sales))
brand_df = cargar_df(resolver_fuente(sel_brand))
ext_df = cargar_df(resolver_fuente(sel_ext))
ctrl_df = cargar_df(resolver_fuente(sel_ctrl))
contrib_df = cargar_df(resolver_fuente(sel_contrib))
rev_df = cargar_df(resolver_fuente(sel_rev))
curv_df = cargar_df(resolver_fuente(sel_curv))

# Normalizar columna Date
for df in [spend_df, imp_df, sales_df, brand_df, ext_df, ctrl_df, contrib_df, rev_df]:
    if df is not None:
        for c in ['DateWeekClose', 'date', 'fecha', 'Fecha', 'Period', 'date_week_close']:
            if c in df.columns:
                df.rename(columns={c: 'Date'}, inplace=True)
                break

if sales_df is None or spend_df is None:
    st.warning("⚠️ Carga al menos los archivos de **Sales** y **Media Spend** para iniciar el análisis.")
    st.stop()

# Introspección de canales y variables
canales_detectados = extraer_variables(spend_df)
canales_meta = {ch: {'name': humanizar(ch), 'color': color_idx(i)} for i, ch in enumerate(canales_detectados)}

vars_sales = extraer_variables(sales_df)
vars_brand = extraer_variables(brand_df)
vars_ext = extraer_variables(ext_df)
vars_ctrl = extraer_variables(ctrl_df)

# Filtros dinámicos globales
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

# Helper universal para variaciones YoY
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
# TABS DE NAVEGACIÓN
# ==============================================================================
tab_results, tab_input = st.tabs([
    "🏆 Resultados MiA (Contribución y ROI)", 
    "📁 Información Input (Data de Entrada)"
])

# ##############################################################################
# TAB 1: RESULTADOS MIA (CONTRIBUCIÓN Y ROI)
# ##############################################################################
with tab_results:
    if contrib_df is None or rev_df is None:
        st.warning("⚠️ Para visualizar los **Resultados MiA**, asigna los archivos de **Model Contribution** y **Media Revenue** en la barra lateral.")
    else:
        # Variable de volumen principal de ventas
        var_vol = vars_sales[0] if vars_sales else 'Litros'
        total_sales_vol = sales_f[var_vol].sum() if var_vol in sales_f.columns else 0.0

        # Medios activos coincidentes con el archivo de contribución
        medios_contrib = [m for m in selected_media_ids if m in contrib_f.columns]
        medios_rev = [m for m in selected_media_ids if m in rev_f.columns]
        medios_spend = [m for m in selected_media_ids if m in spend_f.columns]

        curr_contrib_vol = contrib_f[medios_contrib].sum().sum() if medios_contrib else 0.0
        curr_spend_res = spend_f[medios_spend].sum().sum() if medios_spend else 0.0
        curr_rev_res = rev_f[medios_rev].sum().sum() if medios_rev else 0.0

        pct_media_contrib = (curr_contrib_vol / total_sales_vol) if total_sales_vol > 0 else 0.0
        total_roi_media = (curr_rev_res / curr_spend_res) if curr_spend_res > 0 else 0.0

        # Detectar columnas de Brand Health en contribución
        cols_no_brand = set(canales_detectados + [var_vol, 'Baseline', 'CY', 'Date', 'Litros Totales', 'TotalMedia'])
        cols_brand_contrib = [c for c in contrib_f.columns if c not in cols_no_brand and np.issubdtype(contrib_f[c].dtype, np.number) and 'bardahl' not in c.lower()]
        curr_brand_contrib_vol = contrib_f[cols_brand_contrib].sum().sum() if cols_brand_contrib else 0.0
        pct_brand_contrib = (curr_brand_contrib_vol / total_sales_vol) if total_sales_vol > 0 else 0.0

        # Deltas año anterior para Resultados
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

        # ----------------------------------------------------------------------
        # BANNER TOP KPIS RESULTADOS
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # 1. MODEL CONSTRUCTION (ÁREAS APILADAS)
        # ----------------------------------------------------------------------
        st.markdown(f"""
            <div class="section-card">
                <span class="section-badge">📈 1. Descomposición de Ventas en el Tiempo</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Model Construction Area Chart</h3>
                <p style="margin: 0.15rem 0 0.5rem 0; font-size: 0.8rem; color: #64748b;">Contribución semanal por capa de factores en áreas apiladas vs. Ventas Reales (Línea).</p>
            </div>
        """, unsafe_allow_html=True)

        date_col_c = 'Date' if 'Date' in contrib_f.columns else contrib_f.columns[1]
        fig_decomp = go.Figure()

        # Baseline
        if 'Baseline' in contrib_f.columns:
            fig_decomp.add_trace(go.Scatter(
                x=contrib_f[date_col_c], y=contrib_f['Baseline'],
                name='Baseline', mode='lines', stackgroup='one',
                line=dict(width=0.5, color='#94a3b8'), fillcolor='rgba(148, 163, 184, 0.45)'
            ))

        # Brand Health
        if cols_brand_contrib:
            sum_brand = contrib_f[cols_brand_contrib].sum(axis=1)
            fig_decomp.add_trace(go.Scatter(
                x=contrib_f[date_col_c], y=sum_brand,
                name='Brand Health', mode='lines', stackgroup='one',
                line=dict(width=0.5, color=COLOR_TEAL), fillcolor='rgba(0, 219, 238, 0.45)'
            ))

        # Medios activos apilados individualmente
        for idx, m_id in enumerate(medios_contrib):
            fig_decomp.add_trace(go.Scatter(
                x=contrib_f[date_col_c], y=contrib_f[m_id],
                name=canales_meta[m_id]['name'], mode='lines', stackgroup='one',
                line=dict(width=0.5, color=canales_meta[m_id]['color']),
                fillcolor=canales_meta[m_id]['color']
            ))

        # Línea de ventas reales en primer plano
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

        # ----------------------------------------------------------------------
        # 2. WATERFALL DUE-TO CHART
        # ----------------------------------------------------------------------
        st.markdown("""
            <div class="section-card">
                <span class="section-badge">📊 2. Análisis de Crecimiento por Driver (Waterfall)</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Waterfall Due-To Chart (Interanual YoY)</h3>
                <p style="margin: 0.15rem 0 0.5rem 0; font-size: 0.8rem; color: #64748b;">Explicación paso a paso de los factores que impulsan (+) o reducen (-) el volumen entre períodos.</p>
            </div>
        """, unsafe_allow_html=True)

        if len(cys_disponibles) >= 2:
            wf_start_cy = cys_disponibles[-2]
            wf_end_cy = cys_disponibles[-1]

            s_sales = sales_df[sales_df['CY'] == wf_start_cy][var_vol].sum()
            e_sales = sales_df[sales_df['CY'] == wf_end_cy][var_vol].sum()
            net_growth = ((e_sales - s_sales) / s_sales) * 100 if s_sales > 0 else 0

            # Diferenciales por driver
            s_contrib = contrib_df[contrib_df['CY'] == wf_start_cy]
            e_contrib = contrib_df[contrib_df['CY'] == wf_end_cy]

            diff_baseline = (e_contrib['Baseline'].sum() - s_contrib['Baseline'].sum()) if 'Baseline' in contrib_df.columns else 0.0
            diff_media = e_contrib[medios_contrib].sum().sum() - s_contrib[medios_contrib].sum().sum()
            
            # Drivers específicos de marca/competencia
            wf_drivers_x = [wf_start_cy, "Baseline", "Media Execution"]
            wf_drivers_y = [s_sales, diff_baseline, diff_media]
            wf_measures = ["absolute", "relative", "relative"]

            for c in cols_brand_contrib:
                d_val = e_contrib[c].sum() - s_contrib[c].sum()
                wf_drivers_x.append(humanizar(c))
                wf_drivers_y.append(d_val)
                wf_measures.append("relative")

            # Columnas negativas / competitivas (ej. Bardahl u otros)
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
        else:
            st.info("Se requieren al menos 2 períodos (CY) para calcular el gráfico de cascada YoY.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 3 Y 4. CONTRIBUCIÓN (%) Y ROI POR MEDIO
        # ----------------------------------------------------------------------
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown(f"""
                <div class="section-card">
                    <span class="section-badge">% Contribución</span>
                    <h3 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: #000050;">Contribución de Medios (%) por Canal</h3>
                </div>
            """, unsafe_allow_html=True)
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
                barmode='group', height=330, margin=dict(l=20, r=20, t=20, b=40),
                yaxis=dict(title="% Contribución", gridcolor="#f1f5f9", ticksuffix="%"),
                xaxis=dict(tickangle=-25), plot_bgcolor='white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        with c_m2:
            st.markdown(f"""
                <div class="section-card">
                    <span class="section-badge">Multiplicador ROI</span>
                    <h3 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: #000050;">Retorno de Inversión (ROI) por Canal</h3>
                </div>
            """, unsafe_allow_html=True)
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
                barmode='group', height=330, margin=dict(l=20, r=20, t=20, b=40),
                yaxis=dict(title="ROI (Revenue / Spend)", gridcolor="#f1f5f9", ticksuffix="x"),
                xaxis=dict(tickangle=-25), plot_bgcolor='white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_roi, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 5. DOS ANILLOS: CONTRIBUTION MIX VS. SPEND MIX
        # ----------------------------------------------------------------------
        st.markdown("""
            <div class="section-card">
                <span class="section-badge">🍩 5. Mix de Contribución vs. Mix de Inversión</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Contribution Mix (% Share Medios) vs. Media Spend Mix (% Share Inversión)</h3>
            </div>
        """, unsafe_allow_html=True)

        c_don1, c_don2 = st.columns(2)
        pie_labels = [canales_meta[m]['name'] for m in selected_media_ids]
        pie_colors = [canales_meta[m]['color'] for m in selected_media_ids]
        contrib_sums = [contrib_f[m].sum() if m in contrib_f.columns else 0.0 for m in selected_media_ids]
        spend_sums = [spend_f[m].sum() if m in spend_f.columns else 0.0 for m in selected_media_ids]

        with c_don1:
            fig_d_contrib = go.Figure(data=[go.Pie(
                labels=pie_labels, values=contrib_sums, hole=0.55,
                marker=dict(colors=pie_colors), textinfo='percent',
                hovertemplate="<b>%{label}</b><br>Contribución: %{value:,.0f}<br>Share: %{percent}<extra></extra>"
            )])
            fig_d_contrib.update_layout(title="Contribution Mix (% Share - Solo Medios)", height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_d_contrib, use_container_width=True)

        with c_don2:
            fig_d_spend = go.Figure(data=[go.Pie(
                labels=pie_labels, values=spend_sums, hole=0.55,
                marker=dict(colors=pie_colors), textinfo='percent',
                hovertemplate="<b>%{label}</b><br>Inversión: $%{value:,.0f}<br>Share: %{percent}<extra></extra>"
            )])
            fig_d_spend.update_layout(title="Media Spend Mix (% Share Inversión)", height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_d_spend, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 6. MARKETING & MEDIA DYNAMICS (PANEL CIRCANA / RECKITT)
        # ----------------------------------------------------------------------
        st.markdown("""
            <div class="section-card">
                <span class="section-badge">🎯 6. Marketing & Media Dynamics</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Dynamics per Channel (Inversión, Ingresos y Multiplicador ROI)</h3>
            </div>
        """, unsafe_allow_html=True)

        labels_dyn = ["Total Media"] + pie_labels
        for met_name, suf, df_source, is_roi in [
            ("Media Spend ($ MXN)", "$", spend_df, False),
            ("Media Incremental Net Revenue ($ MXN)", "$", rev_df, False),
            ("Media Net Revenue ROI ($ Revenue / $ Spend)", "x", None, True)
        ]:
            fig_dyn = go.Figure()
            for idx, cy in enumerate(active_cys):
                vals_dyn = []
                # Total Media
                if not is_roi:
                    t_val = df_source[df_source['CY'] == cy][selected_media_ids].sum().sum()
                else:
                    sp_t = spend_df[spend_df['CY'] == cy][selected_media_ids].sum().sum()
                    rv_t = rev_df[rev_df['CY'] == cy][selected_media_ids].sum().sum()
                    t_val = (rv_t / sp_t) if sp_t > 0 else 0.0
                vals_dyn.append(t_val)

                # Por canal
                for m in selected_media_ids:
                    if not is_roi:
                        v = df_source[df_source['CY'] == cy][m].sum() if m in df_source.columns else 0.0
                    else:
                        sp_m = spend_df[spend_df['CY'] == cy][m].sum() if m in spend_df.columns else 0.0
                        rv_m = rev_df[rev_df['CY'] == cy][m].sum() if m in rev_df.columns else 0.0
                        v = (rv_m / sp_m) if sp_m > 0 else 0.0
                    vals_dyn.append(v)

                fig_dyn.add_trace(go.Bar(
                    x=labels_dyn, y=vals_dyn, name=cy, marker_color=color_idx(idx + 1)
                ))

            fig_dyn.update_layout(
                title=met_name, barmode='group', height=280, margin=dict(l=20, r=20, t=35, b=20),
                yaxis=dict(gridcolor="#f1f5f9", ticksuffix=suf), xaxis=dict(tickangle=-20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor='white'
            )
            st.plotly_chart(fig_dyn, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 7. CURVAS DE RESPUESTA S-CURVE
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # 7. OPTIMIZACIÓN DE PRESUPUESTO & CURVAS DE RESPUESTA
        # ----------------------------------------------------------------------
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

            # Gráfica 1: S-Curve
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=c_rows[col_inv], 
                y=c_rows[col_cont] * 100 if c_rows[col_cont].mean() <= 1.0 else c_rows[col_cont],
                mode='lines', name='Curva de Respuesta', line=dict(color=COLOR_NAVY, width=3)
            ))

            # Umbrales de inversión monetaria para la gráfica histórica
            umbrales_spend = {}
            for punto, col_pt, sym in [("Sustain", "#f59e0b", "diamond"), ("Optimal", "#10b981", "circle"), ("Heavy Up", "#ef4444", "square"), ("CurrentSpend", COLOR_CORNFLOWER, "star")]:
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

            # ------------------------------------------------------------------
            # GRÁFICA SOLICITADA 1: INVERSIÓN HISTÓRICA VS. NIVELES GUÍA ($)
            # ------------------------------------------------------------------
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

            # Mapeo del medio seleccionado en la curva con las columnas de spend_f
            col_spend_match = None
            for c in spend_f.columns:
                if c.lower() == sel_curv_media.lower() or sel_curv_media.lower() in c.lower() or c.lower() in sel_curv_media.lower():
                    col_spend_match = c
                    break
            if col_spend_match is None and 'OpenTv' in spend_f.columns and ('azteca' in sel_curv_media.lower() or 'televisa' in sel_curv_media.lower()):
                col_spend_match = 'OpenTv'

            if col_spend_match and col_spend_match in spend_f.columns:
                date_labels = spend_f['CY'].astype(str) + " - " + spend_f[date_col_c].astype(str)
                hist_spend_vals = spend_f[col_spend_match]

                fig_hist_bench = go.Figure()

                # 1. Línea azul de Inversión Histórica Real con marcadores
                fig_hist_bench.add_trace(go.Scatter(
                    x=date_labels,
                    y=hist_spend_vals,
                    name="Inversión Histórica Real",
                    mode="lines+markers",
                    line=dict(color="#2563eb", width=2.5),
                    marker=dict(size=6, color="#2563eb"),
                    hovertemplate="%{x}<br>Gasto Real: $%{y:,.0f}<extra></extra>"
                ))

                # 2. Líneas horizontales de umbrales guía
                if "Sustain" in umbrales_spend:
                    fig_hist_bench.add_hline(
                        y=umbrales_spend["Sustain"],
                        line_dash="dash",
                        line_color="#f59e0b",
                        line_width=1.8,
                        annotation_text="Nivel Sustain (Mínimo)",
                        annotation_position="bottom right"
                    )

                if "Optimal" in umbrales_spend:
                    fig_hist_bench.add_hline(
                        y=umbrales_spend["Optimal"],
                        line_dash="solid",
                        line_color="#10b981",
                        line_width=2,
                        annotation_text="Nivel Optimal (Óptimo)",
                        annotation_position="top right"
                    )

                if "Heavy Up" in umbrales_spend:
                    fig_hist_bench.add_hline(
                        y=umbrales_spend["Heavy Up"],
                        line_dash="dot",
                        line_color="#ef4444",
                        line_width=1.8,
                        annotation_text="Nivel Heavy Up (Máximo)",
                        annotation_position="top right"
                    )

                fig_hist_bench.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=25, b=65),
                    yaxis=dict(
                        title="Inversión ($)",
                        gridcolor="#f1f5f9",
                        tickprefix="$",
                        tickformat=",.0f"
                    ),
                    xaxis=dict(
                        tickangle=-45,
                        tickfont=dict(size=8, color="#64748b")
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.55,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=9)
                    ),
                    plot_bgcolor="white"
                )
                st.plotly_chart(fig_hist_bench, use_container_width=True)
            else:
                st.info(f"No se encontró una columna directa en Media Spend para graficar el histórico de '{sel_curv_media}'.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # GRÁFICA SOLICITADA 2: DESGLOSE DE CONTRIBUCIÓN NO-MEDIA
        # ----------------------------------------------------------------------
        st.markdown(f"""
            <div class="section-card">
                <span class="section-badge">🏛️ Desglose de Contribución No-Media</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Impacto de Factores Orgánicos, Marca y Competencia</h3>
                <p style="margin: 0.15rem 0 0.5rem 0; font-size: 0.8rem; color: #64748b;">
                    Volumen y participación (%) de todas las variables que no corresponden a inversión publicitaria directa.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Extraer todas las variables que no son canales de medios ni estructurales
        cols_todos_medios = set(canales_detectados + [var_vol, 'CY', 'Date', 'Litros Totales', 'TotalMedia', 'Total'])
        cols_non_media = [
            c for c in contrib_f.columns 
            if c not in cols_todos_medios and np.issubdtype(contrib_f[c].dtype, np.number)
        ]

        if cols_non_media:
            non_media_data = []
            for col_nm in cols_non_media:
                vol_nm = contrib_f[col_nm].sum()
                pct_nm = (vol_nm / total_sales_vol * 100) if total_sales_vol > 0 else 0.0
                non_media_data.append({
                    "Driver": humanizar(col_nm),
                    "Volumen": vol_nm,
                    "Porcentaje": pct_nm,
                    "Tipo": "Positivo (Gain)" if vol_nm >= 0 else "Negativo (Loss)"
                })

            df_nm = pd.DataFrame(non_media_data).sort_values(by="Volumen", ascending=True)

            col_nm_bar, col_nm_table = st.columns([3, 2])

            with col_nm_bar:
                # Barras horizontales divergentes
                fig_nm = go.Figure(go.Bar(
                    x=df_nm['Volumen'],
                    y=df_nm['Driver'],
                    orientation='h',
                    marker_color=[COLOR_LIME if v >= 0 else "#ef4444" for v in df_nm['Volumen']],
                    text=[f"{p:+.1f}%" for p in df_nm['Porcentaje']],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Contribución: %{x:,.0f} L<br>Share s/ Ventas: %{text}<extra></extra>"
                ))

                fig_nm.update_layout(
                    title="Contribución Absoluta por Driver No-Media",
                    height=340,
                    margin=dict(l=20, r=40, t=35, b=20),
                    xaxis=dict(title=f"Volumen ({var_vol})", gridcolor="#f1f5f9"),
                    yaxis=dict(tickfont=dict(size=10, color="#1e293b")),
                    plot_bgcolor="white"
                )
                st.plotly_chart(fig_nm, use_container_width=True)

            with col_nm_table:
                # Tabla consolidada con formato
                df_nm_display = df_nm.sort_values(by="Volumen", ascending=False).copy()
                df_nm_display['Volumen Formateado'] = df_nm_display['Volumen'].apply(lambda x: f"{x:,.0f}")
                df_nm_display['% del Total'] = df_nm_display['Porcentaje'].apply(lambda x: f"{x:+.2f}%")
                
                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                st.dataframe(
                    df_nm_display[['Driver', 'Volumen Formateado', '% del Total', 'Tipo']],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No se identificaron columnas No-Media en el archivo de contribución cargado.")

        # ----------------------------------------------------------------------
        # 8. TABLA DETALLADA CONSOLIDADA
        # ----------------------------------------------------------------------
        st.markdown("""
            <div class="section-card">
                <span class="section-badge">📋 8. Tabla Detallada</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Detalle de Contribución y Desempeño por Driver</h3>
            </div>
        """, unsafe_allow_html=True)

        tbl_data = []
        # Medios
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
                "ROI (Rev/Spend)": f"{roi:.2f}x"
            })

        # Baseline y Factores
        if 'Baseline' in contrib_f.columns:
            cb_b = contrib_f['Baseline'].sum()
            pct_b = (cb_b / total_sales_vol * 100) if total_sales_vol > 0 else 0.0
            tbl_data.append({
                "Driver / Canal": "Baseline (Orgánico)",
                "Inversión (Spend)": "-", "Ingresos (Revenue)": "-",
                f"Contribución ({var_vol})": f"{cb_b:,.0f}",
                "% Contribución": f"{pct_b:.2f}%",
                "ROI (Rev/Spend)": "-"
            })

        for c in cols_brand_contrib:
            cb_f = contrib_f[c].sum()
            pct_f = (cb_f / total_sales_vol * 100) if total_sales_vol > 0 else 0.0
            tbl_data.append({
                "Driver / Canal": humanizar(c),
                "Inversión (Spend)": "-", "Ingresos (Revenue)": "-",
                f"Contribución ({var_vol})": f"{cb_f:,.0f}",
                "% Contribución": f"{pct_f:.2f}%",
                "ROI (Rev/Spend)": "-"
            })

        st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

# ##############################################################################
# TAB 2: INFORMACIÓN INPUT (DATA DE ENTRADA)
# ##############################################################################
with tab_input:
    # Función de sincronización para evitar KeyErrors al cambiar datasets
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

    # Inversión Medios KPI
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

    # 1. Media Spend
    st.markdown("""
        <div class="section-card">
            <span class="section-badge">💰 1. Media Spend</span>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Evolución e Inversión por Medio y Período</h3>
        </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_sb = go.Figure()
        ch_l = [canales_meta[ch]['name'] for ch in selected_media_ids]
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

    # 2. Impacts
    if imp_df is not None:
        st.markdown("""
            <div class="section-card">
                <span class="section-badge">👁 2. Media Impacts</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Evolución de Impactos / Contactos por Medio y Período</h3>
            </div>
        """, unsafe_allow_html=True)
        fig_im = go.Figure()
        for idx, cy in enumerate(active_cys):
            cy_r = imp_df[imp_df['CY'] == cy]
            cy_i = [cy_r[ch].sum() if ch in cy_r.columns else 0.0 for ch in selected_media_ids]
            fig_im.add_trace(go.Bar(x=ch_l, y=cy_i, name=cy, marker_color=color_idx(idx + 1)))
        fig_im.update_layout(barmode='group', height=330, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(title="Total Impactos", gridcolor="#f1f5f9"), xaxis=dict(tickangle=-25), plot_bgcolor='white')
        st.plotly_chart(fig_im, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # 3. Sales vs Spend
    var_v_inp = st.session_state.get("kpi_sel_sales", vars_sales[0] if vars_sales else None)
    if var_v_inp not in vars_sales:
        var_v_inp = vars_sales[0] if vars_sales else None

    st.markdown(f"""
        <div class="section-card">
            <span class="section-badge">🛢️ 3. Sales ({var_v_inp})</span>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Respuesta de Ventas vs. Presión Semanal de Inversión</h3>
        </div>
    """, unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)

    # 4, 5 y 6. Multivariables
    def render_seccion_multivar(badge, tit, desc, df_d, v_disp, k_p):
        if not v_disp or df_d is None: return
        st.markdown(f"""
            <div class="section-card">
                <span class="section-badge">{badge}</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">{tit}</h3>
                <p style="margin: 0.15rem 0 0.5rem 0; font-size: 0.8rem; color: #64748b;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)
        v_s = st.multiselect(f"Seleccionar variables ({tit}):", options=v_disp, default=v_disp[:min(4, len(v_disp))], key=f"msel_{k_p}")
        if not v_s: return
        fig_m = go.Figure()
        d_col_loc = 'Date' if 'Date' in df_d.columns else df_d.columns[1]
        if var_v_inp and var_v_inp in sales_f.columns:
            fig_m.add_trace(go.Bar(x=sales_f[d_col_loc], y=sales_f[var_v_inp], name=f"Ventas ({var_v_inp})", marker_color="rgba(147, 223, 227, 0.35)", marker_line=dict(color=COLOR_CYAN, width=1), yaxis='y2'))
        est = ['solid', 'dot', 'dash', 'longdash', 'dashdot']
        for idx, v in enumerate(v_s):
            val = df_d[v]
            es_p = val.mean() <= 1.0 and val.mean() >= 0.0
            disp_v = val * 100 if es_p else val
            suf = "%" if es_p else ""
            fig_m.add_trace(go.Scatter(x=df_d[d_col_loc], y=disp_v, name=humanizar(v), mode='lines', line=dict(color=color_idx(idx + 1), width=2.2, dash=est[idx % len(est)]), yaxis='y1'))
        fig_m.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(title="Nivel de Variable", side='left', showgrid=True, gridcolor="#f1f5f9"), yaxis2=dict(title=var_v_inp, side='right', overlaying='y', showgrid=False), xaxis=dict(tickangle=-45), legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5, font=dict(size=9)), plot_bgcolor='white')
        st.plotly_chart(fig_m, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    render_seccion_multivar("👥 4. Brand Health", "Tracking de Métricas de Marca vs. Ventas", "Indicadores de embudo (Consideration, Purchase Intent, Digital Activity, Index).", brand_f, vars_brand, "brand")
    render_seccion_multivar("🌦️ 5. Factores Externos", "Tracking de Factores Externos vs. Ventas", "Variables continuas exógenas (temperatura media, lluvia, tipo de cambio).", ext_f, vars_ext, "ext")

    if vars_ctrl and ctrl_f is not None:
        st.markdown("""
            <div class="section-card">
                <span class="section-badge">🏷️ 6. Variables de Control</span>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #000050;">Tracking de Eventos Especiales y Dummies (0 a 1) vs. Ventas</h3>
            </div>
        """, unsafe_allow_html=True)
        c_sel = st.multiselect("Seleccionar variables de control:", options=vars_ctrl, default=vars_ctrl[:min(4, len(vars_ctrl))], key="msel_ctrl")
        if c_sel:
            fig_c = go.Figure()
            d_col_c = 'Date' if 'Date' in ctrl_f.columns else ctrl_f.columns[1]
            if var_v_inp and var_v_inp in sales_f.columns:
                fig_c.add_trace(go.Bar(x=sales_f[d_col_c], y=sales_f[var_v_inp], name=f"Ventas ({var_v_inp})", marker_color="rgba(147, 223, 227, 0.35)", marker_line=dict(color=COLOR_CYAN, width=1), yaxis='y2'))
            for idx, v in enumerate(c_sel):
                fig_c.add_trace(go.Scatter(x=ctrl_f[d_col_c], y=ctrl_f[v], name=humanizar(v), mode='lines', line_shape='hv', line=dict(color=color_idx(idx + 3), width=2.5), yaxis='y1'))
            fig_c.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(title="Evento Activo (0 / 1)", tickmode='linear', tick0=0, dtick=1, range=[-0.1, 1.1], side='left', showgrid=True, gridcolor="#f1f5f9"), yaxis2=dict(title=var_v_inp, side='right', overlaying='y', showgrid=False), xaxis=dict(tickangle=-45), legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5, font=dict(size=9)), plot_bgcolor='white')
            st.plotly_chart(fig_c, use_container_width=True)