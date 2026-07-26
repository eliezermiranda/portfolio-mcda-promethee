# -*- coding: utf-8 -*-
"""
Do Plano à Carteira — Seleção de Portfólio de Projetos com MCDA (PROMETHEE II + V)
App Streamlit para apoiar a decisão de portfólio de investimentos alinhada
ao planejamento estratégico empresarial.

Versão 2 — implementa os SEIS tipos de funções de preferência generalizadas
de Brans, Vincke e Mareschal (1986):
  Tipo 1 — usual        : P = 1 se d > 0                        (sem parâmetros)
  Tipo 2 — u-shape      : P = 1 se d > q                        (parâmetro q)
  Tipo 3 — v-shape      : P = d/p em (0, p], 1 acima            (parâmetro p)
  Tipo 4 — level        : P = 1/2 em (q, p], 1 acima            (parâmetros q, p)
  Tipo 5 — linear       : P = (d-q)/(p-q) em (q, p], 1 acima    (parâmetros q, p)
                          (v-shape com indiferença)
  Tipo 6 — gaussiana    : P = 1 - exp(-d²/(2s²)) se d > 0       (parâmetro s)

Como funciona o app:
1) PROMETHEE II ordena os projetos (fluxo líquido de preferência, phi)
2) PROMETHEE V seleciona a carteira que maximiza o phi total sob restrição
   de orçamento (problema da mochila 0-1), respeitando projetos obrigatórios.
"""

import io
import math
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuração da página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Do Plano à Carteira — MCDA PROMETHEE V",
    page_icon="📊",
    layout="wide",
)

FUNCOES_VALIDAS = ["usual", "u-shape", "v-shape", "level", "linear", "gaussiana"]

DESCRICAO_FUNCOES = {
    "usual":     "Tipo 1 — Usual: qualquer diferença positiva gera preferência total (P = 1). Sem parâmetros. Indicada para escalas discretas (ex.: notas 1–5) em que qualquer diferença conta.",
    "u-shape":   "Tipo 2 — U-shape (quase-critério): diferenças até q são indiferentes (P = 0); acima de q, preferência total (P = 1). Parâmetro: q.",
    "v-shape":   "Tipo 3 — V-shape: a preferência cresce linearmente de 0 até p (P = d/p) e é total acima de p. Parâmetro: p. Sem faixa de indiferença.",
    "level":     "Tipo 4 — Level (patamar): P = 0 até q; P = 1/2 entre q e p; P = 1 acima de p. Parâmetros: q e p. Útil para julgamentos em degraus (indiferente / preferência fraca / preferência forte).",
    "linear":    "Tipo 5 — Linear (V-shape com indiferença): P = 0 até q; cresce linearmente (P = (d−q)/(p−q)) entre q e p; P = 1 acima de p. Parâmetros: q e p. A mais usada para critérios contínuos (ex.: VPL).",
    "gaussiana": "Tipo 6 — Gaussiana: P = 1 − exp(−d²/2s²) para d > 0 — crescimento suave, sem descontinuidades. Parâmetro: s (ponto de inflexão, informado na coluna s_gaussiana).",
}

# ----------------------------------------------------------------------------
# Dados de exemplo (usados se o usuário não subir planilha)
# ----------------------------------------------------------------------------
def dados_exemplo():
    criterios = pd.DataFrame({
        "Criterio": ["VPL (R$ mil)", "Alinhamento Estratégico (1-5)",
                     "Risco de Execução (1-5)", "Impacto ESG (1-5)"],
        "Peso": [0.35, 0.30, 0.20, 0.15],
        "Objetivo": ["max", "max", "min", "max"],
        "FuncaoPreferencia": ["linear", "linear", "usual", "linear"],
        "q_indiferenca": [50.0, 0.0, 0.0, 0.0],
        "p_preferencia": [500.0, 2.0, 0.0, 2.0],
        "s_gaussiana": [0.0, 0.0, 0.0, 0.0],
    })
    projetos = pd.DataFrame({
        "Projeto": ["ERP Cloud", "Nova Linha de Produção", "Expansão Nordeste",
                    "Programa ESG", "Automação Logística", "CRM Comercial",
                    "Eficiência Energética", "Data Analytics"],
        "Custo": [1200, 3500, 2800, 800, 1500, 600, 900, 700],
        "Obrigatorio": ["não", "não", "não", "sim", "não", "não", "não", "não"],
        "VPL (R$ mil)": [900, 2200, 1800, 150, 1100, 450, 600, 500],
        "Alinhamento Estratégico (1-5)": [4, 5, 4, 3, 3, 2, 3, 4],
        "Risco de Execução (1-5)": [3, 4, 4, 1, 2, 1, 2, 2],
        "Impacto ESG (1-5)": [2, 2, 3, 5, 3, 1, 5, 2],
    })
    return criterios, projetos


# ----------------------------------------------------------------------------
# Núcleo PROMETHEE — as seis funções de preferência generalizadas
# (Brans, Vincke & Mareschal, 1986)
# ----------------------------------------------------------------------------
def normalizar_tipo(tipo):
    """Aceita sinônimos/variações de escrita e devolve o nome canônico."""
    t = str(tipo).strip().lower().replace("_", "-").replace(" ", "-")
    mapa = {
        "usual": "usual", "tipo1": "usual", "type1": "usual", "1": "usual",
        "u-shape": "u-shape", "ushape": "u-shape", "quase-criterio": "u-shape",
        "quase-critério": "u-shape", "tipo2": "u-shape", "type2": "u-shape", "2": "u-shape",
        "v-shape": "v-shape", "vshape": "v-shape", "linear-simples": "v-shape",
        "tipo3": "v-shape", "type3": "v-shape", "3": "v-shape",
        "level": "level", "patamar": "level", "niveis": "level", "níveis": "level",
        "tipo4": "level", "type4": "level", "4": "level",
        "linear": "linear", "v-shape-indiferenca": "linear", "v-shape-indiferença": "linear",
        "linear-com-indiferenca": "linear", "tipo5": "linear", "type5": "linear", "5": "linear",
        "gaussiana": "gaussiana", "gaussian": "gaussiana", "gauss": "gaussiana",
        "tipo6": "gaussiana", "type6": "gaussiana", "6": "gaussiana",
    }
    return mapa.get(t, "usual")


def funcao_preferencia(d, tipo, q=0.0, p=0.0, s=0.0):
    """
    Grau de preferência P(d) para uma diferença de desempenho d.
    Implementa os 6 critérios generalizados de Brans, Vincke & Mareschal (1986).
    Parâmetros ausentes/zerados degradam graciosamente para o tipo mais simples.
    """
    if d <= 0:
        return 0.0
    tipo = normalizar_tipo(tipo)
    q = max(float(q or 0.0), 0.0)
    p = max(float(p or 0.0), 0.0)
    s = max(float(s or 0.0), 0.0)

    if tipo == "usual":                       # Tipo 1
        return 1.0

    if tipo == "u-shape":                     # Tipo 2
        return 0.0 if d <= q else 1.0

    if tipo == "v-shape":                     # Tipo 3
        if p <= 0:
            return 1.0                        # sem p, degrada para usual
        return min(d / p, 1.0)

    if tipo == "level":                       # Tipo 4
        if p <= q:                            # parâmetros inconsistentes
            return (0.0 if d <= q else 1.0) if q > 0 else 1.0
        if d <= q:
            return 0.0
        if d <= p:
            return 0.5
        return 1.0

    if tipo == "linear":                      # Tipo 5 (v-shape com indiferença)
        if p <= 0:
            return 1.0                        # sem p, degrada para usual
        if d <= q:
            return 0.0
        if d >= p:
            return 1.0
        if p > q:
            return (d - q) / (p - q)
        return 1.0

    if tipo == "gaussiana":                   # Tipo 6
        if s <= 0:
            return 1.0                        # sem s, degrada para usual
        return 1.0 - math.exp(-(d ** 2) / (2.0 * s ** 2))

    return 1.0  # fallback defensivo


def promethee_ii(projetos, criterios, col_nome="Projeto"):
    """Calcula fluxos phi+, phi- e phi líquido (PROMETHEE II)."""
    nomes = projetos[col_nome].tolist()
    n = len(nomes)
    soma_pesos = criterios["Peso"].astype(float).sum()

    pi = np.zeros((n, n))  # índice de preferência agregado pi(a,b)
    for _, cr in criterios.reset_index(drop=True).iterrows():
        col = cr["Criterio"]
        vals = projetos[col].astype(float).values
        sinal = 1.0 if str(cr["Objetivo"]).strip().lower().startswith("max") else -1.0
        q = float(cr.get("q_indiferenca", 0) or 0)
        p = float(cr.get("p_preferencia", 0) or 0)
        s = float(cr.get("s_gaussiana", 0) or 0)
        w = float(cr["Peso"]) / soma_pesos
        tipo = cr["FuncaoPreferencia"]
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                d = sinal * (vals[a] - vals[b])
                pi[a, b] += w * funcao_preferencia(d, tipo, q, p, s)

    phi_pos = pi.sum(axis=1) / (n - 1)
    phi_neg = pi.sum(axis=0) / (n - 1)
    phi = phi_pos - phi_neg
    return pd.DataFrame({
        col_nome: nomes,
        "phi+ (força)": np.round(phi_pos, 4),
        "phi- (fraqueza)": np.round(phi_neg, 4),
        "phi líquido": np.round(phi, 4),
    }).sort_values("phi líquido", ascending=False).reset_index(drop=True)


def promethee_v(ranking, projetos, orcamento, col_nome="Projeto"):
    """
    PROMETHEE V: escolhe a carteira que maximiza a soma dos fluxos líquidos
    (phi) sob restrição de orçamento (mochila 0-1), com projetos obrigatórios.
    Busca exata: força bruta até 20 projetos livres; acima disso, programação
    dinâmica com custos discretizados.
    """
    base = ranking.merge(
        projetos[[col_nome, "Custo", "Obrigatorio"]], on=col_nome
    )
    base["Obrigatorio"] = (
        base["Obrigatorio"].astype(str).str.strip().str.lower().isin(
            ["sim", "s", "yes", "1", "true", "x"])
    )

    obrig = base[base["Obrigatorio"]]
    livres = base[~base["Obrigatorio"]].reset_index(drop=True)

    custo_obrig = obrig["Custo"].sum()
    if custo_obrig > orcamento:
        return None, custo_obrig  # orçamento insuficiente p/ obrigatórios

    saldo = orcamento - custo_obrig
    custos = livres["Custo"].astype(float).values
    phis = livres["phi líquido"].astype(float).values
    m = len(livres)

    melhor_sel, melhor_val = [], -np.inf

    if m == 0:
        melhor_sel, melhor_val = [], 0.0
    elif m <= 20:
        # força bruta exata
        idx = list(range(m))
        melhor_val = 0.0
        for r in range(1, m + 1):
            for comb in combinations(idx, r):
                cst = custos[list(comb)].sum()
                if cst <= saldo:
                    v = phis[list(comb)].sum()
                    if v > melhor_val:
                        melhor_val, melhor_sel = v, list(comb)
    else:
        # DP mochila com custos discretizados
        passo = max(1.0, saldo / 5000.0)
        cap = int(saldo / passo)
        c_int = np.maximum(1, np.ceil(custos / passo).astype(int))
        dp = np.full(cap + 1, -np.inf)
        dp[0] = 0.0
        escolha = [[False] * (cap + 1) for _ in range(m)]
        for i in range(m):
            if phis[i] <= 0:
                continue
            for w in range(cap, c_int[i] - 1, -1):
                cand = dp[w - c_int[i]] + phis[i]
                if cand > dp[w]:
                    dp[w] = cand
                    escolha[i][w] = True
        w = int(np.argmax(dp))
        melhor_val = max(dp[w], 0.0)
        for i in range(m - 1, -1, -1):
            if w >= 0 and escolha[i][w]:
                melhor_sel.append(i)
                w -= c_int[i]

    selecionados = pd.concat(
        [obrig, livres.iloc[sorted(melhor_sel)]], ignore_index=True
    ) if len(melhor_sel) or len(obrig) else obrig

    return selecionados, custo_obrig


# ----------------------------------------------------------------------------
# Geração da planilha modelo (download dentro do app)
# ----------------------------------------------------------------------------
def gerar_modelo_xlsx():
    criterios, projetos = dados_exemplo()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as wr:
        criterios.to_excel(wr, sheet_name="Criterios", index=False)
        projetos.to_excel(wr, sheet_name="Projetos", index=False)
        pd.DataFrame({"Instruções": [
            "1) Aba 'Criterios': liste os critérios de decisão (derivados do seu planejamento estratégico / BSC).",
            "   - Peso: importância relativa (será normalizado automaticamente).",
            "   - Objetivo: 'max' (quanto maior, melhor) ou 'min' (quanto menor, melhor).",
            "   - FuncaoPreferencia: um dos 6 tipos de Brans, Vincke & Mareschal (1986):",
            "       * usual      (Tipo 1) — qualquer diferença gera preferência total. Sem parâmetros.",
            "       * u-shape    (Tipo 2) — indiferente até q; preferência total acima de q. Usa: q_indiferenca.",
            "       * v-shape    (Tipo 3) — preferência cresce linearmente de 0 a p. Usa: p_preferencia.",
            "       * level      (Tipo 4) — 0 até q; 1/2 entre q e p; 1 acima de p. Usa: q_indiferenca e p_preferencia.",
            "       * linear     (Tipo 5) — 0 até q; cresce linearmente entre q e p; 1 acima de p. Usa: q_indiferenca e p_preferencia.",
            "       * gaussiana  (Tipo 6) — crescimento suave 1 - exp(-d²/2s²). Usa: s_gaussiana.",
            "   - q_indiferenca: limiar de indiferença (tipos u-shape, level e linear; 0 nos demais).",
            "   - p_preferencia: limiar de preferência total (tipos v-shape, level e linear; 0 nos demais).",
            "   - s_gaussiana: ponto de inflexão s (apenas tipo gaussiana; 0 nos demais).",
            "   - Parâmetro exigido deixado em 0 faz a função degradar para o tipo 'usual' (o app avisa).",
            "2) Aba 'Projetos': uma linha por projeto candidato.",
            "   - Custo: investimento requerido (mesma unidade do orçamento informado no app).",
            "   - Obrigatorio: 'sim' para projetos mandatórios (regulatórios, segurança) que entram antes da otimização.",
            "   - Demais colunas: desempenho do projeto em CADA critério — os nomes devem ser IDÊNTICOS aos da aba Criterios.",
            "3) Salve e faça upload no app. Informe o orçamento e navegue pelas abas de resultado.",
        ]}).to_excel(wr, sheet_name="Instrucoes", index=False)
    buf.seek(0)
    return buf


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("📊 Do Plano à Carteira")
st.caption("Seleção de portfólio de projetos alinhada ao planejamento estratégico — "
           "método multicritério **PROMETHEE II + V**, com os 6 tipos de funções de "
           "preferência de Brans, Vincke & Mareschal (1986)")

with st.sidebar:
    st.header("⚙️ Entrada de dados")
    st.download_button(
        "⬇️ Baixar planilha modelo (.xlsx)",
        data=gerar_modelo_xlsx(),
        file_name="modelo_dados_portfolio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    arquivo = st.file_uploader("Suba sua planilha preenchida", type=["xlsx"])
    usar_exemplo = st.toggle("Usar dados de exemplo", value=arquivo is None)
    st.divider()
    st.header("💰 Restrição")
    orcamento = st.number_input(
        "Orçamento total disponível", min_value=0.0, value=6000.0, step=100.0,
        help="Mesma unidade da coluna 'Custo' da planilha (ex.: R$ mil)."
    )
    st.divider()
    with st.expander("📖 As 6 funções de preferência"):
        for nome in FUNCOES_VALIDAS:
            st.markdown(f"**{nome}** — {DESCRICAO_FUNCOES[nome]}")

# Carrega dados
erro = None
if arquivo is not None and not usar_exemplo:
    try:
        criterios = pd.read_excel(arquivo, sheet_name="Criterios")
        projetos = pd.read_excel(arquivo, sheet_name="Projetos")
    except Exception as e:
        erro = f"Não consegui ler a planilha: {e}"
        criterios, projetos = dados_exemplo()
else:
    criterios, projetos = dados_exemplo()

if erro:
    st.error(erro + " — usando dados de exemplo.")

# Limpeza defensiva: remove linhas vazias ou de totalização
projetos = projetos.dropna(subset=["Projeto"])
projetos = projetos[~projetos["Projeto"].astype(str).str.upper().str.startswith("TOTAL")]
criterios = criterios.dropna(subset=["Criterio"]).reset_index(drop=True)
projetos = projetos.reset_index(drop=True)

# Compatibilidade: garante as colunas de parâmetros (planilhas da versão anterior)
for col_param in ["q_indiferenca", "p_preferencia", "s_gaussiana"]:
    if col_param not in criterios.columns:
        criterios[col_param] = 0.0
if "Obrigatorio" not in projetos.columns:
    projetos["Obrigatorio"] = "não"

# Validações básicas
faltando = [x for x in criterios["Criterio"] if x not in projetos.columns]
if faltando:
    st.error(f"Colunas de critério ausentes na aba Projetos: {faltando}. "
             "Os nomes devem ser idênticos aos da aba Criterios.")
    st.stop()

# Avisos de parametrização das funções de preferência
avisos = []
for _, cr in criterios.iterrows():
    tipo = normalizar_tipo(cr["FuncaoPreferencia"])
    q = float(cr.get("q_indiferenca", 0) or 0)
    p = float(cr.get("p_preferencia", 0) or 0)
    s = float(cr.get("s_gaussiana", 0) or 0)
    nome = cr["Criterio"]
    if tipo in ("v-shape", "linear") and p <= 0:
        avisos.append(f"'{nome}': função '{tipo}' sem p_preferencia > 0 — tratada como 'usual'.")
    if tipo == "level" and p <= q:
        avisos.append(f"'{nome}': função 'level' exige p_preferencia > q_indiferenca — tratada como degrau simples.")
    if tipo == "linear" and 0 < p <= q:
        avisos.append(f"'{nome}': função 'linear' com p ≤ q — tratada como 'u-shape'.")
    if tipo == "gaussiana" and s <= 0:
        avisos.append(f"'{nome}': função 'gaussiana' sem s_gaussiana > 0 — tratada como 'usual'.")
for a in avisos:
    st.warning(a)

tab1, tab2, tab3 = st.tabs(["1️⃣ Dados & Pesos", "2️⃣ Ranking (PROMETHEE II)",
                            "3️⃣ Carteira Ótima (PROMETHEE V)"])

with tab1:
    st.subheader("Critérios de decisão (edite pesos, funções e parâmetros)")
    st.caption("FuncaoPreferencia aceita: " + ", ".join(FUNCOES_VALIDAS) +
               ". Parâmetros: q_indiferenca (u-shape, level, linear), "
               "p_preferencia (v-shape, level, linear), s_gaussiana (gaussiana).")
    criterios = st.data_editor(
        criterios, num_rows="fixed", use_container_width=True, key="edit_criterios",
        column_config={
            "FuncaoPreferencia": st.column_config.SelectboxColumn(
                "FuncaoPreferencia", options=FUNCOES_VALIDAS, required=True,
                help="Um dos 6 tipos de Brans, Vincke & Mareschal (1986)."
            ),
        },
    )
    soma = criterios["Peso"].astype(float).sum()
    st.info(f"Soma dos pesos = **{soma:.2f}** (será normalizada para 1,00 no cálculo).")
    st.subheader("Projetos candidatos")
    projetos = st.data_editor(projetos, num_rows="dynamic", use_container_width=True,
                              key="edit_projetos")

# Cálculos
ranking = promethee_ii(projetos, criterios)

with tab2:
    st.subheader("Ranking dos projetos — fluxo líquido de preferência (φ)")
    st.caption("φ = 'placar' de cada projeto nas comparações par a par, ponderado pelos "
               "pesos dos critérios e pelas funções de preferência. φ > 0: mais forças que fraquezas.")
    st.dataframe(ranking, use_container_width=True, hide_index=True)
    graf = ranking.set_index("Projeto")[["phi líquido"]]
    st.bar_chart(graf)

with tab3:
    st.subheader("Carteira que maximiza a preferência total sob o orçamento")
    carteira, custo_obrig = promethee_v(ranking, projetos, orcamento)
    if carteira is None:
        st.error(f"Os projetos obrigatórios já custam {custo_obrig:,.0f}, acima do "
                 f"orçamento de {orcamento:,.0f}. Aumente o orçamento ou revise os obrigatórios.")
    else:
        custo_total = carteira["Custo"].sum()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Projetos selecionados", f"{len(carteira)} de {len(projetos)}")
        col2.metric("Custo da carteira", f"{custo_total:,.0f}")
        col3.metric("Orçamento utilizado", f"{(custo_total / orcamento * 100 if orcamento else 0):.1f}%")
        col4.metric("Σ φ da carteira", f"{carteira['phi líquido'].sum():.3f}")

        st.dataframe(
            carteira[["Projeto", "Custo", "Obrigatorio", "phi líquido"]]
            .sort_values("phi líquido", ascending=False),
            use_container_width=True, hide_index=True,
        )

        fora = ranking[~ranking["Projeto"].isin(carteira["Projeto"])]
        if len(fora):
            with st.expander("Projetos fora da carteira (e por quê)"):
                fora_m = fora.merge(projetos[["Projeto", "Custo"]], on="Projeto")
                fora_m["Motivo provável"] = np.where(
                    fora_m["phi líquido"] <= 0,
                    "φ ≤ 0: mais fraquezas que forças",
                    "Não coube no orçamento com maior Σφ",
                )
                st.dataframe(fora_m, use_container_width=True, hide_index=True)

        # Exporta resultado
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as wr:
            ranking.to_excel(wr, sheet_name="Ranking_PROMETHEE_II", index=False)
            carteira.to_excel(wr, sheet_name="Carteira_PROMETHEE_V", index=False)
            criterios.to_excel(wr, sheet_name="Criterios_Utilizados", index=False)
        buf.seek(0)
        st.download_button("⬇️ Exportar resultados (.xlsx)", data=buf,
                           file_name="resultado_portfolio_mcda.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.caption("Método: Brans & Vincke (1985); Brans, Vincke & Mareschal (1986) — PROMETHEE II/V, "
           "com os 6 critérios generalizados. Ferramenta de apoio à decisão — os pesos, funções "
           "e julgamentos são responsabilidade dos decisores. "
           "Artigo tecnológico associado: 'Do Plano à Carteira' (FUCAPE Business School).")
