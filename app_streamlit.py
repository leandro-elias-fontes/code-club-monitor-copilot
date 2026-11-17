"""Interface Streamlit limpa e nativa para o Code Club Monitor Copilot."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src.agent import get_api_keys, run_copilot
from src.demo_cache import DEMO_SCENARIOS

load_dotenv()

st.set_page_config(
    page_title="Code Club Monitor Copilot",
    page_icon="</>",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 2rem; }
    textarea { font-family: 'JetBrains Mono', Consolas, monospace !important; font-size: 0.88rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
head_left, head_right = st.columns([3, 1])
with head_left:
    st.title("Code Club Monitor Copilot")
    st.caption(
        "Diagnostico em 3s para monitores de algoritmos — RAG em YouKn0wWho, CSES, Codeforces, AtCoder e Maratona SBC"
    )
with head_right:
    keys_count = len(get_api_keys())
    if keys_count > 0:
        st.success(f"Gemini 3.8 Flash ({keys_count} chaves)")
    else:
        st.warning("Modo offline (sem chave API)")

# Horizontal quick-demo strip
st.markdown("**Exemplos prontos em cache (0 tokens):**")
demo_cols = st.columns(len(DEMO_SCENARIOS))
for idx, demo in enumerate(DEMO_SCENARIOS):
    with demo_cols[idx]:
        short_label = demo["title"].split("(", 1)[0].strip()
        if st.button(short_label, key=demo["id"], use_container_width=True):
            st.session_state["demo_id"] = demo["id"]
            st.session_state["language"] = demo["language"]
            st.session_state["problem_statement"] = demo["problem_statement"]
            st.session_state["code"] = demo["code"]
            st.session_state["auto_run"] = True

st.divider()

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    with st.container(border=True):
        c1, c2 = st.columns(2)
        lang_options = ["cpp", "c", "python"]
        default_lang = st.session_state.get("language", "cpp")
        with c1:
            language = st.selectbox(
                "Linguagem",
                options=lang_options,
                index=lang_options.index(default_lang) if default_lang in lang_options else 0,
            )
        with c2:
            source_filter = st.selectbox(
                "Fonte do RAG",
                options=[
                    "Todas as 5 fontes",
                    "YouKn0wWho",
                    "CSES",
                    "Codeforces",
                    "AtCoder",
                    "Maratona SBC",
                ],
            )

        problem_statement = st.text_area(
            "Enunciado e limites",
            value=st.session_state.get("problem_statement", ""),
            height=90,
            placeholder="Ex: Dado N <= 10^5, encontre dois elementos com soma K...",
        )
        code = st.text_area(
            "Codigo do aluno",
            value=st.session_state.get("code", ""),
            height=300,
            placeholder="Cole aqui o codigo em C, C++ ou Python...",
        )
        run_clicked = st.button("Diagnosticar Bug", type="primary", use_container_width=True)

should_run = run_clicked or st.session_state.pop("auto_run", False)

with col_out:
    if should_run and code.strip():
        demo_id_to_use = None if run_clicked else st.session_state.get("demo_id")
        filt = None if source_filter == "Todas as 5 fontes" else source_filter
        with st.spinner("Analisando codigo e consultando base RAG..."):
            result = run_copilot(
                language=language,  # type: ignore[arg-type]
                code=code,
                problem_statement=problem_statement,
                demo_id=demo_id_to_use,
                source_filter=filt,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Origem", result.source)
        m2.metric("Tokens", result.tokens_used)
        m3.metric("Latencia", f"{result.latency_ms} ms")
        m4.metric("Big-O", result.static_report.estimated_big_o)

        tab_diag, tab_rag, tab_static = st.tabs(
            ["Diagnostico (4 Blocos)", f"Referencias RAG ({len(result.rag_references)})", "Analise Estatica"]
        )

        with tab_diag:
            st.info(f"**1. Logica do aluno**\n\n{result.student_logic}")
            st.error(f"**2. Onde esta o erro**\n\n{result.bug_location}")
            with st.container(border=True):
                st.markdown("**3. Contra-exemplo minimo**")
                st.code(result.counterexample, language="text")
            st.success(f"**4. Pergunta socratica para o monitor**\n\n{result.monitor_tip}")

        with tab_rag:
            for ref in result.rag_references:
                with st.container(border=True):
                    r1, r2 = st.columns([3, 1])
                    with r1:
                        st.markdown(f"**[{ref.source}] {ref.title}**")
                        st.caption(ref.technical_summary)
                    with r2:
                        st.link_button("Abrir guia", ref.url, use_container_width=True)

        with tab_static:
            st.markdown(f"**Complexidade estimada:** `{result.static_report.estimated_big_o}`")
            if result.static_report.findings:
                for finding in result.static_report.findings:
                    st.warning(finding)
            else:
                st.success("Nenhum alerta estatico deterministico.")

    elif should_run:
        st.warning("Cole o codigo do aluno antes de clicar em Diagnosticar Bug.")
    else:
        with st.container(border=True):
            st.markdown("### Pronto para o Raio-X")
            st.write(
                "Clique em um dos 5 exemplos acima para testar com 0 tokens ou cole o codigo do aluno ao lado."
            )
