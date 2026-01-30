import streamlit as st
from gradio_client import Client

st.set_page_config(
    page_title="Text-to-SQL Demo",
    page_icon="🗃️",
    layout="wide",
)

# ── Session state defaults ──────────────────────────────────────────
# Apply pending example (set BEFORE widgets render)
if "pending_q" in st.session_state:
    st.session_state.q_input = st.session_state.pop("pending_q")
    st.session_state.s_input = st.session_state.pop("pending_s")

# ── Custom CSS (charcoal dark) ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@400;600;800&display=swap');

    /* header */
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    /* badges */
    .badge-row {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .badge {
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        border: 1px solid;
    }
    .badge-blue   { background: #1e3a5f; color: #93c5fd; border-color: #2563eb30; }
    .badge-purple { background: #2e1065; color: #c4b5fd; border-color: #7c3aed30; }
    .badge-green  { background: #064e3b; color: #6ee7b7; border-color: #05966930; }
    .badge-amber  { background: #451a03; color: #fbbf24; border-color: #d9770630; }

    /* pipeline */
    .pipeline {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .pipe-step {
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        font-family: 'Fira Code', monospace;
    }
    .pipe-input  { background: #1e3a5f; color: #7dd3fc; }
    .pipe-model  { background: #2e1065; color: #c4b5fd; }
    .pipe-struct { background: #422006; color: #fbbf24; }
    .pipe-sql    { background: #14532d; color: #86efac; }
    .pipe-arrow  { color: #4b5563; font-size: 1rem; }

    /* sql result */
    .sql-block {
        background: #0a0a0f;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-family: 'Fira Code', monospace;
        font-size: 1.05rem;
        color: #e2e8f0;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* detail cards */
    .detail-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
        color: #d1d5db;
    }
    .detail-label {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }

    /* model selector card */
    .model-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .model-card h4 {
        margin: 0 0 0.3rem 0;
        color: #f3f4f6;
    }
    .model-card p {
        color: #9ca3af;
        font-size: 0.85rem;
        margin: 0;
    }

    /* footer */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #1f2937;
        color: #4b5563;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    .app-footer a { color: #60a5fa; }

    /* hide streamlit default footer */
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Text-to-SQL</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Natural language to SQL — powered by T5 fine-tuned on WikiSQL</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-row">
    <span class="badge badge-blue">T5-base (220M)</span>
    <span class="badge badge-purple">Seq2Seq</span>
    <span class="badge badge-green">WikiSQL 80K+</span>
    <span class="badge badge-amber">Structured Output</span>
</div>
""", unsafe_allow_html=True)


# ── Model selector ──────────────────────────────────────────────────
MODELS = {
    "V6 Structured (RealMati/t2sql-demo)": {
        "space": "RealMati/t2sql-demo",
        "outputs": 4,
        "description": "Outputs structured tokens (SEL/AGG/CONDS) decoded into SQL. Schema required.",
        "pipeline_html": """
            <span class="pipe-step pipe-input">Question</span>
            <span class="pipe-arrow">&rarr;</span>
            <span class="pipe-step pipe-model">T5 Encoder-Decoder</span>
            <span class="pipe-arrow">&rarr;</span>
            <span class="pipe-step pipe-struct">SEL | AGG | CONDS</span>
            <span class="pipe-arrow">&rarr;</span>
            <span class="pipe-step pipe-sql">SQL</span>
        """,
    },
    "V5 Direct SQL (RealMati/t2sql-v5-demo)": {
        "space": "RealMati/t2sql-v5-demo",
        "outputs": 2,
        "description": "Generates SQL directly as free-form text. Schema optional.",
        "pipeline_html": """
            <span class="pipe-step pipe-input">Question</span>
            <span class="pipe-arrow">&rarr;</span>
            <span class="pipe-step pipe-model">T5 Encoder-Decoder</span>
            <span class="pipe-arrow">&rarr;</span>
            <span class="pipe-step pipe-sql">SQL Query</span>
        """,
    },
}

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model")
    model_choice = st.radio(
        "Select model version",
        list(MODELS.keys()),
        label_visibility="collapsed",
    )
    model_cfg = MODELS[model_choice]

    st.markdown(f"""
    <div class="model-card">
        <h4>{model_choice.split(" (")[0]}</h4>
        <p>{model_cfg["description"]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Parameters")
    num_beams = st.slider("Beam Size", 1, 10, 5)
    max_length = st.slider("Max Length", 64, 512, 256, step=64)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Custom Streamlit frontend for T5 text-to-SQL models "
        "fine-tuned on WikiSQL."
    )


# ── Pipeline strip ──────────────────────────────────────────────────
st.markdown(f'<div class="pipeline">{model_cfg["pipeline_html"]}</div>', unsafe_allow_html=True)


# ── Input area ──────────────────────────────────────────────────────
col_in, col_out = st.columns(2, gap="large")

with col_in:
    question = st.text_area(
        "Natural Language Question",
        placeholder="e.g. What is terrence ross' nationality?",
        height=80,
        key="q_input",
    )
    schema = st.text_area(
        "Database Schema",
        placeholder="table_name: col1, col2, col3, ...",
        height=80,
        key="s_input",
    )
    st.caption("Format: `table: col1, col2, col3` — column order = index mapping")

    generate = st.button("Generate SQL", type="primary", use_container_width=True)


# ── Inference ───────────────────────────────────────────────────────
with col_out:
    if generate:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Generating SQL..."):
                try:
                    client = Client(model_cfg["space"])
                    result = client.predict(
                        question=question,
                        schema=schema,
                        num_beams=num_beams,
                        max_length=max_length,
                        api_name="/predict",
                    )

                    if model_cfg["outputs"] == 4:
                        sql, raw, parsed, perf = result
                    else:
                        sql, perf = result
                        raw = None
                        parsed = None

                    st.markdown('<div class="detail-label">Generated SQL</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="sql-block">{sql}</div>', unsafe_allow_html=True)

                    if raw is not None:
                        st.markdown('<div class="detail-label">Raw Structured Tokens</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="detail-card">{raw}</div>', unsafe_allow_html=True)

                    if parsed is not None:
                        st.markdown('<div class="detail-label">Decoded Mapping</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="detail-card">{parsed}</div>', unsafe_allow_html=True)

                    if perf:
                        st.markdown('<div class="detail-label">Performance</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="detail-card">{perf}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"API call failed: {e}")
    else:
        st.markdown('<div class="detail-label">Generated SQL</div>', unsafe_allow_html=True)
        st.markdown('<div class="sql-block">-- Enter a question, then click Generate SQL</div>', unsafe_allow_html=True)


# ── Examples (clickable — fills inputs) ─────────────────────────────
st.markdown("---")
st.markdown("#### Examples — click to fill")

examples = [
    ("What is terrence ross' nationality", "players: Player, No., Nationality, Position, Years in Toronto, School/Club Team"),
    ("how many schools or teams had jalen rose", "players: Player, No., Nationality, Position, Years in Toronto, School/Club Team"),
    ("What was the date of the race in Misano?", "races: No, Date, Round, Circuit, Pole Position, Fastest Lap, Race winner, Report"),
    ("What was the number of race that Kevin Curtain won?", "races: No, Date, Round, Circuit, Pole Position, Fastest Lap, Race winner, Report"),
    ("How many different positions did Sherbrooke Faucons (qmjhl) provide in the draft?", "draft: Pick, Player, Position, Nationality, NHL team, College/junior/club team"),
    ("What are the nationalities of the player picked from Thunder Bay Flyers (ushl)", "draft: Pick, Player, Position, Nationality, NHL team, College/junior/club team"),
]

cols = st.columns(3)
for i, (q, s) in enumerate(examples):
    with cols[i % 3]:
        table = s.split(":")[0]
        short_q = q if len(q) <= 50 else q[:47] + "..."
        def _fill(q=q, s=s):
            st.session_state.pending_q = q
            st.session_state.pending_s = s
        st.button(f"{table}: {short_q}", key=f"ex_{i}", use_container_width=True, on_click=_fill)


# ── Footer ──────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <a href="https://github.com/salesforce/WikiSQL" target="_blank">WikiSQL</a>
    &nbsp;&bull;&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)
