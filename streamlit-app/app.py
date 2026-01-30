import streamlit as st
from gradio_client import Client
import html

st.set_page_config(
    page_title="Text-to-SQL Demo",
    page_icon="⚡",
    layout="wide",
)

# ── Session state ───────────────────────────────────────────────────
if "pending_q" in st.session_state:
    st.session_state.q_input = st.session_state.pop("pending_q")
    st.session_state.s_input = st.session_state.pop("pending_s")

# ── Examples from WikiSQL test set ──────────────────────────────────
EXAMPLES = [
    # Sports
    {"q": "What is terrence ross' nationality", "s": "players: Player, No., Nationality, Position, Years in Toronto, School/Club Team", "cat": "Sports"},
    {"q": "how many schools or teams had jalen rose", "s": "players: Player, No., Nationality, Position, Years in Toronto, School/Club Team", "cat": "Sports"},
    {"q": "Which winning team beat the New York Yankees?", "s": "events: Year, Game or event, Date contested, League or governing body, Sport, Winning team, Losing team, Final score", "cat": "Sports"},
    {"q": "Which podiums did the Williams team have with a margin of defeat of 2?", "s": "f1: Season, Driver, Team, Engine, Poles, Wins, Podiums, Points, Margin of defeat", "cat": "Sports"},
    {"q": "Who got high assists for the game played on February 9?", "s": "games: Game, Date, Team, Score, High points, High rebounds, High assists, Location Attendance, Record", "cat": "Sports"},
    {"q": "What college's team is the Saskatchewan Roughriders?", "s": "draft: Pick #, CFL Team, Player, Position, College", "cat": "Sports"},
    # Racing
    {"q": "Where was Assen held?", "s": "races: No, Date, Round, Circuit, Pole Position, Fastest Lap, Race winner, Report", "cat": "Racing"},
    {"q": "What was the number of race that Kevin Curtain won?", "s": "races: No, Date, Round, Circuit, Pole Position, Fastest Lap, Race winner, Report", "cat": "Racing"},
    {"q": "What was the date of the race in Misano?", "s": "races: No, Date, Round, Circuit, Pole Position, Fastest Lap, Race winner, Report", "cat": "Racing"},
    # Draft / Military
    {"q": "How many different positions did Sherbrooke Faucons (qmjhl) provide in the draft?", "s": "draft: Pick, Player, Position, Nationality, NHL team, College/junior/club team", "cat": "Draft"},
    {"q": "What are the nationalities of the player picked from Thunder Bay Flyers (ushl)", "s": "draft: Pick, Player, Position, Nationality, NHL team, College/junior/club team", "cat": "Draft"},
    {"q": "How many different nationalities do the players of New Jersey Devils come from?", "s": "draft: Pick, Player, Position, Nationality, NHL team, College/junior/club team", "cat": "Draft"},
    {"q": "What could a spanish coronel be addressed as in the commonwealth military?", "s": "ranks: Equivalent NATO Rank Code, Rank in Spanish, Rank in English, Commonwealth equivalent, US Air Force equivalent", "cat": "Military"},
    # Geography / Demographics
    {"q": "What religious groups made up 0.72% of the Indian population in 2001?", "s": "religion: Religious group, Population % 1961, Population % 1971, Population % 1981, Population % 1991, Population % 2001", "cat": "Demographics"},
    {"q": "Which city has a capacity of 41903?", "s": "stadiums: Stadium, Capacity, City, Country, Tenant, Opening", "cat": "Geography"},
    {"q": "What was the percent of slovenes 1951 for bach?", "s": "villages: Village (German), Village (Slovene), Number of people 1991, Percent of Slovenes 1991, Percent of Slovenes 1951", "cat": "Demographics"},
    {"q": "What is the state of Ted Stevens?", "s": "congress: Total tenure rank, Uninterrupted rank, Name, State represented, Dates of service, Total tenure time, Uninterrupted time", "cat": "Politics"},
    # Science / Aviation
    {"q": "What is the smallest possible radius?", "s": "stars: Star (Pismis24-#), Spectral type, Magnitude, Temperature (K), Radius, Mass", "cat": "Science"},
    {"q": "What percentage of seats were filled in 2006?", "s": "airline: Year, Aircraft kilometers, Departures, Flying hours, Passengers, Seat factor, Employees, Profit/loss", "cat": "Aviation"},
    {"q": "When were the ships launched that were laid down on september 1, 1964?", "s": "submarines: #, Shipyard, Laid down, Launched, Commissioned, Fleet, Status", "cat": "Military"},
    # Entertainment / Media
    {"q": "Which wrestlers have had 2 reigns?", "s": "wrestling: Rank, Wrestler, # of reigns, Combined defenses, Combined days", "cat": "Entertainment"},
    {"q": "What's the original air date with title \"Hell\"", "s": "episodes: No. in series, No. in season, Title, Directed by, Written by, Original air date", "cat": "Entertainment"},
    {"q": "What is the original air date for episode 15 of season 6?", "s": "episodes: No. in series, No. in season, Title, Directed by, Written by, Original air date, U.S. viewers (millions)", "cat": "Entertainment"},
    # Politics / Elections
    {"q": "Which province is grey and bell electorate in", "s": "parliament: Member, Electorate, Province, MPs term, Election date", "cat": "Politics"},
    {"q": "If % lunsford is 51.82% what is the % mcconnell in Letcher?", "s": "election: County, Precincts, Lunsford, % Lunsford, McConnell, % McConnell, Total", "cat": "Politics"},
    {"q": "How many countries got 796.7 points?", "s": "rankings: Rank, Member Association, Points, Group stage, Play-off, AFC Cup", "cat": "Sports"},
    # Agriculture / Economics
    {"q": "What's the minimum total with crop being s lupin", "s": "crops: Crop (kilotonnes), New South Wales, Victoria, Queensland, Western Australia, South Australia, Tasmania, Total", "cat": "Agriculture"},
    {"q": "How many significant relationships list Will as a virtue?", "s": "psychology: Approximate Age, Virtues, Psycho Social Crisis, Significant Relationship, Existential Question, Examples", "cat": "Science"},
    {"q": "What's the standard of the country who won its first title in 1992?", "s": "championships: Country, Players, Standard, Minor, First title, Last title", "cat": "Sports"},
    {"q": "What is the regulated retail price for the tariff code ff0 prs?", "s": "tariffs: Scheme, Tariff code, BTs retail price (regulated), Approx premium, Prefixes", "cat": "Economics"},
]

CATEGORIES = sorted(set(e["cat"] for e in EXAMPLES))
CAT_COLORS = {
    "Sports": ("#1e3a5f", "#7dd3fc"),
    "Racing": ("#3b1764", "#c4b5fd"),
    "Draft": ("#1e3a5f", "#93c5fd"),
    "Military": ("#3d1f00", "#fbbf24"),
    "Demographics": ("#064e3b", "#6ee7b7"),
    "Geography": ("#064e3b", "#34d399"),
    "Politics": ("#4a1d6a", "#d8b4fe"),
    "Science": ("#0c4a6e", "#67e8f9"),
    "Aviation": ("#422006", "#fdba74"),
    "Entertainment": ("#831843", "#f9a8d4"),
    "Agriculture": ("#1a4d1a", "#86efac"),
    "Economics": ("#713f12", "#fde68a"),
}

# ── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ─── Global ─── */
    .stApp { font-family: 'Inter', sans-serif; }
    footer, .stDeployButton { display: none !important; }

    /* ─── Header ─── */
    .hero {
        text-align: center;
        padding: 2rem 0 0.5rem;
    }
    .hero-icon {
        font-size: 2.5rem;
        margin-bottom: 0.3rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.1;
    }
    .hero h1 .grad {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0.4rem;
        font-weight: 500;
    }

    /* ─── Badges ─── */
    .badges {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.8rem 0 1rem;
    }
    .b {
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .b-blue   { background: #1e3a5f; color: #93c5fd; }
    .b-purple { background: #2e1065; color: #c4b5fd; }
    .b-green  { background: #064e3b; color: #6ee7b7; }
    .b-amber  { background: #451a03; color: #fbbf24; }

    /* ─── Pipeline ─── */
    .pipeline {
        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 0.75rem 1.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .ps {
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        font-family: 'Fira Code', monospace;
    }
    .ps-q { background: #1e3a5f; color: #7dd3fc; }
    .ps-m { background: #2e1065; color: #c4b5fd; }
    .ps-s { background: #422006; color: #fbbf24; }
    .ps-o { background: #14532d; color: #86efac; }
    .arrow { color: #475569; font-size: 1.1rem; font-weight: 300; }

    /* ─── SQL Output ─── */
    .sql-result {
        background: linear-gradient(180deg, #0a0a12 0%, #0d1117 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        font-family: 'Fira Code', monospace;
        font-size: 1.05rem;
        color: #e2e8f0;
        line-height: 1.7;
        white-space: pre-wrap;
        word-break: break-word;
        position: relative;
        overflow: hidden;
    }
    .sql-result::before {
        content: 'SQL';
        position: absolute;
        top: 0.5rem;
        right: 0.75rem;
        font-size: 0.65rem;
        font-weight: 700;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.6;
    }
    .sql-placeholder {
        color: #4b5563;
        font-style: italic;
    }

    /* ─── Detail boxes ─── */
    .detail {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        margin-top: 0.6rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.82rem;
        color: #d1d5db;
        line-height: 1.5;
    }
    .detail-lbl {
        font-size: 0.65rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 0.15rem;
        margin-top: 0.6rem;
        font-family: 'Inter', sans-serif;
    }

    /* ─── Model card (sidebar) ─── */
    .model-card {
        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .model-card h4 {
        margin: 0 0 0.3rem 0;
        color: #f1f5f9;
        font-size: 0.95rem;
    }
    .model-card p {
        color: #94a3b8;
        font-size: 0.82rem;
        margin: 0;
        line-height: 1.4;
    }
    .model-card .mc-tag {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }
    .mc-v6 { background: #422006; color: #fbbf24; }
    .mc-v5 { background: #14532d; color: #86efac; }

    /* ─── Example cards ─── */
    .ex-section-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #f1f5f9;
        margin: 1.5rem 0 0.3rem;
        letter-spacing: -0.01em;
    }
    .ex-section-sub {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }
    .cat-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* ─── Footer ─── */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #1e293b;
        color: #475569;
        font-size: 0.78rem;
        margin-top: 2.5rem;
    }
    .app-footer a { color: #60a5fa; text-decoration: none; }
    .app-footer a:hover { text-decoration: underline; }

    /* ─── Divider ─── */
    .sep {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 1.5rem 0;
    }

    /* ─── Streamlit overrides ─── */
    .stButton > button {
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        font-size: 0.78rem !important;
        padding: 0.4rem 0.7rem !important;
        transition: all 0.15s ease !important;
        background: #111827 !important;
        color: #d1d5db !important;
    }
    .stButton > button:hover {
        background: #1e293b !important;
        border-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Models ──────────────────────────────────────────────────────────
MODELS = {
    "V6 Structured": {
        "space": "RealMati/t2sql-demo",
        "outputs": 4,
        "desc": "Structured tokens (SEL/AGG/CONDS) decoded into SQL. Schema required.",
        "tag_class": "mc-v6",
        "tag": "Structured Output",
        "pipeline": """
            <span class="ps ps-q">Question</span>
            <span class="arrow">&rarr;</span>
            <span class="ps ps-m">T5 Encoder-Decoder</span>
            <span class="arrow">&rarr;</span>
            <span class="ps ps-s">SEL | AGG | CONDS</span>
            <span class="arrow">&rarr;</span>
            <span class="ps ps-o">SQL</span>
        """,
    },
    "V5 Direct SQL": {
        "space": "RealMati/t2sql-v5-demo",
        "outputs": 2,
        "desc": "Generates SQL directly as free-form text. Schema optional.",
        "tag_class": "mc-v5",
        "tag": "Direct SQL",
        "pipeline": """
            <span class="ps ps-q">Question</span>
            <span class="arrow">&rarr;</span>
            <span class="ps ps-m">T5 Encoder-Decoder</span>
            <span class="arrow">&rarr;</span>
            <span class="ps ps-o">SQL Query</span>
        """,
    },
}

# ── Hero ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">⚡</div>
    <h1><span class="grad">Text-to-SQL</span></h1>
    <div class="hero-sub">Natural language to SQL — T5 fine-tuned on 80K+ WikiSQL examples</div>
</div>
<div class="badges">
    <span class="b b-blue">T5-base 220M</span>
    <span class="b b-purple">Seq2Seq</span>
    <span class="b b-green">WikiSQL</span>
    <span class="b b-amber">Beam Search</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Model")
    model_choice = st.radio("Version", list(MODELS.keys()), label_visibility="collapsed")
    cfg = MODELS[model_choice]

    st.markdown(f"""
    <div class="model-card">
        <h4>{model_choice}</h4>
        <p>{cfg["desc"]}</p>
        <span class="mc-tag {cfg["tag_class"]}">{cfg["tag"]}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Parameters")
    num_beams = st.slider("Beam Size", 1, 10, 5)
    max_length = st.slider("Max Length", 64, 512, 256, step=64)

    st.markdown("---")
    st.markdown(
        "Custom Streamlit frontend for T5 text-to-SQL models "
        "fine-tuned on WikiSQL."
    )

# ── Pipeline ────────────────────────────────────────────────────────
st.markdown(f'<div class="pipeline">{cfg["pipeline"]}</div>', unsafe_allow_html=True)

# ── Main layout ─────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    question = st.text_area(
        "Natural Language Question",
        placeholder="e.g. What is terrence ross' nationality?",
        height=90,
        key="q_input",
    )
    schema = st.text_area(
        "Database Schema",
        placeholder="table_name: col1, col2, col3, ...",
        height=90,
        key="s_input",
    )
    st.caption("Format: `table: col1, col2, col3` — column order maps to index")
    generate = st.button("⚡ Generate SQL", type="primary", use_container_width=True)

with col_out:
    if generate and question.strip():
        with st.spinner("Generating SQL..."):
            try:
                client = Client(cfg["space"])
                result = client.predict(
                    question=question,
                    schema=schema,
                    num_beams=num_beams,
                    max_length=max_length,
                    api_name="/predict",
                )
                if cfg["outputs"] == 4:
                    sql, raw, parsed, perf = result
                else:
                    sql, perf = result
                    raw = parsed = None

                st.markdown(f'<div class="detail-lbl">Generated SQL</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="sql-result">{html.escape(sql)}</div>', unsafe_allow_html=True)

                if raw is not None:
                    st.markdown(f'<div class="detail-lbl">Raw Model Output</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="detail">{html.escape(raw)}</div>', unsafe_allow_html=True)

                if parsed is not None:
                    st.markdown(f'<div class="detail-lbl">Decoded Mapping</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="detail">{html.escape(parsed)}</div>', unsafe_allow_html=True)

                if perf:
                    st.markdown(f'<div class="detail-lbl">Performance</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="detail">{html.escape(perf)}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"API error: {e}")

    elif generate:
        st.warning("Enter a question first.")
    else:
        st.markdown('<div class="detail-lbl">Generated SQL</div>', unsafe_allow_html=True)
        st.markdown('<div class="sql-result"><span class="sql-placeholder">-- Click an example below or type your own question</span></div>', unsafe_allow_html=True)


# ── Examples ────────────────────────────────────────────────────────
st.markdown('<hr class="sep">', unsafe_allow_html=True)
st.markdown('<div class="ex-section-title">Try an Example</div>', unsafe_allow_html=True)
st.markdown('<div class="ex-section-sub">Click any example to populate the inputs above — 30 queries across 12 categories from the WikiSQL test set</div>', unsafe_allow_html=True)

# Category filter
selected_cats = st.multiselect(
    "Filter by category",
    CATEGORIES,
    default=[],
    placeholder="All categories",
)

filtered = [e for e in EXAMPLES if not selected_cats or e["cat"] in selected_cats]

cols = st.columns(3)
for i, ex in enumerate(filtered):
    with cols[i % 3]:
        cat = ex["cat"]
        bg, fg = CAT_COLORS.get(cat, ("#1e293b", "#d1d5db"))
        short_q = ex["q"] if len(ex["q"]) <= 60 else ex["q"][:57] + "..."
        table = ex["s"].split(":")[0]

        st.markdown(f'<span class="cat-pill" style="background:{bg};color:{fg};">{cat}</span>', unsafe_allow_html=True)

        def _fill(q=ex["q"], s=ex["s"]):
            st.session_state.pending_q = q
            st.session_state.pending_s = s

        st.button(
            f"📋 {table}: {short_q}",
            key=f"ex_{i}",
            use_container_width=True,
            on_click=_fill,
        )


# ── Footer ──────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <a href="https://github.com/salesforce/WikiSQL" target="_blank">WikiSQL Dataset</a>
    &nbsp;&bull;&nbsp;
    T5-base &bull; Seq2Seq &bull; Beam Search
    &nbsp;&bull;&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)
