import re
import os
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time

MODEL_ID = "RealMati/t2sql_v6_structured"

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
AGG_LABELS = ["None", "MAX", "MIN", "COUNT", "SUM", "AVG"]
OPS = ["=", ">", "<", ">=", "<=", "!="]

css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path, "r") as f:
    CSS = f.read()


def decode_structured_output(text):
    sel = agg = None
    conds = []
    try:
        for part in text.strip().split(" | "):
            part = part.strip()
            if part.startswith("SEL:"):
                sel = int(part[4:].strip())
            elif part.startswith("AGG:"):
                agg = int(part[4:].strip())
            elif part.startswith("CONDS:"):
                cond_str = part[6:].strip()
                if cond_str:
                    for c in cond_str.split(";"):
                        vals = c.split(",", 2)
                        if len(vals) >= 3:
                            conds.append([int(vals[0]), int(vals[1]), vals[2]])
    except Exception:
        pass
    return sel, agg, conds


def parse_schema(schema_str):
    schema_str = schema_str.strip()
    if not schema_str:
        return "table", []
    first_table = schema_str.split("|")[0].strip()
    if ":" in first_table:
        table_name, cols_str = first_table.split(":", 1)
        table_name = table_name.strip()
        columns = [c.strip() for c in cols_str.split(",") if c.strip()]
    else:
        table_name = "table"
        columns = [c.strip() for c in first_table.split(",") if c.strip()]
    return table_name, columns


def structured_to_sql(sel, agg, conds, columns, table_name="table"):
    if sel is None or agg is None:
        return None
    col_name = columns[sel] if sel < len(columns) else f"col{sel}"
    if agg == 0:
        sql = f"SELECT {col_name} FROM {table_name}"
    else:
        agg_op = AGG_OPS[agg] if agg < len(AGG_OPS) else ""
        sql = f"SELECT {agg_op}({col_name}) FROM {table_name}"
    if conds:
        where_parts = []
        for c_idx, c_op, c_val in conds:
            c_name = columns[c_idx] if c_idx < len(columns) else f"col{c_idx}"
            op_str = OPS[c_op] if c_op < len(OPS) else "="
            try:
                float(c_val)
                val_sql = c_val
            except (ValueError, TypeError):
                val_sql = f"'{c_val}'"
            where_parts.append(f"{c_name} {op_str} {val_sql}")
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
    return sql


def format_parsed(sel, agg, conds, columns):
    parts = []
    if sel is not None and sel < len(columns):
        parts.append(f"Column: {columns[sel]} (index {sel})")
    elif sel is not None:
        parts.append(f"Column index: {sel}")
    if agg is not None:
        agg_label = AGG_LABELS[agg] if agg < len(AGG_LABELS) else str(agg)
        parts.append(f"Aggregation: {agg_label}")
    if conds:
        cond_strs = []
        for c_idx, c_op, c_val in conds:
            c_name = columns[c_idx] if c_idx < len(columns) else f"col{c_idx}"
            op_str = OPS[c_op] if c_op < len(OPS) else "="
            cond_strs.append(f"{c_name} {op_str} {c_val}")
        parts.append(f"Conditions: {' AND '.join(cond_strs)}")
    else:
        parts.append("Conditions: None")
    return "  |  ".join(parts)


def predict(question, schema, num_beams, max_length):
    if not question or not question.strip():
        return (
            "-- Enter a question and schema, then click Generate SQL",
            "Waiting for input...",
            "No query submitted yet",
            "",
        )
    table_name, columns = parse_schema(schema)
    if not columns:
        return (
            "-- Please provide a schema\n-- Format: table_name: col1, col2, col3",
            "Schema required",
            "Cannot map indices without column names",
            "",
        )

    input_text = f"translate to SQL: {question}"
    if schema.strip():
        input_text += f" | schema: {schema.strip()}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=int(max_length),
            num_beams=int(num_beams),
            early_stopping=True,
            do_sample=False,
        )
    latency = time.time() - t0

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sel, agg, conds = decode_structured_output(raw_output)

    if sel is not None and agg is not None and columns:
        sql = structured_to_sql(sel, agg, conds, columns, table_name)
    else:
        sql = f"-- Could not parse model output\n-- Raw: {raw_output}"

    parsed = format_parsed(sel, agg, conds, columns) if sel is not None else "Parse failed"
    perf = f"Inference: {latency:.2f}s  |  Beams: {int(num_beams)}  |  Tokens: {inputs['input_ids'].shape[1]}"
    return sql, raw_output, parsed, perf


theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="purple",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("Fira Code"),
).set(
    body_background_fill="#0d1117",
    body_text_color="#e2e8f0",
    block_background_fill="#161b22",
    block_border_color="#1f2937",
    block_border_width="1px",
    block_label_text_color="#d1d5db",
    block_title_text_color="#f3f4f6",
    block_radius="12px",
    block_shadow="none",
    input_background_fill="#111827",
    input_border_color="#1f2937",
    input_border_width="1px",
    input_placeholder_color="#4b5563",
    input_radius="8px",
    slider_color="#3b82f6",
    button_primary_background_fill="linear-gradient(135deg, #3b82f6, #8b5cf6)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#111827",
    button_secondary_text_color="#d1d5db",
    button_secondary_border_color="#1f2937",
    border_color_primary="#1f2937",
    color_accent_soft="#111827",
)

with gr.Blocks(title="Text-to-SQL | T5 on WikiSQL") as demo:

    # Compact header — one line title + badges + pipeline
    gr.HTML("""
    <div class="app-header">
        <h1><span>Text-to-SQL</span></h1>
    </div>
    <div class="tech-badges">
        <span class="badge badge-indigo">T5-base (220M)</span>
        <span class="badge badge-purple">Seq2Seq</span>
        <span class="badge badge-emerald">WikiSQL 80K+</span>
        <span class="badge badge-amber">Structured Output</span>
    </div>
    <div class="pipeline-strip">
        <span class="step step-input">Question</span>
        <span class="arrow">&rarr;</span>
        <span class="step step-model">T5 Encoder-Decoder</span>
        <span class="arrow">&rarr;</span>
        <span class="step step-struct">SEL | AGG | CONDS</span>
        <span class="arrow">&rarr;</span>
        <span class="step step-sql">SQL</span>
    </div>
    """)

    with gr.Tabs():

        # ══ TAB 1: INFERENCE (main focus) ══
        with gr.Tab("Demo"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    question = gr.Textbox(
                        label="Natural Language Question",
                        placeholder="e.g. What is terrence ross' nationality?",
                        lines=2,
                    )
                    schema = gr.Textbox(
                        label="Database Schema",
                        placeholder="table_name: col1, col2, col3, ...",
                        lines=2,
                    )
                    gr.HTML('<p class="input-hint">Format: <code>table: col1, col2, col3</code> — column order = index mapping</p>')
                    with gr.Row():
                        beams = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Beam Size")
                        max_len = gr.Slider(minimum=64, maximum=512, value=256, step=64, label="Max Length")
                    btn = gr.Button("Generate SQL", variant="primary", elem_classes=["generate-btn"], size="lg")

                with gr.Column(scale=1):
                    sql_out = gr.Textbox(
                        label="Generated SQL",
                        value="-- Enter a question and schema, then click Generate SQL",
                        lines=3,
                        elem_classes=["sql-output"],
                    )
                    raw_out = gr.Textbox(label="Raw Structured Tokens", value="Waiting for input...", lines=1, elem_classes=["decode-box"])
                    parsed_out = gr.Textbox(label="Decoded Mapping", value="No query submitted yet", lines=1, elem_classes=["decode-box"])
                    latency_out = gr.Textbox(label="Performance", value="", lines=1, elem_classes=["decode-box"])

            btn.click(fn=predict, inputs=[question, schema, beams, max_len], outputs=[sql_out, raw_out, parsed_out, latency_out])
            question.submit(fn=predict, inputs=[question, schema, beams, max_len], outputs=[sql_out, raw_out, parsed_out, latency_out])

            gr.Markdown("#### Examples")
            gr.Examples(
                examples=[
                    ["What is terrence ross' nationality", "players: Player, No., Nationality, Position, Years in Toronto, School/Club Team", 5, 256],
                    ["how many schools or teams had jalen rose", "players: Player, No., Nationality, Position, Years in Toronto, School/Club Team", 5, 256],
                    ["What was the date of the race in Misano?", "races: No, Date, Round, Circuit, Pole Position, Fastest Lap, Race winner, Report", 5, 256],
                    ["What was the number of race that Kevin Curtain won?", "races: No, Date, Round, Circuit, Pole Position, Fastest Lap, Race winner, Report", 5, 256],
                    ["Where was Assen held?", "races: No, Date, Round, Circuit, Pole Position, Fastest Lap, Race winner, Report", 5, 256],
                    ["How many different positions did Sherbrooke Faucons (qmjhl) provide in the draft?", "draft: Pick, Player, Position, Nationality, NHL team, College/junior/club team", 5, 256],
                    ["What are the nationalities of the player picked from Thunder Bay Flyers (ushl)", "draft: Pick, Player, Position, Nationality, NHL team, College/junior/club team", 5, 256],
                    ["How many different nationalities do the players of New Jersey Devils come from?", "draft: Pick, Player, Position, Nationality, NHL team, College/junior/club team", 5, 256],
                    ["What's Dorain Anneck's pick number?", "draft: Pick, Player, Position, Nationality, NHL team, College/junior/club team", 5, 256],
                ],
                inputs=[question, schema, beams, max_len],
                outputs=[sql_out, raw_out, parsed_out, latency_out],
                fn=predict,
                cache_examples=False,
            )

        # ══ TAB 2: HOW IT WORKS ══
        with gr.Tab("How It Works"):
            gr.HTML("""
            <div class="arch-card">
                <h3>Architecture</h3>
                <p>A <strong>T5-base</strong> encoder-decoder fine-tuned on WikiSQL.
                Instead of generating raw SQL, it outputs <strong>structured tokens</strong>
                — column indices and operator codes — which a deterministic decoder
                maps to actual SQL using the provided schema.</p>
            </div>
            <div class="arch-grid">
                <div class="arch-card">
                    <h3>1. Input Encoding</h3>
                    <p>Question + schema concatenated:</p>
                    <p><code>translate to SQL: {question} | schema: {table}: {col1}, {col2}</code></p>
                    <p>Column order matters — the model references columns by <strong>0-based index</strong>.</p>
                </div>
                <div class="arch-card">
                    <h3>2. T5 Generation</h3>
                    <p>The encoder processes input, decoder generates structured tokens via beam search.</p>
                    <p>Output: <code>SEL:{col} | AGG:{agg} | CONDS:{col},{op},{val}</code></p>
                </div>
                <div class="arch-card">
                    <h3>3. Structured Decoding</h3>
                    <ul style="margin:0.4rem 0;padding-left:1.2rem;">
                        <li><strong>SEL</strong> — column index to SELECT</li>
                        <li><strong>AGG</strong> — aggregation (0=none, 3=COUNT, etc.)</li>
                        <li><strong>CONDS</strong> — WHERE conditions as <code>col,op,value</code> tuples</li>
                    </ul>
                </div>
                <div class="arch-card">
                    <h3>4. SQL Assembly</h3>
                    <p>Indices mapped back to column names from schema. Operators converted to SQL.
                    Result: a valid, executable query.</p>
                </div>
            </div>
            <div class="arch-card">
                <h3>Why Structured Output?</h3>
                <ul style="margin:0.4rem 0;padding-left:1.2rem;">
                    <li><strong>Schema-agnostic</strong> — learns patterns, not column names</li>
                    <li><strong>Always valid SQL</strong> — deterministic decoder guarantees syntax</li>
                    <li><strong>Smaller search space</strong> — predicts indices, not full strings</li>
                    <li><strong>Interpretable</strong> — each component inspectable independently</li>
                </ul>
            </div>
            <div class="arch-card">
                <h3>Encoding Reference</h3>
                <table class="encoding-table">
                    <tr><th>Component</th><th>Index</th><th>Meaning</th></tr>
                    <tr><td><strong>AGG</strong></td><td class="mono">0</td><td>No aggregation</td></tr>
                    <tr><td></td><td class="mono">1</td><td>MAX</td></tr>
                    <tr><td></td><td class="mono">2</td><td>MIN</td></tr>
                    <tr><td></td><td class="mono">3</td><td>COUNT</td></tr>
                    <tr><td></td><td class="mono">4</td><td>SUM</td></tr>
                    <tr><td></td><td class="mono">5</td><td>AVG</td></tr>
                    <tr><td><strong>OP</strong></td><td class="mono">0</td><td>= (equals)</td></tr>
                    <tr><td></td><td class="mono">1</td><td>> (greater than)</td></tr>
                    <tr><td></td><td class="mono">2</td><td>< (less than)</td></tr>
                    <tr><td></td><td class="mono">3</td><td>>= (greater or equal)</td></tr>
                    <tr><td></td><td class="mono">4</td><td><= (less or equal)</td></tr>
                    <tr><td></td><td class="mono">5</td><td>!= (not equal)</td></tr>
                </table>
            </div>
            """)

        # ══ TAB 3: MODEL INFO ══
        with gr.Tab("Model & Training"):
            gr.HTML("""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">220M</div>
                    <div class="stat-label">Parameters</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">80K+</div>
                    <div class="stat-label">Training Examples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">T5-base</div>
                    <div class="stat-label">Architecture</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">WikiSQL</div>
                    <div class="stat-label">Dataset</div>
                </div>
            </div>
            <div class="arch-grid">
                <div class="arch-card">
                    <h3>Model</h3>
                    <ul style="margin:0.4rem 0;padding-left:1.2rem;">
                        <li><strong>Base:</strong> T5-base (encoder-decoder)</li>
                        <li><strong>Tokenizer:</strong> SentencePiece (32K vocab)</li>
                        <li><strong>Max input:</strong> 512 tokens</li>
                        <li><strong>Max output:</strong> 256 tokens</li>
                        <li><strong>Decoding:</strong> Beam search (5 beams)</li>
                        <li><strong>Framework:</strong> Transformers + PyTorch</li>
                    </ul>
                </div>
                <div class="arch-card">
                    <h3>Training</h3>
                    <ul style="margin:0.4rem 0;padding-left:1.2rem;">
                        <li><strong>Dataset:</strong> WikiSQL (Zhong et al., 2017)</li>
                        <li><strong>Train:</strong> ~56,355 examples</li>
                        <li><strong>Dev:</strong> ~8,421 examples</li>
                        <li><strong>Test:</strong> ~15,878 examples</li>
                        <li><strong>Output:</strong> Structured tokens (SEL/AGG/CONDS)</li>
                        <li><strong>Prefix:</strong> <code>translate to SQL:</code></li>
                    </ul>
                </div>
                <div class="arch-card">
                    <h3>WikiSQL Dataset</h3>
                    <p>80,654 hand-annotated SQL queries across 24,241 Wikipedia tables.
                    Single-table queries with SELECT, aggregation, and WHERE conditions.</p>
                    <p style="margin-top:0.4rem;"><a href="https://github.com/salesforce/WikiSQL" target="_blank">github.com/salesforce/WikiSQL</a></p>
                </div>
                <div class="arch-card">
                    <h3>Limitations</h3>
                    <ul style="margin:0.4rem 0;padding-left:1.2rem;">
                        <li><strong>Single-table only</strong> — no JOINs or subqueries</li>
                        <li><strong>Fixed operators</strong> — =, >, <, >=, <=, !=</li>
                        <li><strong>No GROUP BY / ORDER BY</strong></li>
                        <li><strong>AND-only</strong> conditions</li>
                        <li><strong>Schema required</strong> as input</li>
                    </ul>
                </div>
            </div>
            """)

    gr.HTML("""
    <div class="app-footer">
        <a href="https://huggingface.co/RealMati/t2sql_v6_structured" target="_blank">Model</a>
        &nbsp;&bull;&nbsp;
        <a href="https://github.com/salesforce/WikiSQL" target="_blank">WikiSQL</a>
        &nbsp;&bull;&nbsp;
        Built with Transformers &amp; Gradio
    </div>
    """)

demo.launch(theme=theme, css=CSS)
