import re
import os
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time

MODEL_ID = "RealMati/text2sql-wikisql-v5"

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")

css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path, "r") as f:
    CSS = f.read()

SQL_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "LIKE",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AS",
    "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET",
    "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX",
    "BETWEEN", "EXISTS", "UNION", "ALL", "ANY", "CASE",
    "WHEN", "THEN", "ELSE", "END", "IS", "NULL", "ASC", "DESC",
]


def postprocess_sql(sql):
    sql = sql.strip()
    sql = re.sub(r"<pad>|<unk>|<s>|</s>", "", sql)
    sql = re.sub(r"\s+", " ", sql)
    for kw in SQL_KEYWORDS:
        sql = re.sub(rf"\b{re.escape(kw.lower())}\b", kw, sql, flags=re.IGNORECASE)
    return sql.strip()


def predict(question, schema, num_beams, max_length):
    if not question or not question.strip():
        return (
            "-- Enter a question and schema, then click Generate SQL",
            "",
        )

    input_text = f"translate to SQL: {question}"
    if schema and schema.strip():
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

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = postprocess_sql(raw)
    perf = f"Inference: {latency:.2f}s  |  Beams: {int(num_beams)}  |  Tokens: {inputs['input_ids'].shape[1]}"

    return sql, perf


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

with gr.Blocks(title="Text-to-SQL V5 | Direct SQL Output") as demo:

    gr.HTML("""
    <div class="app-header">
        <h1><span>Text-to-SQL</span> <small>V5</small></h1>
    </div>
    <div class="tech-badges">
        <span class="badge badge-indigo">T5-base (220M)</span>
        <span class="badge badge-purple">Seq2Seq</span>
        <span class="badge badge-emerald">WikiSQL 80K+</span>
        <span class="badge badge-amber">Direct SQL Output</span>
    </div>
    <div class="pipeline-strip">
        <span class="step step-input">Question</span>
        <span class="arrow">&rarr;</span>
        <span class="step step-model">T5 Encoder-Decoder</span>
        <span class="arrow">&rarr;</span>
        <span class="step step-sql">SQL Query</span>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("Demo"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    question = gr.Textbox(
                        label="Natural Language Question",
                        placeholder="e.g. What is terrence ross' nationality?",
                        lines=2,
                    )
                    schema = gr.Textbox(
                        label="Database Schema (optional)",
                        placeholder="table_name: col1, col2, col3, ...",
                        lines=2,
                    )
                    gr.HTML('<p class="input-hint">Format: <code>table: col1, col2, col3</code></p>')
                    with gr.Row():
                        beams = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Beam Size")
                        max_len = gr.Slider(minimum=64, maximum=512, value=256, step=64, label="Max Length")
                    btn = gr.Button("Generate SQL", variant="primary", elem_classes=["generate-btn"], size="lg")

                with gr.Column(scale=1):
                    sql_out = gr.Textbox(
                        label="Generated SQL",
                        value="-- Enter a question, then click Generate SQL",
                        lines=4,
                        elem_classes=["sql-output"],
                    )
                    latency_out = gr.Textbox(label="Performance", value="", lines=1, elem_classes=["decode-box"])

            btn.click(fn=predict, inputs=[question, schema, beams, max_len], outputs=[sql_out, latency_out])
            question.submit(fn=predict, inputs=[question, schema, beams, max_len], outputs=[sql_out, latency_out])

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
                outputs=[sql_out, latency_out],
                fn=predict,
                cache_examples=False,
            )

        with gr.Tab("How It Works"):
            gr.HTML("""
            <div class="arch-card">
                <h3>Architecture</h3>
                <p>A <strong>T5-base</strong> encoder-decoder fine-tuned on WikiSQL.
                This version generates <strong>SQL directly</strong> as free-form text,
                unlike V6 which outputs structured tokens. The model learns to produce
                syntactically correct SQL from natural language questions.</p>
            </div>
            <div class="arch-grid">
                <div class="arch-card">
                    <h3>Input Format</h3>
                    <p>Question and optional schema concatenated:</p>
                    <p><code>translate to SQL: {question} | schema: {table}: {col1}, {col2}</code></p>
                </div>
                <div class="arch-card">
                    <h3>Output Format</h3>
                    <p>The model directly outputs SQL text:</p>
                    <p><code>SELECT Nationality FROM players WHERE Player = 'Terrence Ross'</code></p>
                    <p>Post-processing normalizes whitespace and uppercases SQL keywords.</p>
                </div>
            </div>
            <div class="arch-card">
                <h3>V5 vs V6 Comparison</h3>
                <table class="encoding-table">
                    <tr><th>Aspect</th><th>V5 (Direct SQL)</th><th>V6 (Structured)</th></tr>
                    <tr><td>Output</td><td>Raw SQL string</td><td>SEL/AGG/CONDS tokens</td></tr>
                    <tr><td>Schema dependency</td><td>Optional</td><td>Required (for index mapping)</td></tr>
                    <tr><td>Flexibility</td><td>Can produce any SQL</td><td>Limited to WikiSQL operations</td></tr>
                    <tr><td>Reliability</td><td>May produce invalid SQL</td><td>Guaranteed valid structure</td></tr>
                    <tr><td>Generalization</td><td>Memorizes column names</td><td>Schema-agnostic indices</td></tr>
                </table>
            </div>
            """)

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
                        <li><strong>Output:</strong> Direct SQL strings</li>
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
                        <li><strong>May hallucinate</strong> column names not in schema</li>
                        <li><strong>No syntax guarantee</strong> — free-form output can be invalid</li>
                        <li><strong>AND-only</strong> conditions</li>
                    </ul>
                </div>
            </div>
            """)

    gr.HTML("""
    <div class="app-footer">
        <a href="https://huggingface.co/RealMati/text2sql-wikisql-v5" target="_blank">Model</a>
        &nbsp;&bull;&nbsp;
        <a href="https://github.com/salesforce/WikiSQL" target="_blank">WikiSQL</a>
        &nbsp;&bull;&nbsp;
        Built with Transformers &amp; Gradio
    </div>
    """)

demo.launch(theme=theme, css=CSS)
