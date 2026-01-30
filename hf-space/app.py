import re
import os
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

MODEL_ID = "RealMati/t2sql_v6_structured"

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
OPS = ["=", ">", "<", ">=", "<=", "!="]

# Load CSS from external file
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
        agg_label = AGG_OPS[agg] if agg < len(AGG_OPS) and agg > 0 else "None"
        parts.append(f"Aggregation: {agg_label}")
    if conds:
        cond_strs = []
        for c_idx, c_op, c_val in conds:
            c_name = columns[c_idx] if c_idx < len(columns) else f"col{c_idx}"
            op_str = OPS[c_op] if c_op < len(OPS) else "="
            cond_strs.append(f"{c_name} {op_str} {c_val}")
        parts.append(f"Conditions: {', '.join(cond_strs)}")
    else:
        parts.append("Conditions: None")
    return " | ".join(parts)


def predict(question, schema, num_beams, max_length):
    if not question.strip():
        return "", "", ""

    table_name, columns = parse_schema(schema)
    input_text = f"translate to SQL: {question}"
    if schema.strip():
        input_text += f" | schema: {schema.strip()}"

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=int(max_length),
            num_beams=int(num_beams),
            early_stopping=True,
            do_sample=False,
        )

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sel, agg, conds = decode_structured_output(raw_output)

    if sel is not None and agg is not None and columns:
        sql = structured_to_sql(sel, agg, conds, columns, table_name)
    else:
        sql = "(Provide a schema to convert structured output to SQL)"

    parsed = format_parsed(sel, agg, conds, columns) if sel is not None else ""
    return sql, raw_output, parsed


theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("Fira Code"),
)

with gr.Blocks(title="Text-to-SQL Demo") as demo:
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>Text-to-SQL</h1>
        <p>Fine-tuned T5 model that converts natural language questions
        into structured SQL queries using the WikiSQL dataset</p>
    </div>
    """)

    # Pipeline visualization - dark background so text is always visible
    gr.HTML("""
    <div class="pipeline-box">
        <span class="stage">Natural Language</span>
        <span class="arrow"> &rarr; </span>
        <span class="stage">T5 Encoder</span>
        <span class="arrow"> &rarr; </span>
        <span class="highlight">Structured Tokens (SEL | AGG | CONDS)</span>
        <span class="arrow"> &rarr; </span>
        <span class="stage">SQL Query</span>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### Input", elem_classes=["section-header"])
            question = gr.Textbox(
                label="Natural Language Question",
                placeholder="e.g. What is terrence ross' nationality?",
                lines=2,
            )
            schema = gr.Textbox(
                label="Database Schema",
                placeholder="table_name: col1, col2, col3, ...",
                lines=2,
                info="Format: table_name: column1, column2, column3",
            )
            with gr.Row():
                beams = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Beam Size",
                    info="Higher = better quality, slower",
                )
                max_len = gr.Slider(
                    minimum=64, maximum=512, value=256, step=64,
                    label="Max Length",
                )
            btn = gr.Button("Generate SQL", variant="primary", elem_classes=["generate-btn"])

        with gr.Column(scale=1):
            gr.Markdown("### Output", elem_classes=["section-header"])
            sql_out = gr.Textbox(
                label="Generated SQL",
                lines=3,
                elem_classes=["sql-output"],
            )
            raw_out = gr.Textbox(
                label="Raw Model Output (Structured Tokens)",
                lines=1,
                elem_classes=["raw-output"],
            )
            parsed_out = gr.Textbox(
                label="Decoded Components",
                lines=1,
                elem_classes=["raw-output"],
            )

    btn.click(
        fn=predict,
        inputs=[question, schema, beams, max_len],
        outputs=[sql_out, raw_out, parsed_out],
    )

    gr.Markdown("### Try These Examples", elem_classes=["section-header"])
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
        outputs=[sql_out, raw_out, parsed_out],
        fn=predict,
        cache_examples=False,
    )

    gr.HTML("""
    <div class="footer-section">
        <span class="info-badge">T5-base</span>&nbsp;
        <span class="info-badge">WikiSQL</span>&nbsp;
        <span class="info-badge">Seq2Seq</span>&nbsp;
        <span class="info-badge">Structured Output</span>
        <p style="margin-top:0.75rem;">
            Model: <a href="https://huggingface.co/RealMati/t2sql_v6_structured" target="_blank">RealMati/t2sql_v6_structured</a>
        </p>
    </div>
    """)

demo.launch(theme=theme, css=CSS)
