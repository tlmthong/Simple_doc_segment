import streamlit as st
import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict


# --- Data Schemas ---
class LinePairSchema(BaseModel):
    start: int
    end: int


class SegmentSchema(BaseModel):
    name: str
    line_pairs: List[LinePairSchema]


class SegmentResponseSchema(BaseModel):
    sections: List[SegmentSchema]


# --- OpenAI / Ollama Client Configuration ---
MODEL_ID = "gpt-oss:120b-cloud"
OLLAMA_HOST = "http://localhost:11434"
BASE_URL = f"{OLLAMA_HOST}/v1"
client = OpenAI(base_url=BASE_URL, api_key="NO_KEY")


def segment_doc(doc: str):
    sys_prompt = """
Role
You're an expert in segmenting a document into useful chunks.

Task
Your task is to split the document into meaningful sections according to the document structure. You will be given a document with line number indications enclosed by "<<>>" signs (e.g., <<1>>). You will investigate the document structure and return a JSON object that includes the section name and a list of start/end line pairs. The name should be specific and unique, for example "Section 2 - Subsection 1 - Part 1". 

Rules:
1. Return ONLY raw JSON. Do NOT include markdown code blocks (e.g., no ```json).
2. Names must be unique and specific.
3. If a segment is non-contiguous, include multiple line pairs.
4. Merge consecutive line pairs if they belong to the same segment (e.g., {1, 2} and {3, 4} -> {1, 4}).
5. Only include meaningful segments that contains legal information; exclude metadata or sign-off sections.
6. Segments needs to be split down to the smallest unit of the document strucutre. For example if there is subsections, it needs to be ground to subsection level.
7. The if there is an identifier number of the segment then include it for example "Part A - Clause 4" is better than "Part A - Interpretation"


Example Input:
"
<<1>>Introduction 
<<2>>This is a sample document
<<3>>Section 1
<<4>>This section is about apples. Apples are:
<<5>>a) a fruit grown on a tree 
<<6>>b) Eaten by humans
<<7>>From this fact we can see that apples are safe to eat.
<<8>>And they can be grown in my garden.
<<9>>However I need a big garden.
"

Example Output:
{
  "sections": [
    {
      "name": "Introduction",
      "line_pairs": [{"start": 1, "end": 2}]
    },
    {
      "name": "Section 1",
      "line_pairs": [{"start": 3, "end": 4}, {"start": 7, "end": 9}]
    },
    {
      "name": "Section 1 - Part A",
      "line_pairs": [{"start": 5, "end": 5}]
    },
    {
      "name": "Section 1 - Part B",
      "line_pairs": [{"start": 6, "end": 6}]
    }
  ]
}

Text to analyze:
"""
    message = sys_prompt + doc
    resp = client.beta.chat.completions.parse(
        model=MODEL_ID,
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.2,
        response_format=SegmentResponseSchema,
    )
    return resp.choices[0].message.parsed


def add_line_numbers_to_str(text: str):
    lines = text.splitlines()
    numbered_lines = []
    for count, line in enumerate(lines, 1):
        numbered_lines.append(f"<<{count}>> {line}")
    return "\n".join(numbered_lines), lines


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Document Segment Visualizer")
st.title("📄 Document Segment Visualizer")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")

    with st.spinner("Processing document..."):
        numbered_doc, original_lines = add_line_numbers_to_str(content)
        try:
            result = segment_doc(numbered_doc)
            sections = result.sections
        except Exception as e:
            st.error(f"Error during segmentation: {e}")
            sections = []

    if sections:
        st.subheader("Segmented View")
        st.info("💡 **Hover** over a green block to see its segment name.")

        # Prepare HTML for display
        html_output = """
        <style>
            .line-container {
                font-family: 'Courier New', Courier, monospace;
                white-space: pre;
                line-height: 1.2;
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #dee2e6;
                overflow-x: auto;
            }
            .segment-block {
                display: block;
                border: 1px solid transparent;
                border-radius: 4px;
                transition: all 0.2s ease;
                cursor: help;
                margin: 2px 0;
            }
            .segment-block:hover {
                background-color: #fff3cd !important;
                border: 1px solid #ffeeba;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transform: scale(1.002);
                z-index: 10;
                position: relative;
            }
            .highlighted {
                background-color: #e2f0d9;
            }
            .unhighlighted {
                background-color: transparent;
                cursor: default;
            }
            .line-num {
                color: #adb5bd;
                font-size: 0.85em;
                user-select: none;
                margin-right: 15px;
                border-right: 1px solid #dee2e6;
                padding-right: 8px;
                display: inline-block;
                width: 40px;
                text-align: right;
            }
        </style>
        <div class="line-container">
        """

        # A block is defined as 1 Line Pair
        blocks = []
        for sec in sections:
            for lp in sec.line_pairs:
                blocks.append({"start": lp.start, "end": lp.end, "name": sec.name})

        # Sort blocks by start line
        blocks.sort(key=lambda x: (x["start"], x["end"]))

        total_lines = len(original_lines)
        current_line = 1

        while current_line <= total_lines:
            # Check if current_line starts a block
            matching_blocks = [b for b in blocks if b["start"] == current_line]

            if matching_blocks:
                block = matching_blocks[0]
                end_line = min(block["end"], total_lines)
                block_content = ""
                for l in range(current_line, end_line + 1):
                    line_text = original_lines[l - 1]
                    safe_line = (
                        line_text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    block_content += f'<span class="line-num">{l}</span>{safe_line if safe_line.strip() else " "}\n'

                html_output += f'<div class="segment-block highlighted" title="Segment: {block["name"]}">{block_content}</div>'
                current_line = end_line + 1
            else:
                line_text = original_lines[current_line - 1]
                safe_line = (
                    line_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                html_output += f'<div class="segment-block unhighlighted"><span class="line-num">{current_line}</span>{safe_line if safe_line.strip() else " "}\n</div>'
                current_line += 1

        html_output += "</div>"

        st.components.v1.html(html_output, height=800, scrolling=True)

        with st.expander("Show Raw Segmentation JSON"):
            st.json(result.model_dump())
    else:
        st.warning("No segments detected or an error occurred.")
