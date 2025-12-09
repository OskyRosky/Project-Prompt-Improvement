# app_ollama.py

import json
import textwrap
import requests
import streamlit as st
import os

# ---------------------------
# Basic configuration
# ---------------------------

# Allow configuring the Ollama URL via environment variable (useful in Docker)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"

OLLAMA_MODEL = "llama3.3"   # Change if your model has a different name

st.set_page_config(
    page_title="PromptLab Academy – Ollama Edition",
    page_icon="✨",
    layout="wide",
)

# Global styles
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .prompt-card {
        background-color: #161a23;
        padding: 1.2rem 1.5rem;
        border-radius: 0.8rem;
        border: 1px solid #262b3a;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title & description
st.title("✨ PromptLab Academy – Prompt Quality Analyzer (Ollama)")
st.write(
    "Paste your prompt below and the app will score it from **1 to 100**, "
    "give you a detailed diagnosis, and generate an optimized version ready to copy and use."
)

# Dark blue separator (50% width)
st.markdown(
    """
    <hr style="
        border: 1px solid #1f3b5b;
        width: 50%;
        margin: 0.5rem 0 1.5rem 0;
    ">
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers to talk to Ollama
# ---------------------------

def ollama_chat(messages, model: str = OLLAMA_MODEL, timeout: int = 180) -> str:
    """
    Call the Ollama chat API and return the assistant text.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def extract_json_from_text(text: str) -> dict:
    """
    Try to extract a valid JSON object from the model output.
    Handles cases where JSON is wrapped in ```json ... ``` or other noise.
    """
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                text = part
                break

    if not text.strip().startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    return json.loads(text)


def call_prompt_evaluator(user_prompt: str) -> dict:
    """
    Ask Llama 3.3 (via Ollama) to evaluate and improve the prompt.
    Returns a dict with the defined structure.
    """
    system_message = textwrap.dedent(
        """
        You are an expert in prompt engineering.
        Your task is to evaluate the quality of a prompt that will be used with a ChatGPT-style model,
        and then improve it.

        You must:
        1. Rate the prompt from 1 to 100 using this rubric:
           - Persona / role defined: 0–25
           - Task / objective clearly stated: 0–25
           - Enough context: 0–20
           - Constraints (format, length, language, tone, steps, etc.): 0–15
           - Clarity and precision of the wording: 0–15

        2. Explain in a didactic way what is missing or weak in each dimension.
        3. Propose concrete suggestions to improve the prompt.
        4. Generate an optimized version of the prompt that:
           - Defines a clear role for the model.
           - Has a specific objective.
           - Includes the necessary context.
           - Specifies format, language and other relevant constraints.

        5. ALWAYS return your answer as a valid JSON object with this exact structure:

        {
          "total_score": int,
          "scores": {
            "persona": int,
            "task": int,
            "context": int,
            "constraints": int,
            "clarity": int
          },
          "diagnosis": {
            "persona": "string",
            "task": "string",
            "context": "string",
            "constraints": "string",
            "clarity": "string"
          },
          "improvements": [
            "string",
            "string"
          ],
          "improved_prompt": "string",
          "short_explanation": "string"
        }

        Do NOT include anything outside the JSON object.
        The language of the explanation should match the language of the original prompt.
        """
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Prompt to evaluate:\n\n{user_prompt}"},
    ]

    raw_text = ollama_chat(messages)
    return extract_json_from_text(raw_text)


def call_llm_answer(prompt: str) -> str:
    """
    Ask Llama 3.3 for a normal answer to the prompt.
    Used to compare 'original vs optimized' behavior.
    """
    messages = [
        {
            "role": "system",
            "content": "Answer the following prompt in a clear, useful and concise way.",
        },
        {"role": "user", "content": prompt},
    ]
    return ollama_chat(messages, timeout=180)

# ---------------------------
# Streamlit state
# ---------------------------

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "original_answer" not in st.session_state:
    st.session_state.original_answer = None

if "improved_answer" not in st.session_state:
    st.session_state.improved_answer = None

# ---------------------------
# Main UI: Original prompt
# ---------------------------

st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
st.markdown(
    """
    <p style="
        font-size:1.1rem;
        font-weight:700;
        margin:0 0 0.5rem 0;
    ">
        Original prompt
    </p>
    """,
    unsafe_allow_html=True,
)

default_text = "Explain what machine learning is."
user_prompt = st.text_area(
    "Paste your prompt here:",
    value=default_text,
    height=180,
)
st.markdown("</div>", unsafe_allow_html=True)

# Evaluation button
evaluate_btn = st.button("✅ Evaluate & Optimize Prompt", type="primary")

# ---------------------------
# Evaluation logic
# ---------------------------

if evaluate_btn:
    if not user_prompt.strip():
        st.warning("Please write a prompt before evaluating it.")
    else:
        with st.spinner("Evaluating prompt with Llama 3.3 (Ollama)..."):
            try:
                evaluation = call_prompt_evaluator(user_prompt)
                st.session_state.evaluation = evaluation
                # Clear previous comparison answers, if any
                st.session_state.original_answer = None
                st.session_state.improved_answer = None
            except Exception as e:
                st.error(f"An error occurred while evaluating the prompt: {e}")

evaluation = st.session_state.evaluation

# ---------------------------
# Show evaluation results
# ---------------------------

if evaluation:
    st.markdown("---")

    total_score = evaluation.get("total_score", 0)
    scores = evaluation.get("scores", {})
    diagnosis = evaluation.get("diagnosis", {})
    improvements = evaluation.get("improvements", [])
    improved_prompt = evaluation.get("improved_prompt", "")
    short_explanation = evaluation.get("short_explanation", "")

    # 1) Dimension breakdown
    st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="
            font-size:1.1rem;
            font-weight:700;
            margin:0 0 0.5rem 0;
        ">
            Dimension breakdown
        </p>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Persona / Role", scores.get("persona", 0))
    c2.metric("Task / Objective", scores.get("task", 0))
    c3.metric("Context", scores.get("context", 0))
    c4.metric("Constraints", scores.get("constraints", 0))
    c5.metric("Clarity", scores.get("clarity", 0))

    st.markdown("</div>", unsafe_allow_html=True)

    # 2) Global result
    st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="
            font-size:1.1rem;
            font-weight:700;
            margin:0 0 0.5rem 0;
        ">
            Overall prompt score
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.metric("Total score (1–100)", total_score)

    if short_explanation:
        st.info(short_explanation)

    st.markdown("</div>", unsafe_allow_html=True)

    # 3) Didactic diagnosis
    st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="
            font-size:1.1rem;
            font-weight:700;
            margin:0 0 0.5rem 0;
        ">
            Didactic diagnosis
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.write(f"**Persona / Role:** {diagnosis.get('persona', '')}")
    st.write(f"**Task / Objective:** {diagnosis.get('task', '')}")
    st.write(f"**Context:** {diagnosis.get('context', '')}")
    st.write(f"**Constraints:** {diagnosis.get('constraints', '')}")
    st.write(f"**Clarity:** {diagnosis.get('clarity', '')}")

    st.markdown("</div>", unsafe_allow_html=True)

    # 4) Improvement suggestions
    if improvements:
        st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="
                font-size:1.1rem;
                font-weight:700;
                margin:0 0 0.5rem 0;
            ">
                Improvement suggestions
            </p>
            """,
            unsafe_allow_html=True,
        )
        for i, idea in enumerate(improvements, start=1):
            st.markdown(f"- **{i}.** {idea}")
        st.markdown("</div>", unsafe_allow_html=True)

    # 5) Optimized prompt (vertical scroll)
    st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="
            font-size:1.1rem;
            font-weight:700;
            margin:0 0 0.5rem 0;
        ">
            Optimized prompt
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.text_area(
        "You can copy and reuse this optimized prompt:",
        value=improved_prompt or "The optimized prompt could not be generated.",
        height=220,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # 6) Prompt & answer comparison (at the end)
    # ---------------------------
    if improved_prompt:
        st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="
                font-size:1.1rem;
                font-weight:700;
                margin:0 0 0.5rem 0;
            ">
                Comparison: original vs optimized prompt
            </p>
            """,
            unsafe_allow_html=True,
        )

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown("**Original prompt**")
            st.text_area(
                "Original prompt",
                value=user_prompt,
                height=180,
                key="original_prompt_display",
            )

        with col_p2:
            st.markdown("**Optimized prompt**")
            st.text_area(
                "Optimized prompt",
                value=improved_prompt,
                height=180,
                key="optimized_prompt_display",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Button to generate and compare answers
        compare_btn = st.button("🔄 Generate & compare answers")

        if compare_btn:
            with st.spinner("Generating answers for both prompts..."):
                try:
                    original_answer = call_llm_answer(user_prompt)
                    improved_answer = call_llm_answer(improved_prompt)
                    st.session_state.original_answer = original_answer
                    st.session_state.improved_answer = improved_answer
                except requests.exceptions.ReadTimeout:
                    st.error(
                        "⏱️ The model took too long to respond. "
                        "You can try again or use a shorter prompt."
                    )
                except Exception as e:
                    st.error(f"An error occurred while generating the answers: {e}")

        # If we already have answers, show them side by side
        if st.session_state.original_answer and st.session_state.improved_answer:
            st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
            st.markdown(
                """
                <p style="
                    font-size:1.1rem;
                    font-weight:700;
                    margin:0 0 0.5rem 0;
                ">
                    Generated answers: original vs optimized
                </p>
                """,
                unsafe_allow_html=True,
            )

            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.markdown("**Answer with the original prompt**")
                st.write(st.session_state.original_answer)

            with col_r2:
                st.markdown("**Answer with the optimized prompt**")
                st.write(st.session_state.improved_answer)

            st.markdown("</div>", unsafe_allow_html=True)