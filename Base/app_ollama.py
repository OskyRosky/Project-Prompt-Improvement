# app_ollama.py
import json
import textwrap
import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.3"   # Ajusta si tu modelo tiene otro nombre

st.set_page_config(
    page_title="PromptLab Academy – Ollama Edition",
    page_icon="✨",
    layout="wide",
)

st.title("✨ PromptLab Academy – Prompt Quality Analyzer (Ollama)")
st.write(
    "Pega tu prompt abajo y la app lo evaluará del **1 al 100**, "
    "te dará un diagnóstico detallado y generará una versión optimizada."
)

# ---------------------------
# Helpers para hablar con Ollama
# ---------------------------

def ollama_chat(messages, model: str = OLLAMA_MODEL) -> str:
    """
    Llama a la API de chat de Ollama y devuelve el texto del assistant.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,  # importante: respuesta no en streaming
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Estructura típica: {"message": {"role": "assistant", "content": "..."}, "done": true, ...}
    return data["message"]["content"]


def extract_json_from_text(text: str) -> dict:
    """
    Intenta extraer un JSON válido desde el texto devuelto por el modelo.
    Maneja casos donde venga envuelto en ```json ... ``` u otro ruido.
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
    Pide a Llama 3.3 (vía Ollama) que evalúe y optimice el prompt.
    Devuelve un dict con la estructura definida.
    """
    system_message = textwrap.dedent(
        """
        Eres un experto en ingeniería de prompts.
        Tu tarea es evaluar la calidad de un prompt que se usará con un modelo tipo ChatGPT y luego mejorarlo.

        Debes:
        1. Calificar el prompt de 1 a 100 usando esta rúbrica:
           - Persona / rol definido: 0–25
           - Tarea / objetivo claro: 0–25
           - Contexto suficiente: 0–20
           - Restricciones (formato, longitud, idioma, tono, pasos, etc.): 0–15
           - Claridad y precisión del lenguaje: 0–15
        2. Explicar de forma didáctica qué falla en cada dimensión.
        3. Proponer sugerencias concretas para mejorar el prompt.
        4. Generar una versión optimizada del prompt, que:
           - Defina un rol claro para el modelo.
           - Tenga un objetivo específico.
           - Incluya el contexto necesario.
           - Especifique formato, idioma y otras restricciones.
        5. Devolver SIEMPRE la respuesta en formato JSON válido con esta estructura:

        {
          "total_score": int,
          "scores": {
            "persona": int,
            "tarea": int,
            "contexto": int,
            "restricciones": int,
            "claridad": int
          },
          "diagnosis": {
            "persona": "string",
            "tarea": "string",
            "contexto": "string",
            "restricciones": "string",
            "claridad": "string"
          },
          "improvements": [
            "string",
            "string"
          ],
          "improved_prompt": "string",
          "short_explanation": "string"
        }

        No incluyas nada fuera del JSON.
        El idioma de la explicación debe coincidir con el idioma del prompt original.
        """
    )

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": f"Prompt a evaluar:\n\n{user_prompt}",
        },
    ]

    raw_text = ollama_chat(messages)
    return extract_json_from_text(raw_text)


def call_llm_answer(prompt: str) -> str:
    """
    Pide a Llama 3.3 una respuesta normal al prompt.
    Se usa para comparar 'original vs optimizado'.
    """
    messages = [
        {
            "role": "system",
            "content": "Responde al siguiente prompt de forma útil, clara y concisa.",
        },
        {"role": "user", "content": prompt},
    ]
    return ollama_chat(messages)

# ---------------------------
# UI principal Streamlit
# ---------------------------

st.subheader("🧾 Prompt original")

default_text = "Explícame qué es el aprendizaje automático."

user_prompt = st.text_area(
    "Pega aquí tu prompt:",
    value=default_text,
    height=200,
)

col_eval, col_compare = st.columns([1, 1])

with col_eval:
    evaluate_btn = st.button("✅ Evaluar y Optimizar Prompt", type="primary")

with col_compare:
    compare_btn = st.button("🔄 Generar y Comparar Respuestas")

# Estado
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None
if "original_answer" not in st.session_state:
    st.session_state.original_answer = None
if "improved_answer" not in st.session_state:
    st.session_state.improved_answer = None

# ---------------------------
# Evaluación
# ---------------------------
if evaluate_btn:
    if not user_prompt.strip():
        st.warning("Escribe un prompt antes de evaluarlo.")
    else:
        with st.spinner("Evaluando prompt con Llama 3.3 (Ollama)..."):
            try:
                evaluation = call_prompt_evaluator(user_prompt)
                st.session_state.evaluation = evaluation
                st.session_state.original_answer = None
                st.session_state.improved_answer = None
            except Exception as e:
                st.error(f"Ocurrió un error al evaluar el prompt: {e}")

evaluation = st.session_state.evaluation

if evaluation:
    st.markdown("---")
    st.subheader("📊 Resultado de la Evaluación")

    total_score = evaluation.get("total_score", 0)
    scores = evaluation.get("scores", {})
    diagnosis = evaluation.get("diagnosis", {})
    improvements = evaluation.get("improvements", [])
    improved_prompt = evaluation.get("improved_prompt", "")
    short_explanation = evaluation.get("short_explanation", "")

    st.metric("Puntuación total (1–100)", total_score)

    st.write("### Desglose por dimensión")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Persona / Rol", scores.get("persona", 0))
    c2.metric("Tarea / Objetivo", scores.get("tarea", 0))
    c3.metric("Contexto", scores.get("contexto", 0))
    c4.metric("Restricciones", scores.get("restricciones", 0))
    c5.metric("Claridad", scores.get("claridad", 0))

    st.write("### Diagnóstico didáctico")
    st.write(f"**Persona / Rol:** {diagnosis.get('persona', '')}")
    st.write(f"**Tarea / Objetivo:** {diagnosis.get('tarea', '')}")
    st.write(f"**Contexto:** {diagnosis.get('contexto', '')}")
    st.write(f"**Restricciones:** {diagnosis.get('restricciones', '')}")
    st.write(f"**Claridad:** {diagnosis.get('claridad', '')}")

    if short_explanation:
        st.info(short_explanation)

    if improvements:
        st.write("### Sugerencias de mejora")
        for i, idea in enumerate(improvements, start=1):
            st.markdown(f"- **{i}.** {idea}")

    st.write("### 🧠 Prompt optimizado")
    st.code(
        improved_prompt or "No se pudo generar un prompt optimizado.",
        language="markdown",
    )

# ---------------------------
# Comparación de respuestas
# ---------------------------
if compare_btn:
    if not evaluation:
        st.warning("Primero evalúa el prompt para generar una versión optimizada.")
    else:
        improved_prompt = evaluation.get("improved_prompt", "")
        if not improved_prompt:
            st.warning("No hay prompt optimizado disponible.")
        else:
            with st.spinner("Generando respuestas (original vs optimizada)..."):
                try:
                    original_answer = call_llm_answer(user_prompt)
                    improved_answer = call_llm_answer(improved_prompt)
                    st.session_state.original_answer = original_answer
                    st.session_state.improved_answer = improved_answer
                except Exception as e:
                    st.error(f"Ocurrió un error al generar las respuestas: {e}")

if st.session_state.original_answer and st.session_state.improved_answer:
    st.markdown("---")
    st.subheader("🔍 Comparación de respuestas: Original vs Optimizada")

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Prompt original")
        st.code(user_prompt, language="markdown")
        st.write("#### Respuesta")
        st.write(st.session_state.original_answer)

    with col2:
        st.write("#### Prompt optimizado")
        st.code(evaluation.get("improved_prompt", ""), language="markdown")
        st.write("#### Respuesta")
        st.write(st.session_state.improved_answer)

    st.success(
        "Deberías notar más estructura, contexto y claridad en la respuesta con el prompt optimizado."
    )