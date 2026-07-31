"""
agent.py
--------
The orchestrator. Ties together:
  - tools.py   -> PDF/text extraction & keyword extraction
  - rag.py     -> TF-IDF fit-score retrieval
  - memory.py  -> per-session history
  - Groq LLM   -> plain-language explanation of the fit score

This is the single entry point `main.py` calls; it hides the pipeline
details behind one `analyze()` function.
"""

import os

from rag import compute_fit_score, top_matching_terms
from tools import extract_keywords
import memory

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_client = Groq(api_key=GROQ_API_KEY) if (GROQ_AVAILABLE and GROQ_API_KEY) else None


def _generate_explanation(resume_text: str, job_description: str, score: float, matched_terms: list[str]) -> str:
    """Ask the Groq LLM (Llama 3.3) to explain the fit score in plain language."""
    if not _client:
        return (
            f"LLM explanation unavailable (set GROQ_API_KEY to enable). "
            f"Computed fit score: {score}/100. Matched terms: {', '.join(matched_terms) or 'none'}."
        )

    prompt = f"""You are a recruiting assistant. A candidate's resume was compared
against a job description using a similarity algorithm and scored {score}/100.
Overlapping key terms found in both: {', '.join(matched_terms) or 'none'}.

Job Description:
{job_description[:1500]}

Resume:
{resume_text[:1500]}

In 3-4 concise bullet points, explain why this candidate is or isn't a strong
fit, referencing specific skills/experience. Be direct and specific."""

    completion = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return completion.choices[0].message.content


def analyze(resume_text: str, job_description: str, session_id: str = "default") -> dict:
    """
    Run the full pipeline: score -> matched terms -> LLM explanation -> save to memory.
    Returns a dict ready to be serialized as the API response.
    """
    score = compute_fit_score(resume_text, job_description)
    matched_terms = top_matching_terms(resume_text, job_description)
    jd_keywords = extract_keywords(job_description)
    explanation = _generate_explanation(resume_text, job_description, score, matched_terms)

    result = {
        "fit_score": score,
        "matched_keywords": matched_terms,
        "job_keywords": jd_keywords,
        "explanation": explanation,
    }

    memory.add_analysis(session_id, {
        "job_description": job_description[:200],
        "fit_score": score,
    })

    return result


def get_session_history(session_id: str = "default") -> list[dict]:
    return memory.get_history(session_id)
