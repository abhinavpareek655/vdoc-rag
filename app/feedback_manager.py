import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.json")


def _load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def _save_feedback(data):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def record_feedback(question, answer, correctness, sources=None):
    """
    Store user feedback about RAG answer correctness.
    correctness: 'correct' | 'incorrect' | 'partial'
    """
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "correctness": correctness,
        "sources": sources or [],
    }
    data = _load_feedback()
    data.append(entry)
    _save_feedback(data)
    print(f"📝 Feedback recorded ({correctness}) for: {question[:60]}...")
    return entry


def get_feedback_summary():
    data = _load_feedback()
    total = len(data)
    if total == 0:
        return "No feedback yet."
    correct = sum(1 for x in data if x.get("correctness") == "correct")
    incorrect = sum(1 for x in data if x.get("correctness") == "incorrect")
    return f"Feedback Stats — ✅ {correct} correct, ❌ {incorrect} incorrect, total {total}"
