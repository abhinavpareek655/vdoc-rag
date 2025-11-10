"""
train_feedback_embeddings.py
Fine-tune the VDoc-RAG embedding model using stored user feedback.

Place this file at the repository root and run:

    python train_feedback_embeddings.py

It will load feedback from `app/feedback.json`, prepare training pairs, fine-tune a
SentenceTransformer model, and save checkpoints under `models/vdoc_feedback_tuned/`.
"""
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
except Exception as e:
    raise ImportError("Please install sentence-transformers and torch to run this script: pip install sentence-transformers torch")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")
FEEDBACK_PATH = os.path.join(APP_DIR, "feedback.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "vdoc_feedback_tuned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: Load Feedback ---
if not os.path.exists(FEEDBACK_PATH):
    raise FileNotFoundError(f"❌ No feedback.json found at {FEEDBACK_PATH}")

with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
    feedback = json.load(f)

if not feedback:
    raise ValueError("⚠️ feedback.json is empty — collect feedback first!")

# --- Step 2: Prepare Training Data ---
train_examples = []
for fb in feedback:
    question = fb.get("question", "").strip()
    answer = fb.get("answer", "").strip()
    correctness = (fb.get("correctness") or "").lower()
    if not question or not answer:
        continue
    if correctness not in ("correct", "incorrect"):
        continue
    label = 1.0 if correctness == "correct" else 0.0
    train_examples.append(InputExample(texts=[question, answer], label=label))

if len(train_examples) < 5:
    raise ValueError(f"⚠️ Too few feedback entries ({len(train_examples)}). Need at least 5 to fine-tune meaningfully.")

print(f"✅ Loaded {len(train_examples)} feedback samples for training.")

# --- Step 3: Load Base Model ---
base_model = os.environ.get("VDOCRAG_FEEDBACK_BASE", "all-MiniLM-L6-v2")
print(f"📦 Loading base model: {base_model}")
model = SentenceTransformer(base_model)

# --- Step 4: Create DataLoader and Loss ---
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)

# --- Step 5: Train ---
print("🚀 Starting fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=10,
    show_progress_bar=True,
)

# --- Step 6: Save Fine-tuned Model ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(OUTPUT_DIR, f"checkpoint_{timestamp}")
os.makedirs(save_path, exist_ok=True)
model.save(save_path)
print(f"✅ Fine-tuned model saved at: {save_path}")

# --- Step 7: Create "latest" symlink / pointer ---
latest_path = os.path.join(OUTPUT_DIR, "latest")
try:
    if os.path.exists(latest_path):
        if os.path.islink(latest_path):
            os.unlink(latest_path)
        else:
            import shutil

            shutil.rmtree(latest_path)
    os.symlink(save_path, latest_path, target_is_directory=True)
    print(f"🔗 Symlink created: {latest_path} → {save_path}")
except Exception:
    # On Windows, symlink may fail — copy instead
    import shutil

    if os.path.exists(latest_path):
        shutil.rmtree(latest_path, ignore_errors=True)
    shutil.copytree(save_path, latest_path)
    print(f"📁 Copied model to {latest_path} (symlink not supported).")

print("\n🎉 Training complete! Your VDoc-RAG can now use:")
print(f"   models/vdoc_feedback_tuned/latest/")
