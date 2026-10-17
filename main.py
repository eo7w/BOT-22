"""
BOT-22 — بوت تليجرام ذكي مع ذاكرة دائمة على GitHub و RAG دلالي.
المكدس: FastAPI + pyTelegramBotAPI (webhook) + Groq (LLM) + Sentence-Transformers (تضمين) + GitHub (ذاكرة).
"""

from datetime import datetime
import base64
import os
import threading
import time

import numpy as np
import requests
import telebot
from fastapi import FastAPI, Request
from groq import Groq

# ---------------------------------------------------------------------------
# الإعدادات من المتغيرات البيئية
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")            # مثال: eo7w/BOT-22
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")            # مثال: https://bot-22.onrender.com
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low")
MAX_COMPLETION_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", "2048"))
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "6"))
SYNC_DEBOUNCE_SEC = int(os.environ.get("SYNC_DEBOUNCE_SEC", "20"))
EMBEDDINGS_ENABLED = os.environ.get("EMBEDDINGS_ENABLED", "true").lower() == "true"

KNOWLEDGE_FILE = "knowledge_base.txt"

# ---------------------------------------------------------------------------
# العملاء
# ---------------------------------------------------------------------------
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
# telebot لا يتحقق من التوكن عند الإنشاء، فننشئه دائماً حتى لا يتحطم تزيين المعالجات
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
app = FastAPI(title="BOT-22 — AI Agent with Groq & GitHub Memory")

embedder = None
torch = None
util = None
if EMBEDDINGS_ENABLED:
    try:
        import torch as _torch
        from sentence_transformers import SentenceTransformer, util as _util

        torch = _torch
        util = _util
        print("جاري تحميل نموذج التضمين all-MiniLM-L6-v2 ...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("تم تحميل نموذج التضمين.")
    except Exception as e:
        print(f"تعذر تحميل نموذج التضمين، سيُستخدم البحث بالكلمات المفتاحية: {e}")
        embedder = None
        torch = None
        util = None
else:
    print("التضمينات معطلة (EMBEDDINGS_ENABLED=false)، بحث بالكلمات المفتاحية فقط.")

# ---------------------------------------------------------------------------
# الذاكرة والمتجهات (مصدر الحقيقة الوحيد في الذاكرة + ملف محلي)
# ---------------------------------------------------------------------------
memory_lock = threading.Lock()
memory_lines = []          # قائمة السطور النصية
memory_embeddings = None   # متجهات مطابقة لـ memory_lines (إن فعّل التضمين)
dirty = False              # تغييرات غير مرفوعة لـ GitHub
sync_event = threading.Event()

SYSTEM_PROMPT = (
    "أنت وكيل ذكاء اصطناعي ذكي ومخصص لتعلم وإدارة مهام المستخدم. تتحدث العربية بطلاقة وبدقة. "
    "هذه ذاكرتك وسجلاتك السابقة؛ استخدمها لمعرفة تفضيلات المستخدم وتاريخ تفاعلك معه:\n"
    "{context}\n"
    "الآن أجب على رسالة المستخدم الحالية بشكل مفيد ومختصر."
)


def _recompute_embeddings():
    global memory_embeddings
    if embedder is not None and memory_lines:
        memory_embeddings = embedder.encode(
            memory_lines, convert_to_tensor=True, show_progress_bar=False
        )
    else:
        memory_embeddings = None


def load_knowledge():
    """تحميل الذاكرة: محلياً إن وُجدت، وإلا من GitHub، وإلا إنشاء سجل أولي."""
    global memory_lines, dirty
    if os.path.exists(KNOWLEDGE_FILE) and os.path.getsize(KNOWLEDGE_FILE) > 0:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            memory_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        print(f"تم تحميل الذاكرة محلياً: {len(memory_lines)} سطر.")
        return

    fetched = False
    if GITHUB_TOKEN and GITHUB_REPO:
        try:
            url = (
                f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{KNOWLEDGE_FILE}"
            )
            res = requests.get(
                url, headers={"Authorization": f"token {GITHUB_TOKEN}"}, timeout=20
            )
            if res.status_code == 200 and res.text.strip():
                memory_lines = [
                    ln.rstrip("\n") for ln in res.text.splitlines() if ln.strip()
                ]
                with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
                    f.write(res.text)
                fetched = True
                print(f"تم استعادة الذاكرة من GitHub: {len(memory_lines)} سطر.")
        except Exception as e:
            print(f"خطأ في جلب الذاكرة من GitHub: {e}")

    if not fetched:
        seed = "البداية: هذا هو السجل الأساسي لعقل الذكاء الاصطناعي المخصص لك."
        memory_lines = [seed]
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            f.write(seed + "\n")
        dirty = True
        sync_event.set()
        print("تم إنشاء ملف ذاكرة جديد.")

    _recompute_embeddings()


def retrieve_context(query: str, top_k: int = TOP_K_RETRIEVE, recent: int = 4) -> str:
    """استرجاع السياق: بحث دلالي (RAG) + أحدث السجلات لضمان سياق المحادثة."""
    if not memory_lines:
        return ""

    parts = []

    if embedder is not None and memory_embeddings is not None:
        q_emb = embedder.encode(query, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(q_emb, memory_embeddings)[0].cpu().numpy()
        k = min(top_k, len(memory_lines))
        top_idx = np.argsort(-scores)[:k]
        for i in sorted(set(int(x) for x in top_idx)):
            parts.append(memory_lines[i])
    else:
        # بديل خفيف: تطابق الكلمات المفتاحية
        q_words = set(query.lower().split())
        scored = []
        for i, ln in enumerate(memory_lines):
            score = len(q_words & set(ln.lower().split()))
            scored.append((score, i))
        scored.sort(reverse=True)
        idxs = [i for s, i in scored[:top_k] if s > 0]
        parts = [memory_lines[i] for i in sorted(idxs)]

    for ln in memory_lines[-recent:]:
        if ln not in parts:
            parts.append(ln)

    return "\n".join(parts)


def append_memory(entry: str):
    """إضافة سطر ذاكرة جديد: للقائمة + المتجه + الملف المحلي + إشعار المزامنة."""
    global memory_embeddings, dirty
    with memory_lock:
        memory_lines.append(entry)
        if embedder is not None:
            new_emb = embedder.encode(
                [entry], convert_to_tensor=True, show_progress_bar=False
            )
            if memory_embeddings is None:
                memory_embeddings = new_emb
            else:
                memory_embeddings = torch.cat([memory_embeddings, new_emb], dim=0)
        dirty = True
        with open(KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    sync_event.set()


# ---------------------------------------------------------------------------
# المزامنة مع GitHub في خيط خلفي (مع تجميع/تأجيل لتجنب ضرب الـ API)
# ---------------------------------------------------------------------------
def _sync_loop():
    global dirty
    while True:
        sync_event.wait()
        time.sleep(SYNC_DEBOUNCE_SEC)          # تجميع الرسائل المتتالية
        sync_event.clear()
        if not (GITHUB_TOKEN and GITHUB_REPO):
            continue
        with memory_lock:
            if not dirty:
                continue
            content = "\n".join(memory_lines) + "\n"
            dirty = False

        try:
            api_url = (
                f"https://api.github.com/repos/{GITHUB_REPO}/contents/{KNOWLEDGE_FILE}"
            )
            headers = {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
            }
            get_res = requests.get(api_url, headers=headers, timeout=20)
            sha = get_res.json().get("sha") if get_res.status_code == 200 else None

            payload = {
                "message": f"AI Self-Learning: update knowledge base ({datetime.now():%Y-%m-%d %H:%M})",
                "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            }
            if sha:
                payload["sha"] = sha

            put_res = requests.put(api_url, headers=headers, json=payload, timeout=30)
            if put_res.status_code in (200, 201):
                print("تم رفع الذاكرة إلى GitHub بنجاح.")
            else:
                print(f"فشل الرفع إلى GitHub: {put_res.status_code} {put_res.text[:200]}")
                with memory_lock:
                    dirty = True  # إعادة المحاولة لاحقاً
        except Exception as e:
            print(f"فشل مزامنة GitHub: {e}")
            with memory_lock:
                dirty = True


threading.Thread(target=_sync_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# معالجات أوامر تليجرام
# ---------------------------------------------------------------------------
@bot.message_handler(commands=["start"])
def cmd_start(message):
    bot.reply_to(
        message,
        "وعليكم السلام ورحمة الله! أنا بوتك الذكي BOT-22 🧠\n"
        "أتعلم من كل تفاعل وأحفظ ذاكرتي بشكل دائم على GitHub.\n"
        "أرسل لي أي سؤال أو أمر.",
    )


@bot.message_handler(commands=["status"])
def cmd_status(message):
    with memory_lock:
        n = len(memory_lines)
        d = dirty
    bot.reply_to(
        message,
        f"الحالة:\n• السطور في الذاكرة: {n}\n• النموذج: {GROQ_MODEL}\n"
        f"• تغييرات غير مرفوعة: {'نعم' if d else 'لا'}\n• التضمين الدلالي: {'مفعّل' if embedder else 'معطّل'}",
    )


@bot.message_handler(commands=["reset"])
def cmd_reset(message):
    global memory_embeddings, dirty
    with memory_lock:
        memory_lines.clear()
        memory_embeddings = None
        seed = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] تم إعادة ضبط الذاكرة بأمر من المستخدم."
        memory_lines.append(seed)
        if embedder is not None:
            memory_embeddings = embedder.encode(
                [seed], convert_to_tensor=True, show_progress_bar=False
            )
        dirty = True
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            f.write(seed + "\n")
    sync_event.set()
    bot.reply_to(message, "تم مسح الذاكرة وبدء سجل جديد ✅")


@bot.message_handler(
    content_types=["photo", "sticker", "video", "audio", "voice", "document"]
)
def handle_non_text(message):
    bot.reply_to(message, "أستقبل النصوص فقط حالياً 🙂 أرسل لي رسالة نصية.")


@bot.message_handler(func=lambda m: True, content_types=["text"])
def handle_message(message):
    chat_id = message.chat.id
    user_text = (message.text or "").strip()
    if not user_text:
        return

    try:
        bot.send_chat_action(chat_id, "typing")
    except Exception:
        pass

    context = retrieve_context(user_text)

    if client is None:
        bot.reply_to(message, "عذراً، لم يتم ضبط GROQ_API_KEY بعد.")
        return

    try:
        kwargs = dict(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": user_text},
            ],
            temperature=0.6,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            top_p=0.95,
        )
        if GROQ_REASONING_EFFORT:
            kwargs["reasoning_effort"] = GROQ_REASONING_EFFORT

        response = client.chat.completions.create(**kwargs)
        ai_reply = response.choices[0].message.content or "(لا يوجد رد)"
    except Exception as e:
        bot.reply_to(message, f"عذراً، حدث خطأ في Groq: {str(e)[:300]}")
        return

    # حفظ التعلم في الذاكرة (مقصوص لطول معقول)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] User: {user_text[:200]} | AI: {ai_reply[:300]}"
    append_memory(entry)

    bot.reply_to(message, ai_reply)


# ---------------------------------------------------------------------------
# نقاط واجهة FastAPI
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    load_knowledge()
    if TELEGRAM_BOT_TOKEN and WEBHOOK_URL:
        webhook = WEBHOOK_URL.rstrip("/") + "/webhook"
        try:
            bot.remove_webhook()
            bot.set_webhook(url=webhook)
            print(f"تم ضبط الويب هوك: {webhook}")
        except Exception as e:
            print(f"فشل ضبط الويب هوك: {e}")


@app.post("/webhook")
async def telegram_webhook(request: Request):
    if not TELEGRAM_BOT_TOKEN:
        return {"status": "bot not configured"}
    try:
        payload = await request.json()
    except Exception:
        return {"status": "bad request"}
    update = telebot.types.Update.de_json(payload)
    if update is not None:
        # المعالج متزامن؛ نشغّله في خيط حتى لا نحظر event loop ونرد بسرعة على تليجرام
        threading.Thread(
            target=bot.process_new_updates, args=([update],), daemon=True
        ).start()
    return {"status": "ok"}


@app.get("/")
def home():
    return {
        "status": "BOT-22 AI Agent with Groq & GitHub Memory is running",
        "model": GROQ_MODEL,
        "memory_lines": len(memory_lines),
        "embeddings": bool(embedder),
    }


@app.get("/health")
def health():
    return {"ok": True}
