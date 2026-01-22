import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st

# -----------------------------
# 1️⃣ تحميل نص الدومين
# -----------------------------
DOMAIN_FILE = r"C:\Users\abdul\Desktop\final\nlp_domain_guide.txt"

if not os.path.exists(DOMAIN_FILE):
    raise FileNotFoundError(f"{DOMAIN_FILE} غير موجود! تأكدي من أن الملف موجود في المسار.")

with open(DOMAIN_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"عدد الأحرف قبل التنظيف: {len(raw_text)}")

# -----------------------------
# 2️⃣ تنظيف النص
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\sء-ي]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_text = clean_text(raw_text)
print(f"عدد الأحرف بعد التنظيف: {len(cleaned_text)}")

# -----------------------------
# 3️⃣ تقسيم النص إلى قطع
# -----------------------------
def split_text(text, chunk_size=50):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

text_chunks = split_text(cleaned_text)
print(f"عدد القطع: {len(text_chunks)}")

if not text_chunks:
    raise ValueError("لم يتم استخراج أي نصوص من ملف الدومين. تأكدي من أن الملف ليس فارغًا.")

# -----------------------------
# 4️⃣ تحميل أو إنشاء نموذج Embeddings
# -----------------------------
MODEL_FOLDER = r"C:\Users\abdul\Desktop\models\all-MiniLM-L6-v2"

if os.path.exists(MODEL_FOLDER):
    print("تم العثور على النموذج محليًا. جاري التحميل...")
else:
    print("النموذج غير موجود محليًا. سيتم تنزيله تلقائيًا من الإنترنت...")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embed_model.save(MODEL_FOLDER)
    print(f"تم تنزيل النموذج وحفظه محليًا في: {MODEL_FOLDER}")

# تحميل النموذج
embed_model = SentenceTransformer(MODEL_FOLDER)

# -----------------------------
# 5️⃣ إنشاء Embeddings وVector DB
# -----------------------------
embeddings = embed_model.encode(text_chunks, convert_to_numpy=True)
print("شكل الـ Embeddings:", embeddings.shape)

dimension = embeddings.shape[1]  # عدد أبعاد الـ Embeddings
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("تم إنشاء Vector DB بنجاح!")

# -----------------------------
# 6️⃣ RAG – استرجاع المعلومات
# -----------------------------
def retrieve_answer(question, k=2):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    return " ".join(retrieved_chunks)

def generate_answer(question):
    context = retrieve_answer(question)
    answer = f"أنت مساعد ذكي مخصص لمادة NLP.\nاقرأ النص التالي بعناية:\n{context}\nثم أجب على السؤال:\n{question}"
    return answer

# -----------------------------
# 7️⃣ اختبار النظام
# -----------------------------
print("\n=== اختبار النظام ===\n")
test_questions = [
    "ما هو Tokenization؟",
    "ما الفرق بين NLP و Machine Learning؟"
]

for q in test_questions:
    ans = generate_answer(q)
    print("سؤال:", q)
    print("جواب:", ans, "\n")

# -----------------------------
# 8️⃣ واجهة Streamlit
# -----------------------------
st.title("مساعد مادة NLP – نظام RAG محلي")
question = st.text_input("اكتب سؤالك هنا:")
if question:
    with st.spinner("جارٍ معالجة السؤال..."):
        answer = generate_answer(question)
    st.write("**الإجابة:**", answer)

    # -----------------------------
# ✅ قائمة الأسئلة للاختبار
# -----------------------------
test_questions = [
    # داخل المجال – NLP
    "ما هو Tokenization في معالجة اللغة الطبيعية؟",
    "الفرق بين Word Embeddings وOne-Hot Encoding؟",
    "ما هو مفهوم Stop Words ولماذا نزيلها؟",
    "ما الفرق بين Stemming وLemmatization؟",
    "ما هو POS Tagging ولماذا نستخدمه؟",
    "ما معنى Named Entity Recognition (NER)؟",
    "الفرق بين Bag-of-Words وTF-IDF؟",
    "ما هو مفهوم Sequence-to-Sequence في NLP؟",
    "ما الفرق بين Classification وClustering في النصوص؟",
    "ما هو Transformer ولماذا يستخدم في NLP؟",
    "ما هو Contextual Embedding مثل BERT؟",
    "ما معنى Sentiment Analysis؟",
    "كيف نستخدم FAISS لإنشاء Vector Database؟",
    "ما هو Retrieval-Augmented Generation (RAG)؟",
    "الفرق بين NLP التقليدي وNLP القائم على التعلم العميق؟",

    # خارج المجال – General Knowledge
    "ما هو الفرق بين الذكاء الاصطناعي والذكاء البشري؟",
    "من هو مؤسس شركة مايكروسوفت؟",
    "ما هو قانون نيوتن الأول؟",
    "ما هي عاصمة اليابان؟",
    "من كتب رواية 'البؤساء'؟"
]

# -----------------------------
# ✅ كود لاختبار النظام على جميع الأسئلة
# -----------------------------
print("\n=== اختبار النظام على جميع الأسئلة ===\n")

for i, question in enumerate(test_questions, start=1):
    answer = generate_answer(question)  # هذه الدالة موجودة في كودك الحالي
    print(f"سؤال {i}: {question}")
    print(f"جواب {i}: {answer}\n")