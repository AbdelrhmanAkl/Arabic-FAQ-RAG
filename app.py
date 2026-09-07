import streamlit as st
from pathlib import Path

from src.intent import ArabicFAQIntentClassifier
from src.retrieval import ArabicFAQRetriever
from src.reranking import IntentAwareReranker


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FAISS_PATH = BASE_DIR / "data" / "train_chunks.faiss"
METADATA_PATH = BASE_DIR / "data" / "train_chunks_metadata_v2.parquet"
INTENT_MODEL_PATH = BASE_DIR / "models" / "intent_classifier.joblib"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

RETRIEVAL_TOP_K = 10
FINAL_TOP_K = 5
RERANK_ALPHA = 0.03


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Arabic FAQ RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(99, 102, 241, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(14, 165, 233, 0.07),
                transparent 25%
            ),
            #f8fafc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hide Streamlit chrome ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* ---------- RTL ---------- */

    .rtl {
        direction: rtl;
        text-align: right;
    }

    /* ---------- Hero ---------- */

    .hero {
        padding: 2.2rem 2rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #1e293b 55%,
                #312e81 100%
            );
        box-shadow:
            0 20px 45px rgba(15, 23, 42, 0.16);
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        color: white;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.8;
        margin-top: 0.65rem;
        max-width: 850px;
    }

    /* ---------- Section titles ---------- */

    .section-title {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 750;
        margin-top: 1.4rem;
        margin-bottom: 0.7rem;
    }

    /* ---------- Search ---------- */

    .search-label {
        direction: rtl;
        text-align: right;
        color: #0f172a;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    div[data-testid="stTextArea"] textarea {
        direction: rtl !important;
        text-align: right !important;
        min-height: 125px !important;
        border-radius: 16px !important;
        border: 1px solid #cbd5e1 !important;
        background: white !important;
        font-size: 1.05rem !important;
        line-height: 1.9 !important;
        padding: 1rem !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow:
            0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    }

    /* ---------- Search button ---------- */

    div.stButton > button {
        width: 100%;
        min-height: 50px;
        border-radius: 14px;
        border: none;
        background: #4f46e5;
        color: white;
        font-size: 1rem;
        font-weight: 700;
        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }

    div.stButton > button:hover {
        background: #4338ca;
        color: white;
        transform: translateY(-1px);
        box-shadow:
            0 10px 24px rgba(79, 70, 229, 0.25);
    }

    /* ---------- Metric cards ---------- */

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.15rem;
        min-height: 115px;
        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.02rem;
        font-weight: 750;
        line-height: 1.5;
    }

    .metric-number {
        color: #4f46e5;
        font-size: 1.65rem;
        font-weight: 800;
    }

    /* ---------- Answer ---------- */

    .answer-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 0.8rem;
        box-shadow:
            0 10px 30px rgba(15, 23, 42, 0.06);
    }

    .answer-label {
        color: #4f46e5;
        font-size: 0.85rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }

    .answer-text {
        direction: rtl;
        text-align: right;
        color: #1e293b;
        font-size: 1.08rem;
        line-height: 2;
        font-weight: 500;
    }

    /* ---------- FAQ information ---------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 90px;
    }

    .info-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .info-value {
        color: #0f172a;
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.5;
    }

    /* ---------- Sources ---------- */

    .source-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 0.7rem;
    }

    .source-rank {
        color: #4f46e5;
        font-weight: 800;
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }

    .source-question {
        direction: rtl;
        text-align: right;
        color: #334155;
        font-size: 0.94rem;
        line-height: 1.8;
    }

    .source-score {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.45rem;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .sidebar-text {
        color: #94a3b8 !important;
        font-size: 0.88rem;
        line-height: 1.7;
    }

    .sidebar-divider {
        height: 1px;
        background: #334155;
        margin: 1.2rem 0;
    }

    .tech-item {
        color: #cbd5e1 !important;
        font-size: 0.85rem;
        margin: 0.45rem 0;
    }

    /* ---------- Footer ---------- */

    .custom-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD SYSTEM
# ============================================================

@st.cache_resource(show_spinner=False)
def load_system():

    intent_classifier = ArabicFAQIntentClassifier(
        model_path=INTENT_MODEL_PATH
    )

    retriever = ArabicFAQRetriever(
        faiss_path=str(FAISS_PATH),
        metadata_path=str(METADATA_PATH),
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    reranker = IntentAwareReranker(
        alpha=RERANK_ALPHA
    )

    return intent_classifier, retriever, reranker


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            🧠 Arabic FAQ RAG
        </div>
        <div class="sidebar-text">
            Intelligent Arabic FAQ question answering
            using semantic retrieval and intent-aware reranking.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tech-item">✓ Arabic Intent Classification</div>
        <div class="tech-item">✓ Multilingual E5 Embeddings</div>
        <div class="tech-item">✓ FAISS Vector Search</div>
        <div class="tech-item">✓ Intent-Aware Reranking</div>
        <div class="tech-item">✓ Top-K Source Retrieval</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-text">
            <b>Embedding Model</b><br>
            intfloat/multilingual-e5-base
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-text" style="margin-top:1rem;">
            <b>Retrieval Top-K</b><br>
            {RETRIEVAL_TOP_K}
        </div>

        <div class="sidebar-text" style="margin-top:1rem;">
            <b>Final Results</b><br>
            {FINAL_TOP_K}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero rtl">
        <div class="hero-badge">
            AI-Powered Arabic Question Answering
        </div>
        <div class="hero-title">
            Arabic FAQ RAG System
        </div>
        <div class="hero-subtitle">
            نظام ذكي للإجابة عن الأسئلة العربية باستخدام
            تصنيف النية، البحث الدلالي، واسترجاع أفضل المصادر
            مع إعادة ترتيب النتائج حسب الـ Intent.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD
# ============================================================

try:

    with st.spinner("جاري تحميل النظام..."):
        intent_classifier, retriever, reranker = load_system()

except Exception as e:

    st.error(
        f"حدث خطأ أثناء تحميل النظام: {e}"
    )

    st.stop()


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    '<div class="search-label">💬 اكتب سؤالك بالعربية</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Question",
    label_visibility="collapsed",
    placeholder="مثال: أريد إغلاق حسابي فوراً، ما هي الإجراءات المطلوبة؟",
    height=130,
)

search_clicked = st.button(
    "🔍 البحث عن الإجابة",
    use_container_width=True,
)


# ============================================================
# SEARCH PIPELINE
# ============================================================

if search_clicked:

    if not question.strip():

        st.warning(
            "من فضلك اكتب سؤالاً أولاً."
        )

        st.stop()

    try:

        with st.spinner(
            "جاري تحليل السؤال والبحث في قاعدة المعرفة..."
        ):

            # ------------------------------------------------
            # 1. Intent Classification
            # ------------------------------------------------

            predicted_intent = intent_classifier.predict(
                question
            )

            # ------------------------------------------------
            # 2. Semantic Retrieval
            # ------------------------------------------------

            retrieved = retriever.search(
                question,
                top_k=RETRIEVAL_TOP_K,
            )

            # ------------------------------------------------
            # 3. Intent-Aware Reranking
            # ------------------------------------------------

            reranked = reranker.rerank(
                retrieved,
                predicted_intent=predicted_intent,
                top_k=FINAL_TOP_K,
            )

        if not reranked:

            st.warning(
                "لم يتم العثور على نتائج مناسبة."
            )

            st.stop()

        best = reranked[0]

        # ====================================================
        # ANALYSIS
        # ====================================================

        st.markdown(
            '<div class="section-title rtl">🧠 تحليل السؤال</div>',
            unsafe_allow_html=True,
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.markdown(
                f"""
                <div class="metric-card rtl">
                    <div class="metric-label">
                        Question Type
                    </div>
                    <div class="metric-value">
                        {predicted_intent}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Top E5 Similarity
                    </div>
                    <div class="metric-number">
                        {best["e5_score"]:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Final Rerank Score
                    </div>
                    <div class="metric-number">
                        {best["final_score"]:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ====================================================
        # ANSWER
        # ====================================================

        st.markdown(
            '<div class="section-title rtl">💡 الإجابة</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="answer-card">
                <div class="answer-label">
                    BEST MATCHED ANSWER
                </div>
                <div class="answer-text">
                    {best["answer"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ====================================================
        # FAQ INFORMATION
        # ====================================================

        st.markdown(
            '<div class="section-title rtl">📌 تفاصيل أفضل نتيجة</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                f"""
                <div class="info-card rtl">
                    <div class="info-label">
                        الشركة
                    </div>
                    <div class="info-value">
                        {best["company"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="info-card rtl">
                    <div class="info-label">
                        الدولة
                    </div>
                    <div class="info-value">
                        {best["country"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:

            st.markdown(
                f"""
                <div class="info-card rtl">
                    <div class="info-label">
                        المجال
                    </div>
                    <div class="info-value">
                        {best["domain"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col4, col5, col6 = st.columns(3)

        with col4:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">
                        FAQ ID
                    </div>
                    <div class="info-value">
                        {best["id"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col5:

            intent_status = (
                "✓ نعم"
                if best["intent_match"]
                else "✗ لا"
            )

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">
                        Intent Match
                    </div>
                    <div class="info-value">
                        {intent_status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col6:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">
                        Chunk
                    </div>
                    <div class="info-value">
                        {best["chunk_idx"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ====================================================
        # SOURCES
        # ====================================================

        st.markdown(
            '<div class="section-title rtl">📚 المصادر المسترجعة</div>',
            unsafe_allow_html=True,
        )

        for item in reranked:

            intent_icon = (
                "✓"
                if item["intent_match"]
                else "—"
            )

            with st.expander(
                f'#{item["rerank_rank"]} · Score {item["final_score"]:.4f} · Intent {intent_icon}'
            ):

                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-rank">
                            Rerank #{item["rerank_rank"]}
                        </div>
                        <div class="source-question">
                            {item["question"]}
                        </div>
                        <div class="source-score">
                            E5 Score: {item["e5_score"]:.4f}
                            &nbsp; | &nbsp;
                            Final Score: {item["final_score"]:.4f}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="rtl">
                        <b>الشركة:</b> {item["company"]}<br>
                        <b>الدولة:</b> {item["country"]}<br>
                        <b>المجال:</b> {item["domain"]}<br>
                        <b>FAQ ID:</b> {item["id"]}<br>
                        <b>Intent:</b> {item["candidate_intent"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(
            f"حدث خطأ أثناء تنفيذ البحث: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">
        Arabic FAQ RAG System · Intent Classification +
        Multilingual Semantic Retrieval + Intent-Aware Reranking
    </div>
    """,
    unsafe_allow_html=True,
)