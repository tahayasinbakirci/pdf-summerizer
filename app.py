import io
import os
import re
import time

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from pypdf import PdfReader

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
]

st.set_page_config(page_title="PDF Özetleyici", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
        .block-container { padding-top: 4rem; max-width: 720px; }
        h1 { text-align: center; }
        [data-testid="stFileUploader"] {
            display: flex;
            justify-content: center;
        }
        [data-testid="stFileUploader"] section {
            width: 100%;
            max-width: 480px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 PDF Özetleyici")
st.markdown(
    "<p style='text-align: center; color: #666;'>PDF dosyanızı yükleyin, yapay zeka sizin için özetlesin.</p>",
    unsafe_allow_html=True,
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def summarize_with_gemini(text: str, api_key: str) -> str:
    genai.configure(api_key=api_key)

    prompt = (
        "Aşağıdaki PDF metnini Türkçe olarak özetle. "
        "Özet net, yapılandırılmış ve ana fikirleri kapsasın. "
        "Gerektiğinde madde işaretleri kullan.\n\n"
        f"--- PDF METNİ ---\n{text}"
    )

    last_error = None
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        for attempt in range(2):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except ResourceExhausted as e:
                last_error = e
                delay_match = re.search(r"saniye:\s*(\d+)", str(e))
                time.sleep(int(delay_match.group(1)) + 1 if delay_match else 8)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    last_error = e
                    time.sleep(8)
                    continue
                raise

    raise last_error or RuntimeError("Özet oluşturulamadı.")


uploaded_file = st.file_uploader(
    "PDF dosyanızı seçin",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI_API_KEY bulunamadı. Lütfen .env dosyanıza ekleyin.")
        st.stop()

    with st.spinner("PDF okunuyor ve özetleniyor..."):
        try:
            pdf_bytes = uploaded_file.read()
            text = extract_text_from_pdf(pdf_bytes)

            if not text.strip():
                st.warning("PDF'den metin çıkarılamadı. Dosya taranmış bir görüntü olabilir.")
                st.stop()

            if len(text) > 100_000:
                text = text[:100_000] + "\n\n[... metin kısaltıldı ...]"

            summary = summarize_with_gemini(text, api_key)

            st.success(f"**{uploaded_file.name}** başarıyla özetlendi.")
            st.markdown("### Özet")
            st.markdown(summary)

        except Exception as e:
            error_text = str(e)
            if "429" in error_text or "quota" in error_text.lower():
                st.error(
                    "Gemini API kotası aşıldı veya bu model ücretsiz planda kullanılamıyor. "
                    "Birkaç dakika bekleyip tekrar deneyin veya `.env` dosyasında "
                    "`GEMINI_MODEL=gemini-1.5-flash` ayarlayın."
                )
            else:
                st.error(f"Bir hata oluştu: {e}")
