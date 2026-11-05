"""Local Streamlit interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from paper_localizer.core import (OLLAMA_URL, SUMMARY_MODEL, TRANSLATION_MODEL,
                                  OllamaClient, ProcessingError, extract_pdf, process_pdf)
from paper_localizer.bundle import standalone_html
from paper_localizer.ocr import OcrOptions
from paper_localizer.vision import DEFAULT_VISION_MODEL


def main() -> None:
    st.set_page_config(page_title="PLoTrans", page_icon="📄", layout="wide")
    st.title("PLoTrans")
    st.caption("Turn PDF papers into HTML and Markdown, with optional translation and summaries.")
    mode = st.radio("Processing mode", ["Extract only", "Translate and summarize"], horizontal=True)
    translate = mode == "Translate and summarize"
    if translate:
        st.info("Extract the original paper, then add a translation and summary. Requires Ollama and both models.")
    else:
        st.info("Generate original HTML, Markdown, and ZIP without translation or summaries. Default settings work without Ollama.")
    uploaded = st.file_uploader("Choose a paper PDF", type="pdf")
    vision_choice = st.selectbox(
        "High quality document OCR", ["Disabled", DEFAULT_VISION_MODEL, "Custom model"],
        key="document_ocr_choice",
        help="Optionally transcribe tables and verify equations or OCR text with a local model. Installation is checked when processing starts.",
    )
    vision_model = ""
    if vision_choice == "Custom model":
        vision_model = st.text_input("OCR model name", placeholder=f"Example: {DEFAULT_VISION_MODEL}").strip()
    elif vision_choice != "Disabled":
        vision_model = vision_choice
    with st.expander("Settings", expanded=False, key="processing_settings"):
        if translate:
            translation_model = st.text_input("Translation model", TRANSLATION_MODEL)
            summary_model = st.text_input("Summary model", SUMMARY_MODEL)
            target_language = st.text_input("Target language", "English")
        chunk_size = st.slider("Chunk size (characters)", 500, 3500, 1800, 100)
        ocr_mode = st.selectbox("OCR mode", ["auto", "off", "force"],
                                help="auto: detect scans and unreadable text; off: disable OCR; force: process every nonblank page")
        ocr_dpi = st.slider("OCR resolution (DPI)", 96, 400, 216, 8)
        ollama_url = OLLAMA_URL
        if translate or vision_choice != "Disabled":
            ollama_url = st.text_input("Ollama URL", OLLAMA_URL)
    action = "Translate and summarize" if translate else "Extract and generate HTML"
    if st.button(action, disabled=uploaded is None, type="primary") and uploaded:
        if vision_choice == "Custom model" and not vision_model:
            st.warning("Enter a model name for document OCR.")
            st.stop()
        if vision_model:
            try:
                model_tag = vision_model if ":" in vision_model else f"{vision_model}:latest"
                OllamaClient(ollama_url, timeout=10).check_models(model_tag)
            except ProcessingError as exc:
                st.warning(f"Document OCR cannot start.\n\n{exc}")
                st.caption("Set up Ollama and the selected model, then retry, or set High quality document OCR to Disabled.")
                st.stop()
        bar = st.progress(0, text="Preparing")

        def update(stage: str, index: int, total: int) -> None:
            bar.progress(min(index / max(total, 1), 1.0), text=f"{stage} · {index}/{total}")

        with tempfile.TemporaryDirectory(prefix="paper-localizer-upload-") as directory:
            source = Path(directory) / Path(uploaded.name).name
            suffix = "translated" if translate else "extracted"
            destination = Path("outputs") / f"{source.stem}.{suffix}.md"
            try:
                source.write_bytes(uploaded.getvalue())
                ocr = OcrOptions(
                    mode=ocr_mode, dpi=ocr_dpi, vision_model=vision_model, vision_url=ollama_url,
                )
                if translate:
                    result = process_pdf(
                        source, destination, translation_model=translation_model,
                        summary_model=summary_model, target_language=target_language,
                        chunk_size=chunk_size, ollama_url=ollama_url, progress=update, ocr=ocr,
                    )
                else:
                    bar.progress(0, text="Extracting the original PDF and generating HTML")
                    with st.spinner("Extracting the original paper and generating HTML..."):
                        result = extract_pdf(source, destination, chunk_size=chunk_size, ocr=ocr)
                artifacts = {
                    "markdown": result.read_bytes(),
                    "filename": result.name,
                    "bundle": result.with_suffix(".zip").read_bytes(),
                    "html": result.with_suffix(".html").read_bytes(),
                    "result_path": str(result),
                    "result_mode": mode,
                }
            except (ProcessingError, ValueError, OSError) as exc:
                bar.empty()
                st.error(str(exc))
            else:
                st.session_state.update(artifacts)
                bar.progress(1.0, text="Complete")
                st.success(f"Complete ({mode}): {result.resolve()}")
    if "markdown" in st.session_state:
        st.subheader("Latest result")
        st.caption(f"{st.session_state.get('result_mode', '')} · {st.session_state['filename']}")
        if "bundle" in st.session_state:
            st.download_button("Download paper ZIP", st.session_state["bundle"],
                               file_name=Path(st.session_state["filename"]).stem + ".zip",
                               mime="application/zip", on_click="ignore")
        st.download_button("Download Markdown", st.session_state["markdown"],
                           file_name=st.session_state["filename"], mime="text/markdown", on_click="ignore")
        if "html" in st.session_state:
            st.download_button("Download HTML", st.session_state["html"],
                               file_name=Path(st.session_state["filename"]).with_suffix(".html").name,
                               mime="text/html", on_click="ignore")
        st.caption("Use the ZIP to move the document to another computer with its CSS and images.")
        with st.expander("HTML preview", expanded=True):
            html_path = Path(st.session_state["result_path"]).with_suffix(".html")
            st.iframe(standalone_html(html_path), height=1000, alt="Generated paper preview")


if __name__ == "__main__":
    main()
