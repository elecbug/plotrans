"""Local Streamlit interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from paper_localizer.core import (OLLAMA_URL, SUMMARY_MODEL, TRANSLATION_MODEL,
                                  ProcessingError, process_pdf)
from paper_localizer.bundle import FIGURE_LINK, build_bundle, figure_path, inline_equation_preview
from paper_localizer.markup import image_reference


def main() -> None:
    st.set_page_config(page_title="논문 로컬 번역기", page_icon="📄", layout="wide")
    st.title("논문 로컬 번역기")
    st.caption("OpenDataLoader PDF + Ollama · PDF 분석, 요약, 원문/번역 병렬 Markdown")
    uploaded = st.file_uploader("논문 PDF 선택", type="pdf")
    with st.expander("설정", expanded=False):
        translation_model = st.text_input("번역 모델", TRANSLATION_MODEL)
        summary_model = st.text_input("요약 모델", SUMMARY_MODEL)
        target_language = st.text_input("번역 언어", "Korean")
        ollama_url = st.text_input("Ollama 주소", OLLAMA_URL)
        chunk_size = st.slider("청크 크기 (문자)", 500, 3500, 1800, 100)
    if st.button("분석 및 번역", disabled=uploaded is None, type="primary") and uploaded:
        bar = st.progress(0, text="준비 중")

        def update(stage: str, index: int, total: int) -> None:
            bar.progress(min(index / max(total, 1), 1.0), text=f"{stage} · {index}/{total}")

        with tempfile.TemporaryDirectory(prefix="paper-localizer-upload-") as directory:
            source = Path(directory) / Path(uploaded.name).name
            source.write_bytes(uploaded.getvalue())
            destination = Path("outputs") / f"{source.stem}.ko.md"
            try:
                result = process_pdf(
                    source, destination, translation_model=translation_model,
                    summary_model=summary_model, target_language=target_language,
                    chunk_size=chunk_size, ollama_url=ollama_url, progress=update,
                )
            except (ProcessingError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state["markdown"] = result.read_bytes()
                st.session_state["filename"] = result.name
                st.session_state["bundle"] = build_bundle(result)
                st.session_state["result_path"] = str(result)
                st.success(f"완료: {result.resolve()}")
    if "markdown" in st.session_state:
        if "bundle" in st.session_state:
            st.download_button("Markdown + 그림 ZIP 다운로드", st.session_state["bundle"],
                               file_name=Path(st.session_state["filename"]).stem + ".zip",
                               mime="application/zip")
        st.download_button("Markdown 다운로드", st.session_state["markdown"],
                           file_name=st.session_state["filename"], mime="text/markdown")
        with st.expander("결과 미리보기"):
            markdown = st.session_state["markdown"].decode("utf-8")
            markdown = inline_equation_preview(Path(st.session_state["result_path"]), markdown)
            cursor = 0
            for match in FIGURE_LINK.finditer(markdown):
                st.markdown(markdown[cursor:match.start()], unsafe_allow_html=True)
                reference = image_reference(match)
                image_path = figure_path(Path(st.session_state["result_path"]), reference.source)
                st.image(str(image_path), caption=reference.alt)
                cursor = match.end()
            st.markdown(markdown[cursor:], unsafe_allow_html=True)


if __name__ == "__main__":
    main()
