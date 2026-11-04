"""Local Streamlit interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from paper_localizer.core import (OLLAMA_URL, SUMMARY_MODEL, TRANSLATION_MODEL,
                                  ProcessingError, process_pdf)
from paper_localizer.bundle import standalone_html


def main() -> None:
    st.set_page_config(page_title="PLoTrans", page_icon="📄", layout="wide")
    st.title("PLoTrans")
    st.caption("PDF 논문을 원문·번역이 함께 있는 Markdown과 HTML로 정리합니다.")
    uploaded = st.file_uploader("논문 PDF 선택", type="pdf")
    with st.expander("설정", expanded=False):
        translation_model = st.text_input("번역 모델", TRANSLATION_MODEL)
        summary_model = st.text_input("요약 모델", SUMMARY_MODEL)
        target_language = st.text_input("번역 언어", "English")
        ollama_url = st.text_input("Ollama 주소", OLLAMA_URL)
        chunk_size = st.slider("청크 크기 (문자)", 500, 3500, 1800, 100)
    if st.button("분석 및 번역", disabled=uploaded is None, type="primary") and uploaded:
        bar = st.progress(0, text="준비 중")

        def update(stage: str, index: int, total: int) -> None:
            bar.progress(min(index / max(total, 1), 1.0), text=f"{stage} · {index}/{total}")

        with tempfile.TemporaryDirectory(prefix="paper-localizer-upload-") as directory:
            source = Path(directory) / Path(uploaded.name).name
            source.write_bytes(uploaded.getvalue())
            destination = Path("outputs") / f"{source.stem}.translated.md"
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
                st.session_state["bundle"] = result.with_suffix(".zip").read_bytes()
                st.session_state["html"] = result.with_suffix(".html").read_bytes()
                st.session_state["result_path"] = str(result)
                st.success(f"완료: {result.resolve()}")
    if "markdown" in st.session_state:
        if "bundle" in st.session_state:
            st.download_button("논문 패키지 ZIP 다운로드", st.session_state["bundle"],
                               file_name=Path(st.session_state["filename"]).stem + ".zip",
                               mime="application/zip")
        st.download_button("Markdown 다운로드", st.session_state["markdown"],
                           file_name=st.session_state["filename"], mime="text/markdown")
        if "html" in st.session_state:
            st.download_button("HTML 다운로드", st.session_state["html"],
                               file_name=Path(st.session_state["filename"]).with_suffix(".html").name,
                               mime="text/html")
        st.caption("다른 PC로 옮길 때는 CSS와 그림이 포함된 ZIP을 사용하세요.")
        with st.expander("결과 미리보기"):
            import streamlit.components.v1 as components
            html_path = Path(st.session_state["result_path"]).with_suffix(".html")
            components.html(standalone_html(html_path), height=1000, scrolling=True)


if __name__ == "__main__":
    main()
