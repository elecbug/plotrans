"""Command line entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

from .core import (OLLAMA_URL, SUMMARY_MODEL, TRANSLATION_MODEL,
                   OllamaClient, ProcessingError, extract_pdf, process_pdf)
from .ocr import OcrError, OcrOptions, model_paths


def doctor(url: str, translation_model: str, summary_model: str, *, ocr: bool = False, skip_ollama: bool = False) -> int:
    failures = 0
    try:
        version = importlib.metadata.version("opendataloader-pdf")
        print(f"[OK] OpenDataLoader PDF {version}")
    except importlib.metadata.PackageNotFoundError:
        print("[오류] opendataloader-pdf가 설치되지 않았습니다: pip install -e .")
        failures += 1
    java = shutil.which("java")
    if java:
        try:
            result = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"[OK] Java: {(result.stderr or result.stdout).splitlines()[0]}")
            else:
                print("[오류] Java 실행에 실패했습니다.")
                failures += 1
        except (OSError, subprocess.TimeoutExpired):
            print("[오류] Java 실행에 실패하거나 10초 안에 응답하지 않았습니다.")
            failures += 1
    else:
        print("[오류] Java 11+가 PATH에 없습니다.")
        failures += 1
    if ocr:
        try:
            model_paths()
            print("[OK] Local CPU OCR: RapidOCR 3.10.0 / ONNX Runtime 1.23.2; 3 bundled model checksums verified")
        except OcrError as exc:
            print(f"[오류] {exc}"); failures += 1
    if not skip_ollama:
        try:
            OllamaClient(url, timeout=10).check_models(translation_model, summary_model)
            print(f"[OK] Ollama 및 모델: {translation_model}, {summary_model}")
        except ProcessingError as exc:
            print(f"[오류] {exc}")
            failures += 1
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="논문 PDF를 로컬에서 요약·번역합니다.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="PDF를 Markdown·HTML 논문 패키지로 변환")
    run.add_argument("pdf", type=Path)
    run.add_argument("-o", "--output", type=Path)
    run.add_argument("--translation-model", default=TRANSLATION_MODEL)
    run.add_argument("--summary-model", default=SUMMARY_MODEL)
    run.add_argument("--target-language", default="English")
    run.add_argument("--chunk-size", type=int, default=1800)
    run.add_argument("--ollama-url", default=OLLAMA_URL)
    extract = sub.add_parser("extract", help="Ollama 없이 PDF 추출 결과만 확인")
    extract.add_argument("pdf", type=Path)
    extract.add_argument("-o", "--output", type=Path)
    extract.add_argument("--chunk-size", type=int, default=1800)
    for command in (run, extract):
        command.add_argument("--ocr-mode", choices=("auto","off","force"), default="auto")
        command.add_argument("--ocr-dpi", type=int, default=216)
        command.add_argument("--ocr-min-confidence", type=float, default=.85)
    check = sub.add_parser("doctor", help="설치 및 모델 상태 점검")
    check.add_argument("--translation-model", default=TRANSLATION_MODEL)
    check.add_argument("--summary-model", default=SUMMARY_MODEL)
    check.add_argument("--ollama-url", default=OLLAMA_URL)
    check.add_argument("--ocr", action="store_true", help="로컬 OCR 모델과 체크섬 확인")
    check.add_argument("--skip-ollama", action="store_true", help="Ollama 없이 추출 설치 상태만 확인")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor(args.ollama_url, args.translation_model, args.summary_model,ocr=args.ocr,skip_ollama=args.skip_ollama)
        ocr = OcrOptions(args.ocr_mode,args.ocr_dpi,args.ocr_min_confidence)
        if args.command == "extract":
            output = args.output or Path("outputs") / f"{args.pdf.stem}.extracted.md"
            path = extract_pdf(args.pdf, output, chunk_size=args.chunk_size,ocr=ocr)
            print(f"추출 완료: {path.resolve()}\nHTML: {path.with_suffix('.html').resolve()}\nZIP: {path.with_suffix('.zip').resolve()}")
            return 0
        output = args.output or Path("outputs") / f"{args.pdf.stem}.translated.md"
        path = process_pdf(
            args.pdf, output, translation_model=args.translation_model,
            summary_model=args.summary_model, target_language=args.target_language,
            chunk_size=args.chunk_size, ollama_url=args.ollama_url,
            ocr=ocr,
            progress=lambda stage, index, total: print(f"{stage}: {index}/{total}", flush=True),
        )
        print(f"완료: {path.resolve()}\nHTML: {path.with_suffix('.html').resolve()}\nZIP: {path.with_suffix('.zip').resolve()}")
        return 0
    except (ProcessingError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
