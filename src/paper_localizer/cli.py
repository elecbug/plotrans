"""Command line entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

from .core import (OLLAMA_URL, TRANSLATION_MODEL,
                   OllamaClient, ProcessingError, extract_pdf, process_pdf)
from .ocr import OcrError, OcrOptions, model_paths


def doctor(url: str, translation_model: str, *, ocr: bool = False, skip_ollama: bool = False,
           vision_model: str = "") -> int:
    failures = 0
    try:
        version = importlib.metadata.version("opendataloader-pdf")
        print(f"[OK] OpenDataLoader PDF {version}")
    except importlib.metadata.PackageNotFoundError:
        print("[ERROR] opendataloader-pdf is not installed: pip install -e .")
        failures += 1
    java = shutil.which("java")
    if java:
        try:
            result = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"[OK] Java: {(result.stderr or result.stdout).splitlines()[0]}")
            else:
                print("[ERROR] Java failed to run.")
                failures += 1
        except (OSError, subprocess.TimeoutExpired):
            print("[ERROR] Java failed to run or did not respond within 10 seconds.")
            failures += 1
    else:
        print("[ERROR] Java 11+ was not found on PATH.")
        failures += 1
    if ocr:
        try:
            model_paths()
            print("[OK] Local CPU OCR: RapidOCR 3.10.0 / ONNX Runtime 1.23.2; 3 bundled model checksums verified")
        except OcrError as exc:
            print(f"[ERROR] {exc}"); failures += 1
    if not skip_ollama:
        try:
            OllamaClient(url, timeout=10).check_models(translation_model)
            print(f"[OK] Ollama translation model: {translation_model}")
        except ProcessingError as exc:
            print(f"[ERROR] {exc}")
            failures += 1
    if vision_model:
        try:
            OllamaClient(url, timeout=10).check_models(vision_model)
            print(f"[OK] Local vision OCR model: {vision_model}")
        except ProcessingError as exc:
            print(f"[ERROR] {exc}"); failures += 1
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse and translate PDF papers locally.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Parse and translate a PDF into Markdown, HTML, and ZIP")
    run.add_argument("pdf", type=Path)
    run.add_argument("-o", "--output", type=Path)
    run.add_argument("--translation-model", default=TRANSLATION_MODEL)
    run.add_argument("--target-language", default="English")
    run.add_argument("--chunk-size", type=int, default=1800,
                     help="Target document chunk size; complete sentences may exceed it.")
    run.add_argument("--ollama-url", default=OLLAMA_URL)
    extract = sub.add_parser("extract", help="Extract original Markdown, HTML, and ZIP with optional local document OCR")
    extract.add_argument("pdf", type=Path)
    extract.add_argument("-o", "--output", type=Path)
    extract.add_argument("--chunk-size", type=int, default=1800,
                         help="Target document chunk size; complete sentences may exceed it.")
    for command in (run, extract):
        command.add_argument("--ocr-mode", choices=("auto","off","force"), default="auto")
        command.add_argument("--ocr-dpi", type=int, default=216)
        command.add_argument("--ocr-min-confidence", type=float, default=.85)
        command.add_argument("--vision-ocr-model", default="", help="Optional local document OCR model (e.g. glm-ocr:latest)")
        command.add_argument("--vision-ocr-url", default=OLLAMA_URL, help="Local Ollama URL for document OCR")
        command.add_argument("--vision-ocr-digest", default="", help="Verify the OCR model SHA-256 digest for reproducibility")
    check = sub.add_parser("doctor", help="Check dependencies and model availability")
    check.add_argument("--translation-model", default=TRANSLATION_MODEL)
    check.add_argument("--ollama-url", default=OLLAMA_URL)
    check.add_argument("--ocr", action="store_true", help="Verify local OCR models and checksums")
    check.add_argument("--skip-ollama", action="store_true", help="Check extraction dependencies without Ollama")
    check.add_argument("--vision-ocr-model", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor(args.ollama_url, args.translation_model,ocr=args.ocr,
                          skip_ollama=args.skip_ollama,vision_model=args.vision_ocr_model)
        ocr = OcrOptions(args.ocr_mode,args.ocr_dpi,args.ocr_min_confidence,
                         args.vision_ocr_model, args.vision_ocr_url, args.vision_ocr_digest)
        if args.command == "extract":
            output = args.output or Path("outputs") / f"{args.pdf.stem}.extracted.md"
            path = extract_pdf(args.pdf, output, chunk_size=args.chunk_size,ocr=ocr)
            print(f"Extraction complete: {path.resolve()}\nHTML: {path.with_suffix('.html').resolve()}\nZIP: {path.with_suffix('.zip').resolve()}")
            return 0
        output = args.output or Path("outputs") / f"{args.pdf.stem}.translated.md"
        path = process_pdf(
            args.pdf, output, translation_model=args.translation_model,
            target_language=args.target_language,
            chunk_size=args.chunk_size, ollama_url=args.ollama_url,
            ocr=ocr,
            progress=lambda stage, index, total: print(f"{stage}: {index}/{total}", flush=True),
        )
        print(f"Complete: {path.resolve()}\nHTML: {path.with_suffix('.html').resolve()}\nZIP: {path.with_suffix('.zip').resolve()}")
        return 0
    except (ProcessingError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
