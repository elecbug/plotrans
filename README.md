# PLoTrans

Turn PDF papers into structured Markdown and HTML with OpenDataLoader PDF. Extract the original document on its own, or add translation with Tencent HY-MT2-7B and a summary from a separate local model. Documents include a title, abstract, table of contents, sections, figures, captions, and references. PDF content is sent only to local processors and Ollama.

## Components

- PDF extraction: `opendataloader-pdf==2.5.17` (local Java process)
- OCR: `rapidocr==3.10.0`, `onnxruntime==1.23.2` (local CPU), `opencv-python-headless==4.12.0.88`, `numpy==2.2.6`
- Figure extraction and rendering: `pypdf==6.10.0`, `pypdfium2==5.14.0`, `Pillow==12.3.0`
- Translation: `kaelri/hy-mt2:7b-q4_K_M`, Tencent's official `Hy-MT2-7B` GGUF weights with an Ollama template
- Summaries: `qwen3:4b`
- Web UI: `streamlit==1.66.1`
- Document rendering: `markdown-it-py==4.0.0`, `mdurl==0.1.2`
- Equation rendering: `latex2mathml==3.78.1` (MathML without external JavaScript)
- Python: 3.11 or later, 3.12 recommended
- Java: 11 or later, 17 recommended
- Default translation language: English; change it with `--target-language`
- Documentation, interface labels, and document navigation are in English. Figures use the label `Fig.`. Extracted content retains its original language.

The default translation model tag is listed on the [HY-MT2 Ollama distribution page](https://ollama.com/kaelri/hy-mt2). Tencent's [official model page](https://huggingface.co/tencent/Hy-MT2-7B) provides the translation prompt and recommended inference settings. To use another installed tag, set `--translation-model` in the CLI or change the web UI settings.

## Option 1: Docker Compose (Windows, macOS, Linux)

Requires Docker Desktop or Docker Engine with Compose and the project's `docker-compose.yaml`. The first run downloads images and models and requires internet access. Processing does not require external API keys.

```sh
docker compose up -d ollama
docker compose exec ollama ollama pull kaelri/hy-mt2:7b-q4_K_M
docker compose exec ollama ollama pull qwen3:4b
docker compose up --build -d app
```

Open <http://127.0.0.1:8501> and choose a **Processing mode**. The default, **Extract only**, generates the original document without translation or summaries: upload a PDF and click **Extract and generate HTML**. To include a translation and summary, select **Translate and summarize** and click the matching button.

Download the result with **Download paper ZIP**, or find the files in `outputs/`. The ZIP includes Markdown, HTML, shared CSS, and every referenced image. Stop the services with `docker compose down`; models remain in the Docker volume.

The Docker setup uses Ollama `0.40.14` and a Python `3.12` image. A GPU is optional, but the 7B translation model requires substantial memory and processing time. Adjust Docker's memory limit for your environment. Model downloads are unnecessary when only extracting with the default OCR settings.

## Option 2: Local installation

1. Install Python 3.11 or later and Java 11 or later. Check `python --version` and `java -version`. On Windows, open a new PowerShell session after changing PATH.
2. Create a virtual environment in this directory and install the packages.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui,ocr]"
```

**macOS / Linux**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ui,ocr]'
```

3. For translation and summaries, install [Ollama](https://ollama.com/download), start it, and download both models. Skip this step for extraction with the default settings.

```sh
ollama pull kaelri/hy-mt2:7b-q4_K_M
ollama pull qwen3:4b
paper-localizer doctor
```

If Ollama is not running, start `ollama serve` in a separate terminal. `doctor` checks Python packages, Java, the Ollama connection, and both models. For extraction dependencies alone, use `paper-localizer doctor --skip-ollama`.

4. Start the web UI or use the CLI.

```sh
python -m streamlit run src/paper_localizer/web.py
paper-localizer extract "path/to/paper.pdf" -o "outputs/paper.extracted.md"
paper-localizer run "path/to/paper.pdf" -o "outputs/paper.translated.md"
```

The web UI opens at <http://localhost:8501>. Run `paper-localizer extract --help` or `paper-localizer run --help` for CLI options. Use `--target-language` to change the translation language and `--chunk-size` to adjust the number of characters per model request.

`extract` generates the original document without translation or summaries. Specify a `.md` path with `-o`; matching `.html` and `.zip` files are created alongside it. Ollama is unnecessary unless you enable the optional document OCR model.

The web UI's **Extract only** mode follows the same extraction path, hides translation and summary settings, and writes `<paper>.extracted.md`, `.html`, and `.zip`. **Translate and summarize** writes `<paper>.translated.*`. Both modes provide downloads and an HTML preview. The latest result shows which mode produced it.

## Local OCR

The default `auto` mode detects pages with little or no text, or pages whose body text uses undecodable fonts, and runs OCR on the CPU. Pages with usable text retain their normal extraction path. No separate OCR server or system Tesseract installation is required.

RapidOCR 3.10.0 includes PP-OCRv6 small detection and recognition models and an orientation classifier in its wheel. Once the package is installed, no additional model download is needed. Before inference, the three ONNX files are checked against the SHA-256 values in `assets/ocr-models.json` and loaded from explicit local paths. Missing or mismatched models produce reinstallation instructions. See the [official RapidOCR installation guide](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/install/) for its distribution layout.

```sh
paper-localizer doctor --ocr --skip-ollama
paper-localizer extract "paper.pdf" -o outputs/paper.extracted.md --ocr-mode auto
paper-localizer extract "scan.pdf" -o outputs/scan.extracted.md --ocr-mode force --ocr-dpi 216
paper-localizer extract "paper.pdf" -o outputs/native.extracted.md --ocr-mode off
```

The default resolution is 216 DPI and the minimum confidence is 0.85. Adjust them with `--ocr-dpi` and `--ocr-min-confidence`. `force` processes every nonblank page; `off` preserves source images instead. Docker includes the OCR dependencies. For a CLI installation with OCR, use `pip install -e ".[ocr]"`.

OCR paragraphs use page coordinates to reconstruct reading order and columns. Recognized text is removed from the figure detection mask so an entire scanned page is not classified as one figure. Actual figures are cropped from the original, retaining their internal labels. Lines identified as equations and lines with low confidence are preserved as source images and excluded from translation requests; some surrounding prose on those lines may therefore remain untranslated. All original OCR pages are available in the collapsed **Original pages used for OCR** section of HTML and Markdown and are included in the ZIP.

Raw OCR results are cached in `.ocr-cache/` under the output directory, keyed by PDF content, engine version, and DPI. Paragraph layout and confidence filtering are reapplied when using the cache. OCR confidence is not an accuracy guarantee. The base OCR path does not automatically convert complex equations to LaTeX or reconstruct complex table cells. Local inference for this path was validated on the 13-page Kademlia paper and synthetic scanned PDFs.

### Optional document, table, and equation OCR

The local Ollama model `glm-ocr:latest` can additionally verify superscripts, subscripts, and prose containing equations that basic OCR may miss, and provide searchable transcriptions of image tables. HY-MT2-7B remains the translation model. GLM-OCR is a separate document recognition model; after downloading its weights, source content stays local. No additional PyTorch installation is required.

```sh
ollama pull glm-ocr:latest
paper-localizer doctor --ocr --skip-ollama --vision-ocr-model glm-ocr:latest
paper-localizer extract "paper.pdf" -o outputs/paper.extracted.md --vision-ocr-model glm-ocr:latest
paper-localizer run "paper.pdf" -o outputs/paper.translated.md --vision-ocr-model glm-ocr:latest
```

In the web UI, select **High quality document OCR** below the upload control:

- **Disabled** (default): use the standard extraction and OCR settings.
- **glm-ocr:latest**: enable additional table transcription and document/equation verification.
- **Custom model**: enter another compatible Ollama model name.

You can select a model before installing it. Availability is checked only when you click the processing button. If the model is missing, the UI shows a warning with the required `ollama pull` command and stops before processing the PDF. An unavailable Ollama server also produces a warning. Prepare Ollama and the model, then retry, or select **Disabled**. The app does not download models automatically or silently substitute another OCR mode.

For Docker, install the model with `docker compose exec ollama ollama pull glm-ocr:latest`. The selection applies to both processing modes. Without a GPU, the model runs on the CPU and processing takes longer. In the CLI, use `--vision-ocr-url` to select a separate OCR Ollama server. Base CPU extraction continues to work without these optional settings.

Decodable equations on a single line are first checked against source Unicode, effective character sizes, and baseline positions to preserve subscripts and superscripts. Other equations are rendered from the original PDF at 4x and 6x resolution, with an 8x check when results disagree. Only white padding is added outside each equation's owned region, avoiding neighboring characters. These transcriptions are compared across resolutions and against readable source digits and variables. Repeated short equations are compared with checked equations in the same PDF, using both whole images and individual glyphs.

Accepted OCR prose is restored as editable text with LaTeX equations, rendered offline as MathML in HTML and Markdown. Original equation images stay in position, and their recognized meaning is provided to translation and summary models. Recovered source regions remain available in a collapsed comparison section. Equations that fail agreement checks remain images and are marked so models do not guess their content. Agreement checks do not guarantee accuracy.

Tables whose text is drawn as vector outlines cannot be recovered through native text extraction. Their original images remain in the article, with captions above them. Optional OCR adds a collapsible **Searchable table transcription** only after two rendering resolutions agree on cells and spans, and independent CPU OCR supports at least 98% of long words and every plain numeric value. It checks 3x and 4.5x crops, with a 6x retry when needed. Install the `ocr` extra for this independent check. Equations inside cells still require visual review; consistency does not establish correct interpretation or cell assignment. Table images and transcriptions are not translated.

Malformed table markup and empty or truncated region responses retain their source representation and record the reason in the OCR report. Model setup and initial connection errors stop processing with setup instructions.

The `.ocr.json` output and ZIP record the actual model SHA-256 digest, preprocessing profile, region decisions, and equation/table transcriptions. `.vision-cache/` uses PDF content, model digest, region, resolution, and profile as its key. Use `--vision-ocr-digest` to verify the exact model version when reproducing a run. The model digest used for the documented experiment was:

```text
89d735dd9c07791953756edd275e5501464df61288b1cd5486c5c50c8c058b32
```

The implementation uses Ollama's `/api/generate`, following the [official GLM-OCR Ollama deployment guide](https://github.com/zai-org/GLM-OCR/blob/main/examples/ollama-deploy/README.md). Equation-specific [RapidLaTeXOCR](https://github.com/RapidAI/RapidLaTeXOCR) was also compared, but it produced incorrect bracket structures for complex piecewise functions and was not selected as the default.

```sh
python scripts/audit_corpus.py .tmp/example --output outputs/quality --only Kademlia,Bitcoin,Pastry --vision-ocr-model glm-ocr:latest --sheets
python scripts/benchmark_quality.py outputs/quality
```

The second command evaluates 21 semantic regression checks prepared against the source documents and six independent recalculations of the Bitcoin probability table. These pass rates are not overall paper accuracy. Actual translation and summary quality requires separate evaluation.

Reproduce the equation OCR comparison in a separate virtual environment to avoid mixing its NumPy/OpenCV versions with the base OCR installation. Use Python 3.12:

```sh
python -m venv .tmp/math-benchmark
# Windows: .tmp/math-benchmark/Scripts/activate
# macOS/Linux: source .tmp/math-benchmark/bin/activate
python -m pip install -r scripts/requirements-math-benchmark.txt
python scripts/compare_formula_ocr.py .tmp/example --download-models
```

The four models are verified against the SHA-256 values in `scripts/math-ocr-models.json`. Later comparisons can run offline without `--download-models`. Use `--scale 4` to compare the effect of source rendering resolution.

### Example papers

Place example PDFs in `.tmp/example/`, which is excluded from Git. The extraction path was checked with the following papers:

```sh
paper-localizer extract ".tmp/example/(2008) Bitcoin_ A Peer-to-Peer Electronic Cash System.pdf" -o outputs/bitcoin.extracted.md
paper-localizer extract ".tmp/example/(2001) Pastry_ Scalable, decentralized object location and routing for large-scale peer-to-peer systems.pdf" -o outputs/pastry.extracted.md
paper-localizer run ".tmp/example/(2008) Bitcoin_ A Peer-to-Peer Electronic Cash System.pdf" -o outputs/bitcoin.translated.md
```

Figure extraction found eight regions across the nine-page Bitcoin paper and ten figures across the 22-page Pastry paper. Use `run` for the full translation pipeline after installing Ollama and downloading both models.

## Outputs and resuming interrupted runs

The original abstract and generated summary are displayed separately. The body retains the paper's section and paragraph order. The HTML reading layout draws on IEEE Xplore: a left table of contents, title and author information, abstract, body, figures with captions in context, and references. The header reads `PLoTrans`. Translated documents provide **Original**, **Translation**, and **Parallel view** controls, print styles, and layouts for smaller screens. Markdown follows the same section order with collapsible translations.

Source glyphs and page geometry repair run-in headings and common two-column reading order, including full-width figures and tables. Heading promotion and spacing repairs require matching source characters. Authors, index terms, and small first-page notes retain separate roles. Continued source paragraphs are rejoined across page/column and model chunk boundaries. Figures, tables, and portraits reserve their space while loading, keeping document positions stable when navigating the table of contents. Complex layouts still need source review.

Web links stored in PDF annotations are retained. Printed targets are connected to a unique matching passage on the same source page. ORCID identifiers remain source metadata without guessing which author owns an icon. Other unmatched links remain in a collapsed list. Wrapped URL spaces and omitted DOI hyphens are repaired only when the PDF annotation and decoded source glyphs confirm the same target; source digits, case, and other punctuation must agree.

In numbered reference sections, hanging-indent continuation lines retain their reference's paragraph identity. URLs split between reference blocks are rejoined only when the complete target is confirmed by the source page. This keeps split publication years and URLs within the same entry and prevents a year such as `2001.` from becoming an HTML list number.

If translation is interrupted, a `.progress.json` checkpoint remains next to the output file. Rerun with the same PDF, models, settings, and output path to skip completed chunks. Successful completion removes the checkpoint. HTML files can be opened locally without a CDN or server.

## Separating figures from text

Alongside OpenDataLoader's text structure, the app renders an in-memory PDF with text display instructions removed to detect actual graphic regions. This preserves both JPG/PNG images and vector figures made from PDF shapes, arrows, and charts as PNG files. The original PDF is never modified.

Text inside figures remains in the figure and is excluded from duplicate body extraction. Rotated axis titles and nearby labels are included in figure regions. Labels in PDF Form XObjects are checked against transformed page character coordinates. Short captions do not reduce figure width. Panels sharing a caption are grouped, while figures in different columns remain separate. Ordinary tables with a clear structure remain Markdown tables.

Repeated margin artwork is excluded before caption matching, preventing journal headers from joining nearby figures. Small detached drawing fragments are assigned to a nearby captioned figure when their positions support it. Author portraits beside degree biographies are kept separately from numbered figures.

Captions misclassified as headings or split across blocks are reconnected using fonts, positions, and line spacing. Captions stay intact across chunk boundaries and are attached to figures as complete units. Ordinary captions are normalized to `Fig.`; source image captions containing equations are preserved. Unmatched captions remain in the body.

When caption coordinates include panel labels such as `(a)` and `(b)`, the caption's top boundary is corrected to the actual `Fig.` text. Correction is deferred if raised mathematical symbols or fonts could be clipped. A table caption is not attached to a preceding figure solely because of proximity.

Outputs use a PDF content hash for figure subdirectories, preventing collisions between papers:

```text
outputs/
  paper.extracted.md
  paper.extracted.html
  paper.extracted.zip
  paper.translated.md
  paper.translated.html
  paper.translated.zip
  assets/
    article.css
  figs/
    <PDF-hash>/
      page-002-fig-01.png
      page-002-fig-02.png
      page-016-math-003.png
```

Markdown uses relative references such as `![Fig. p.2-1](figs/<PDF-hash>/page-002-fig-01.png)`. Extract the ZIP and retain its directory structure when moving documents to another computer. The web UI previews the same HTML document. Image paths are never sent to translation models, so models cannot alter them.

## Preserving equations

PDFs such as the Pastry paper use Type3 fonts without ToUnicode character mappings. Equations may be visible on the page but absent from extracted text. These equation fonts and separate mathematical font regions are rendered as small PNG crops and inserted into the text. They share the `figs/<PDF-hash>/` directory with figures and are included in the ZIP.

For example, `b=4`, `|L|=16`, and `|M|=32` in the Fig.7 caption are preserved as source equation images. Character coordinates group mathematical fonts with ordinary-font digits, operators, brackets, function names, and subscripts belonging to the same expression. This reduces partial image fragmentation of expressions such as `O(log N)`.

Inline equations use HTML tags such as `<img class="inline-math" ...>`. Their `em` height is based on surrounding text size, and vertical alignment follows the PDF baseline. Explicit `width` and `height` attributes keep them small in viewers that strip CSS. The same tags are restored in the original and translation. Use a Markdown viewer that supports HTML.

If extraction omits an entire equation line, a standalone equation image is inserted. Regions with uncertain correspondence appear once as a source image. When a piecewise equation spans multiple text blocks, other blocks covered by the image are also cleaned up. Images are expanded to preserve content crossing region boundaries before duplicate text is removed. Text belonging to figure regions is also removed from the body using coordinate checks.

Translation requests replace inline equations with markers such as `[[MATH_0001]]`. Each marker must appear exactly once in the response before its original image is restored. Missing or duplicated markers trigger one retry; persistent failure raises an error instead of saving a translation with missing equations. Chunk boundaries never split image paths.

The base path preserves equations visually without automatic LaTeX conversion or semantic recognition. Summary models cannot read equations preserved only as images. Enable optional document OCR to provide validated transcriptions where available.

## Limitations and review

- The default OpenDataLoader path suits PDFs with text. Local OCR supplements scans and undecodable pages. Figures retain their original appearance; internal labels are not translated, and complex layouts are not analyzed semantically.
- Pages made entirely of undecodable fonts, such as those in Kademlia, use the default OCR path to recover prose. If OCR is disabled or recognizes no text, a source image and explanation are retained. Compare equations and proper names with the source pages.
- Figure extraction uses rules based on pixels, coordinates, and captions. Review documents with overlapping figures and text or patterned backgrounds against their originals.
- Native tables with usable cells become Markdown tables. Other tables retain source images; optional document OCR can add checked searchable transcriptions. Check merged cells, row/column assignments, equations, and important values against the PDF.
- Translations and summaries are generated by local models. Verify important numbers, equations, and citations against the original.
- `outputs/` may contain original and translated paper content. Review it before sharing.

## Development checks

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
python scripts/audit_corpus.py .tmp/example --output outputs/review --sheets
```

An integration test runs the real bundled OCR models offline. It is skipped by default. In PowerShell, set `$env:PLOTRANS_TEST_OCR="1"` before running the tests. On macOS/Linux, use `PLOTRANS_TEST_OCR=1 python -m unittest discover -s tests -v`.

The corpus audit processes all PDFs in a directory and produces per-paper Markdown, HTML, and ZIP files, source/figure/caption comparison sheets, and `audit-report.html`, `.md`, and `.json`. It checks image files and internal/source links, text duplicated by equation images, missing or unmatched captions, undecodable characters, and page image fallbacks. OpenDataLoader JSON is cached in `.scratch/corpus/native/` using PDF content, engine version, and extraction profile, so changed PDFs and different papers sharing a filename do not reuse each other's parsing data. Automated checks do not guarantee semantic completeness, so review the comparison sheets alongside the original.

For a page-by-page comparison, pass the processed audit JSON and its Markdown export:

```sh
python scripts/review_comparison.py .tmp/example/paper.pdf outputs/review/paper.audit.json outputs/review/paper.md --output outputs/review/page-comparison
```

Open `page-comparison/review.html` for all source pages alongside their extracted content. `review.json` checks decodable letter/digit preservation in the processed structure or retained images and identifies missing or blank assets. Caption abbreviations are counted separately. This check does not establish semantic accuracy, reading order, complete graphics, or correct OCR.

The documented audit of 18 example papers validated extraction without running Ollama inference. Tests cover duplicated piecewise equations, split captions, multiple panels across two columns, Form XObject coordinates, subscripts, undecodable-font preservation, and output bundles.

Dependency versions are pinned in `pyproject.toml`. Use `ollama list` to record actual model hashes. Environments requiring identical model bytes should also retain model file hashes and Docker image digests.
