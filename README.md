# PLoTrans

논문 PDF를 OpenDataLoader PDF로 구조화하고, Tencent HY-MT2-7B로 번역하며, 별도의 로컬 요약 모델로 핵심 내용을 정리합니다. 제목·초록·목차·절·그림 및 캡션·참고문헌을 갖춘 Markdown과 HTML을 함께 생성합니다. PDF 내용은 로컬 처리기와 Ollama에만 전달됩니다.

## 구성

- PDF 추출: `opendataloader-pdf==2.5.17` (로컬 Java 프로세스)
- OCR: `rapidocr==3.10.0`, `onnxruntime==1.23.2` (로컬 CPU), `opencv-python-headless==4.12.0.88`, `numpy==2.2.6`
- 그림 분리/렌더링: `pypdf==6.10.0`, `pypdfium2==5.14.0`, `Pillow==12.3.0`
- 번역: Tencent `Hy-MT2-7B` 공식 GGUF 가중치를 Ollama용 템플릿으로 감싼 `kaelri/hy-mt2:7b-q4_K_M`
- 요약: `qwen3:4b`
- 화면: `streamlit==1.66.1`
- 문서 렌더링: `markdown-it-py==4.0.0`, `mdurl==0.1.2`
- Python: 3.11 이상, 권장 3.12 / Java: 11 이상, 권장 17
- 기본 번역 언어: 영어 (`--target-language`로 변경 가능)
- 출력 문서의 제목·메뉴·설명·그림 표기는 영어로 제공하며, 그림은 `Fig.`로 표기합니다.

기본 모델 태그는 [HY-MT2 Ollama 배포 페이지](https://ollama.com/kaelri/hy-mt2)에서 확인할 수 있습니다. Tencent의 [공식 모델 페이지](https://huggingface.co/tencent/Hy-MT2-7B)는 번역 프롬프트와 권장 추론 설정을 제공합니다. 다른 태그를 이미 보유하셨다면 CLI의 `--translation-model` 또는 화면의 설정에서 지정할 수 있습니다.

## 방법 1: Docker Compose (Windows, macOS, Linux)

Docker Desktop 또는 Docker Engine/Compose가 필요합니다. 최초 실행 때 이미지와 모델을 다운로드하므로 인터넷이 필요합니다. 이후 처리에는 외부 API 키가 필요하지 않습니다.

```sh
docker compose up -d ollama
docker compose exec ollama ollama pull kaelri/hy-mt2:7b-q4_K_M
docker compose exec ollama ollama pull qwen3:4b
docker compose up --build -d app
```

브라우저에서 <http://127.0.0.1:8501>을 열고 PDF를 선택한 다음 **분석 및 번역**을 누르세요. 완료된 문서는 화면의 **논문 패키지 ZIP 다운로드**로 함께 내려받거나 프로젝트의 `outputs/` 폴더에서 찾을 수 있습니다. ZIP에는 Markdown, HTML, 공통 CSS와 모든 이미지가 포함됩니다. 종료 명령은 `docker compose down`입니다. 모델은 Docker 볼륨에 유지됩니다.

Docker 설정은 Ollama `0.40.14`와 Python `3.12` 기반 이미지를 사용합니다. GPU 없이도 실행할 수 있지만 7B 모델은 상당한 메모리와 시간이 필요합니다. 환경에 따라 Docker의 메모리 한도를 늘려 주세요.

## 방법 2: 직접 설치

1. Python 3.11 이상, Java 11 이상, [Ollama](https://ollama.com/download)를 설치합니다. 터미널에서 `python --version`, `java -version`, `ollama --version`을 확인하세요. Windows에서는 새 PowerShell을 열어 PATH 변경을 반영하세요.
2. 이 폴더에서 가상 환경과 Python 패키지를 설치합니다.

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

3. Ollama가 실행 중인지 확인하고 모델을 내려받습니다.

```sh
ollama pull kaelri/hy-mt2:7b-q4_K_M
ollama pull qwen3:4b
paper-localizer doctor
```

Ollama가 실행 중이지 않으면 별도 터미널에서 `ollama serve`를 실행하세요. `doctor`는 Python 패키지, Java, Ollama 연결과 두 모델을 확인합니다.

4. 웹 화면 또는 CLI로 처리합니다.

```sh
python -m streamlit run src/paper_localizer/web.py
paper-localizer run "path/to/paper.pdf" -o "outputs/paper.translated.md"
paper-localizer extract "path/to/paper.pdf" -o "outputs/paper.extracted.md"
```

웹 화면은 <http://localhost:8501>에서 열립니다. CLI는 `paper-localizer run --help`로 옵션을 볼 수 있습니다. `--target-language`로 번역 언어를 바꾸고 `--chunk-size`로 모델에 보내는 문자 수를 조정할 수 있습니다.
`extract`는 Ollama 없이 OpenDataLoader의 추출 결과를 먼저 검토할 때 사용합니다.

## 로컬 OCR

`auto` 모드가 기본입니다. 텍스트가 없거나 매우 적은 페이지, 본문 전체가 해독 불가능한 글꼴인 페이지를 감지해 CPU에서 OCR을 수행합니다. 정상적으로 추출되는 페이지는 기존 경로를 유지합니다. 별도 OCR 서버나 시스템 Tesseract 설치가 필요하지 않습니다.

RapidOCR 3.10.0 wheel에 포함된 PP-OCRv6 small 탐지·인식 모델과 방향 분류 모델을 사용합니다. 설치 시 패키지를 다운로드한 뒤에는 모델 파일을 추가로 받지 않습니다. 실행 전에 세 ONNX 파일을 `assets/ocr-models.json`의 SHA-256과 대조하고 명시적인 로컬 경로로 로드합니다. 모델 파일이 없거나 다르면 재설치 방법을 안내합니다. [RapidOCR 공식 설치 안내](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/install/)에서 배포 구성을 확인할 수 있습니다.

```sh
paper-localizer doctor --ocr --skip-ollama
paper-localizer extract "paper.pdf" -o outputs/paper.extracted.md --ocr-mode auto
paper-localizer extract "scan.pdf" -o outputs/scan.extracted.md --ocr-mode force --ocr-dpi 216
paper-localizer extract "paper.pdf" -o outputs/native.extracted.md --ocr-mode off
```

기본 해상도는 216 DPI, 최소 신뢰도는 0.85이며 `--ocr-dpi`와 `--ocr-min-confidence`로 조정합니다. `force`는 모든 비어 있지 않은 페이지를 OCR 처리하고, `off`는 이전의 원문 이미지 보존 방식을 사용합니다. Docker에는 OCR 의존성이 포함되며 CLI만 필요하면 `pip install -e ".[ocr]"`로 설치할 수 있습니다.

OCR 문단은 페이지 좌표를 사용해 읽기 순서와 단을 재구성합니다. 그림 탐색 마스크에서 OCR 글자를 지워 스캔 페이지 전체가 하나의 그림으로 분류되는 현상을 방지합니다. 실제 그림은 원본에서 잘라 내므로 그림 내부 레이블은 유지합니다. 수식으로 판단한 행과 신뢰도가 낮은 행은 전체를 원문 이미지로 보존하고 번역 모델에 보내지 않습니다. 따라서 해당 행의 문장 일부는 번역되지 않을 수 있습니다. HTML과 Markdown의 접힌 **Original pages used for OCR** 영역에서 모든 OCR 원본 페이지를 대조할 수 있으며 ZIP에도 포함됩니다.

원시 OCR 결과는 출력 폴더의 `.ocr-cache/`에 PDF 내용·엔진 버전·DPI를 기준으로 저장합니다. 문단 배치와 신뢰도 기준은 캐시를 사용해도 다시 적용합니다. OCR 신뢰도는 정확도 보장이 아니며, 복잡한 수식의 자동 LaTeX 변환과 복잡한 표의 셀 복원은 수행하지 않습니다. 이 경로는 Kademlia 13쪽과 합성 스캔 PDF로 실제 로컬 추론을 검증했습니다.

### 포함된 예시로 확인

`.tmp/example/`에 논문 PDF를 넣으면 Git에서 제외됩니다. 아래 두 논문으로 추출 경로를 확인했습니다.

```sh
paper-localizer extract ".tmp/example/(2008) Bitcoin_ A Peer-to-Peer Electronic Cash System.pdf" -o outputs/bitcoin.extracted.md
paper-localizer extract ".tmp/example/(2001) Pastry_ Scalable, decentralized object location and routing for large-scale peer-to-peer systems.pdf" -o outputs/pastry.extracted.md
paper-localizer run ".tmp/example/(2008) Bitcoin_ A Peer-to-Peer Electronic Cash System.pdf" -o outputs/bitcoin.translated.md
```

그림 분리를 적용한 결과 Bitcoin 논문은 9페이지에서 8개 영역, Pastry 논문은 22페이지에서 10개 그림을 추출했습니다. 전체 번역은 Ollama 설치와 두 모델 다운로드 후 `run` 명령으로 실행합니다.

## 출력과 중단 후 재개

원본 초록과 생성 요약은 구분해 표시하며, 본문은 청크 보고서 대신 원래 절과 문단 순서로 제공합니다. HTML은 IEEE Xplore의 논문 읽기 구성을 참고해 왼쪽 목차, 제목과 저자 정보, 초록, 본문, 문맥 속 그림과 캡션, 참고문헌을 배치합니다. 헤더는 `PLoTrans`입니다. 번역이 있는 문서는 `Original`, `Translation`, `Parallel view`로 전환할 수 있으며 인쇄용 스타일과 작은 화면용 레이아웃도 포함됩니다. Markdown에는 같은 절 순서와 접을 수 있는 번역 영역을 제공합니다.

처리 도중 중단되면 출력 파일 옆에 `.progress.json` 체크포인트가 남습니다. 같은 PDF와 동일한 모델·설정·출력 경로로 다시 실행하면 완료된 청크를 건너뜁니다. 성공하면 체크포인트를 삭제합니다. HTML은 외부 CDN이나 서버 없이 로컬 파일로 열 수 있습니다.

## 그림과 글 분리

OpenDataLoader의 텍스트 구조에 더해, 글자 표시 명령만 제거한 메모리상의 PDF를 렌더링해 실제 그래픽 영역을 찾습니다. 따라서 JPG/PNG 이미지뿐 아니라 PDF 도형·화살표·차트로 구성된 벡터 그림도 PNG로 보존합니다. 원본 PDF는 수정하지 않습니다.

그림 안의 글자는 원본 그림에 남기고 본문에서 중복 추출하지 않습니다. 회전된 축 제목과 가까운 레이블도 그림 영역에 포함합니다. PDF 내부 Form XObject에 들어간 레이블은 변환된 페이지 글자 좌표로 대조합니다. 캡션이 짧아도 실제 그림 너비를 자르지 않으며, 같은 캡션에 속한 여러 패널은 묶고 서로 다른 단의 그림은 분리합니다. 구조가 분명한 일반 표는 Markdown 표로 유지합니다.

캡션이 제목으로 오인되거나 여러 블록으로 나뉘면 글꼴·위치·줄 간격으로 연결합니다. 캡션은 청크 경계에서 나누지 않고 완성된 단위로 그림에 붙입니다. 일반 캡션은 `Fig.`로 정규화하며 수식을 포함한 원본 이미지 캡션은 그대로 보존합니다. 대응 그림을 찾지 못한 캡션도 본문에 남깁니다.

캡션 좌표가 `(a)`·`(b)` 같은 패널 이름까지 포함한 경우에는 실제 `Fig.` 글자 위치로 캡션 상단을 보정합니다. 캡션 위로 솟은 수학 글꼴·기호가 있으면 보정을 보류해 수식을 자르지 않습니다. 단순히 직전에 그림이 있다는 이유로 표 캡션을 그 그림에 붙이지 않습니다.

출력 구조는 다음과 같습니다. 하위 폴더는 PDF 내용의 해시로 구분하므로 여러 논문의 그림 파일이 충돌하지 않습니다.

```text
outputs/
  paper.translated.md
  paper.translated.html
  paper.translated.zip
  assets/
    article.css
  figs/
    <PDF 해시>/
      page-002-fig-01.png
      page-002-fig-02.png
      page-016-math-003.png
```

Markdown은 `![Fig. p.2-1](figs/<PDF 해시>/page-002-fig-01.png)`처럼 상대 경로를 참조합니다. 다른 PC로 옮길 때는 ZIP을 풀어 폴더 구성을 유지하세요. 웹 화면에도 동일한 HTML 문서를 미리보기로 제공합니다. 그림 파일 경로는 번역 모델에 보내지 않아 모델이 경로를 변경하지 않습니다.

## 수식 누락 방지

Pastry 논문처럼 문자 대응 정보(ToUnicode)가 없는 Type3 글꼴을 사용하는 PDF는 화면에 수식이 보이더라도 텍스트 추출 결과에서 수식이 빠질 수 있습니다. 이런 수식 글꼴과 별도 수학 글꼴 구간은 원본 PDF에서 작은 PNG로 렌더링해 문장 안에 삽입합니다. 그림과 같은 `figs/<PDF 해시>/` 폴더에 보관되며 ZIP에도 함께 들어갑니다.

예를 들어 Fig.7 캡션의 `b=4`, `|L|=16`, `|M|=32`는 원본 수식 이미지로 보존합니다. 문자별 좌표를 사용해 수식 글꼴뿐 아니라 같은 식에 속한 일반 글꼴 숫자, 연산자, 괄호, 함수명과 첨자까지 하나의 이미지로 묶습니다. 따라서 `O(log N)` 같은 식이 일부만 이미지로 나뉘는 현상을 줄입니다.

문장 속 수식은 `<img class="inline-math" ...>` 형태의 인라인 HTML로 삽입합니다. 주변 글자 크기에 대한 `em` 높이와 원본 PDF 기준선으로 수직 정렬을 계산하며, CSS를 제거하는 뷰어에도 작은 크기를 유지하도록 `width`·`height` 속성을 함께 제공합니다. 원문과 번역에 같은 HTML 태그를 복원합니다. HTML을 허용하는 Markdown 뷰어를 사용하세요.

수식만 있는 행을 추출기가 통째로 누락했으면 독립된 수식 이미지를 추가합니다. 대응 위치가 불확실한 구간은 원문 이미지 한 번으로 표시합니다. 하나의 조건부 수식이 여러 텍스트 블록에 걸쳐 있으면 이미지에 포함된 다른 블록도 함께 정리합니다. 경계에 걸친 블록은 먼저 이미지 영역을 확장해 보존한 뒤 중복을 제거합니다. 그림 영역에 속한 글자도 좌표를 대조해 본문에서 제거합니다.

번역 요청 시에는 인라인 수식을 `[[MATH_0001]]` 같은 표식으로 바꿉니다. 응답에서 각 표식이 정확히 한 번 나타나는지 검사한 뒤 원본 수식 이미지로 복원합니다. 누락·중복이면 한 번 재시도하고, 계속 실패하면 오류를 내어 수식이 사라진 번역을 저장하지 않습니다. 청크 경계에서도 이미지 경로를 쪼개지 않습니다.

이 기능은 원본 수식의 시각적 보존을 제공합니다. 자동 LaTeX 변환이나 수식 의미 인식은 수행하지 않으며, 요약 모델은 이미지로만 보존된 수식의 내용을 읽을 수 없습니다.

## 한계와 검토

- 기본 OpenDataLoader 모드는 텍스트 기반 PDF에 적합하며, 스캔·해독 불가 페이지는 로컬 OCR로 보완합니다. 그림은 원본 그대로 보존하며 그림 내부 글자 번역과 복잡한 배치의 의미 분석은 수행하지 않습니다.
- Kademlia처럼 페이지 전체가 해독 불가능한 글꼴이면 기본 OCR 경로로 본문을 복원합니다. OCR을 끄거나 인식한 글자가 없으면 원문 이미지를 보존하고 출력에 안내합니다. OCR 결과의 수식과 고유명사는 원본 페이지와 대조하세요.
- 그림 분리는 그래픽의 실제 픽셀·좌표와 캡션을 사용하는 규칙 방식입니다. 본문과 그림이 겹치거나 배경 무늬가 많은 문서는 원본과 확인해 주세요.
- 표는 JSON에서 Markdown 표로 재구성합니다. 병합 셀과 복잡한 표는 원본 PDF를 대조해 주세요.
- 요약과 번역은 로컬 모델의 생성 결과이므로 중요한 수치·수식·인용은 원문과 확인해야 합니다.
- `outputs/`에는 원문과 번역이 들어갈 수 있습니다. 이 폴더를 외부에 공유하기 전에 내용을 확인하세요.

## 개발 검증

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
python scripts/audit_corpus.py .tmp/example --output outputs/review --sheets
```

실제 OCR 모델을 네트워크 연결 없이 실행하는 통합 테스트도 포함되어 있습니다. 기본 테스트에서는 이 한 항목을 건너뜁니다. PowerShell에서는 `$env:PLOTRANS_TEST_OCR="1"`을 설정한 뒤 테스트를 실행하고, macOS/Linux에서는 `PLOTRANS_TEST_OCR=1 python -m unittest discover -s tests -v`를 사용하세요.

코퍼스 검사는 폴더 안 PDF를 모두 처리하고 논문별 Markdown·HTML·ZIP, 원본과 그림·캡션 비교 시트, `audit-report.html`/`.md`/`.json`을 생성합니다. 검사 항목은 이미지 파일과 내부 링크, 수식 이미지와 중복되는 텍스트, 캡션 누락·미연결, 해독 불가능한 문자, 페이지 이미지 대체 여부입니다. OpenDataLoader의 JSON은 `.scratch/corpus/native/`에 캐시하며 원본 PDF를 바꾸었다면 해당 캐시도 지운 뒤 다시 실행하세요. 자동 검사만으로 의미적 완전성을 보장하지 않으므로 비교 시트와 원본을 함께 검토하세요.

이번 예시 18편의 검사는 추출 경로를 검증한 것이며 Ollama 모델 추론은 실행하지 않았습니다. 테스트는 조건부 수식 중복, 줄이 나뉜 캡션, 두 단의 여러 패널, Form XObject 좌표, 아래첨자, 해독 불가능한 글꼴 보존과 출력 번들 등을 확인합니다.

의존성은 `pyproject.toml`에서 버전을 고정했습니다. Ollama 모델의 실제 해시를 기록하려면 `ollama list`를 사용하세요. 완전히 동일한 모델 바이트가 필요한 환경에서는 모델 파일 해시와 Docker 이미지 digest도 별도로 보관해야 합니다.
