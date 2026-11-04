# 논문 로컬 번역기

논문 PDF를 OpenDataLoader PDF로 구조화하고, Tencent HY-MT2-7B로 번역하며, 별도의 로컬 요약 모델로 핵심 내용을 정리합니다. 결과 Markdown에는 페이지 표시와 함께 원문 청크 및 번역 청크가 순서대로 담깁니다. PDF 내용은 OpenDataLoader와 로컬 Ollama에만 전달됩니다.

## 구성

- PDF 추출: `opendataloader-pdf==2.5.17` (로컬 Java 프로세스)
- 번역: Tencent `Hy-MT2-7B` 공식 GGUF 가중치를 Ollama용 템플릿으로 감싼 `kaelri/hy-mt2:7b-q4_K_M`
- 요약: `qwen3:4b`
- 화면: `streamlit==1.66.1`
- Python: 3.11 이상, 권장 3.12 / Java: 11 이상, 권장 17
- 기본 번역 언어: 한국어

기본 모델 태그는 [HY-MT2 Ollama 배포 페이지](https://ollama.com/kaelri/hy-mt2)에서 확인할 수 있습니다. Tencent의 [공식 모델 페이지](https://huggingface.co/tencent/Hy-MT2-7B)는 번역 프롬프트와 권장 추론 설정을 제공합니다. 다른 태그를 이미 보유하셨다면 CLI의 `--translation-model` 또는 화면의 설정에서 지정할 수 있습니다.

## 방법 1: Docker Compose (Windows, macOS, Linux)

Docker Desktop 또는 Docker Engine/Compose가 필요합니다. 최초 실행 때 이미지와 모델을 다운로드하므로 인터넷이 필요합니다. 이후 처리에는 외부 API 키가 필요하지 않습니다.

```sh
docker compose up -d ollama
docker compose exec ollama ollama pull kaelri/hy-mt2:7b-q4_K_M
docker compose exec ollama ollama pull qwen3:4b
docker compose up --build -d app
```

브라우저에서 <http://127.0.0.1:8501>을 열고 PDF를 선택한 다음 **분석 및 번역**을 누르세요. 완료된 Markdown은 화면에서 내려받거나 프로젝트의 `outputs/` 폴더에서 찾을 수 있습니다. 종료 명령은 `docker compose down`입니다. 모델은 Docker 볼륨에 유지됩니다.

Docker 설정은 Ollama `0.40.14`와 Python `3.12` 기반 이미지를 사용합니다. GPU 없이도 실행할 수 있지만 7B 모델은 상당한 메모리와 시간이 필요합니다. 환경에 따라 Docker의 메모리 한도를 늘려 주세요.

## 방법 2: 직접 설치

1. Python 3.11 이상, Java 11 이상, [Ollama](https://ollama.com/download)를 설치합니다. 터미널에서 `python --version`, `java -version`, `ollama --version`을 확인하세요. Windows에서는 새 PowerShell을 열어 PATH 변경을 반영하세요.
2. 이 폴더에서 가상 환경과 Python 패키지를 설치합니다.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui]"
```

**macOS / Linux**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ui]'
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
paper-localizer run "path/to/paper.pdf" -o "outputs/paper.ko.md"
paper-localizer extract "path/to/paper.pdf" -o "outputs/paper.extracted.md"
```

웹 화면은 <http://localhost:8501>에서 열립니다. CLI는 `paper-localizer run --help`로 옵션을 볼 수 있습니다. `--target-language`로 번역 언어를 바꾸고 `--chunk-size`로 모델에 보내는 문자 수를 조정할 수 있습니다.
`extract`는 Ollama 없이 OpenDataLoader의 추출 결과를 먼저 검토할 때 사용합니다.

### 포함된 예시로 확인

`.tmp/example/`에 논문 PDF를 넣으면 Git에서 제외됩니다. 아래 두 논문으로 추출 경로를 확인했습니다.

```sh
paper-localizer extract ".tmp/example/(2008) Bitcoin_ A Peer-to-Peer Electronic Cash System.pdf" -o outputs/bitcoin.extracted.md
paper-localizer extract ".tmp/example/(2001) Pastry_ Scalable, decentralized object location and routing for large-scale peer-to-peer systems.pdf" -o outputs/pastry.extracted.md
```

Bitcoin 논문은 9페이지, 15청크로, Pastry 논문은 22페이지, 39청크로 추출됐습니다. 전체 번역은 Ollama 설치와 두 모델 다운로드 후 `run` 명령으로 실행합니다.

## 출력과 중단 후 재개

Markdown 첫 부분에는 목적·방법·주요 결과·한계 요약이 들어갑니다. 이어서 각 청크별 원문과 번역이 함께 정리되며 페이지 번호가 표시됩니다. 처리 도중 중단되면 출력 파일 옆에 `.progress.json` 체크포인트가 남습니다. 같은 PDF와 동일한 모델·설정·출력 경로로 다시 실행하면 완료된 청크를 건너뜁니다. 성공하면 체크포인트를 삭제합니다.

## 한계와 검토

- 기본 OpenDataLoader 모드는 텍스트 기반 PDF에 적합합니다. 스캔본의 OCR, 복잡한 수식·그림 해석은 별도 하이브리드 설정이 필요합니다.
- 표는 JSON에서 Markdown 표로 재구성합니다. 병합 셀과 복잡한 표는 원본 PDF를 대조해 주세요.
- 요약과 번역은 로컬 모델의 생성 결과이므로 중요한 수치·수식·인용은 원문과 확인해야 합니다.
- `outputs/`에는 원문과 번역이 들어갈 수 있습니다. 이 폴더를 외부에 공유하기 전에 내용을 확인하세요.

## 개발 검증

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

의존성은 `pyproject.toml`에서 버전을 고정했습니다. Ollama 모델의 실제 해시를 기록하려면 `ollama list`를 사용하세요. 완전히 동일한 모델 바이트가 필요한 환경에서는 모델 파일 해시와 Docker 이미지 digest도 별도로 보관해야 합니다.
