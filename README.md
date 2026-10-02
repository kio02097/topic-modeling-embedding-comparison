# Topic Modeling by Document Embedding Generation Methods

**문서 임베딩 생성 방법에 따른 토픽 모델링 성능 비교: S-BERT와 LLM 임베딩 모델 중심으로**

박지원 · 양치복 · 이충권 (계명대학교 경영정보학과)

This repository reorganizes the supplied research notebook into a configurable pipeline for comparing S-BERT and LLaMA document embeddings in BERTopic. It supports the three datasets and seven pooling configurations described in the manuscript, with an optional LDA baseline from the notebook.

이 저장소는 제공된 `topic (1).ipynb`를 정리한 연구 코드입니다. **논문 수치의 완전 재현은 아직 검증하지 않았습니다.** 정리 과정에서 바뀐 점과 확인할 실험 설정은 [재현성 문서](docs/REPRODUCIBILITY.md)에 기록했습니다. 원래 실험 환경의 정확한 버전과 데이터·모델 revision은 확인되지 않았습니다.

## Experiment design

| Item | Options |
| --- | --- |
| Dataset | BBC News (`bbc`), 20 Newsgroups (`20ng`), IMDB (`imdb`) |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, `meta-llama/Llama-3.2-1B` |
| Pooling | C, X, M, CM, CX, XM, CXM |
| Requested topic counts | 2, 4, …, 24 |
| Metrics | Gensim c_v, c_npmi, topic diversity (top 10 terms) |
| Clustering | UMAP → HDBSCAN → c-TF-IDF |

`C` is CLS/first-token pooling, `X` is max pooling, and `M` is mean pooling. Combined strategies concatenate vectors. For LLaMA, C takes the first token (BOS when present); LLaMA has no learned CLS token. Labels such as CM describe selected components, not a guaranteed vector concatenation order across library versions.

논문 표 2에 따라 UMAP은 `n_neighbors=15`, `n_components=5`, `min_dist=0.0`, `metric=cosine`, `random_state=None`을 사용합니다. `--umap-seed 42`로 고정할 수 있지만 논문의 설정과 달라집니다. HDBSCAN은 `min_cluster_size=10`, `metric=euclidean`, `cluster_selection_method=eom`입니다.

## Installation

Python 3.10–3.12 환경에서 실행하도록 구성했습니다. 아래 의존성 범위는 설치 후보이며, 전체 환경 설치와 모델 실행을 검증한 lockfile은 아닙니다. PyTorch의 CPU/CUDA 빌드는 장비에 맞게 설치하세요.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -e .
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```

LLaMA 실험은 모델 접근 승인을 받은 Hugging Face 계정이 필요할 수 있습니다. 모델 접근은 사용자가 직접 승인받아야 합니다. 토큰은 코드에 넣지 말고 로그인 또는 환경변수로 전달하세요.

```bash
huggingface-cli login
```

`.env.example`은 안내용이며 자동으로 로드되지 않습니다. 토큰·데이터·모델·임베딩은 저장소에 커밋하지 않습니다.

## Run

작은 BBC 입력으로 실행 흐름을 확인합니다. 이 결과는 논문 결과와 비교하지 않습니다.

```bash
topic-experiment --dataset bbc --model sbert --pooling CX --topics 2 4 --limit 200 --umap-seed 42
```

한 데이터셋에서 7개 풀링과 12개 토픽 수를 실행합니다.

```bash
topic-experiment --dataset bbc --model sbert --pooling all
topic-experiment --dataset bbc --model llama --pooling all --batch-size 4
```

`--dataset 20ng` 또는 `--dataset imdb`로 바꾸어 반복합니다. 모든 조합은 GPU 메모리·시간·저장 공간을 상당히 사용할 수 있으므로 먼저 단일 조합으로 실행하세요. SBERT 길이는 256이며, LLaMA는 노트북의 명시적 길이인 **512를 잠정 기본값**으로 사용합니다. 논문 표의 131,072는 모델 지원 길이이며 실제 실험 입력 길이로 확인되지 않았습니다.

```bash
# 선택: 원본 노트북에 있던 청크 기반 확장 (논문 적용 여부 미확인)
topic-experiment --dataset imdb --model llama --pooling XM --max-length 512 --chunk-size 512 --chunk-overlap 64 --batch-size 4
# 선택: LDA baseline
topic-experiment --dataset bbc --model lda
# 입력 옵션 확인
topic-experiment --help
```

모델·데이터의 Hugging Face commit SHA가 있으면 `--model-revision`과 `--dataset-revision`으로 고정하세요. 모델 revision을 고정한 경우에만 기존 임베딩 캐시를 재사용합니다. 캐시는 문서 순서·풀링·길이·버전·배치 등으로 구분합니다. 해시 seed는 Python 실행 전 `PYTHONHASHSEED=42`로 설정해야 적용됩니다. 같은 seed만으로 GPU나 라이브러리 차이를 포함한 완전 재현을 보장하지 않습니다.

## Outputs

각 실행은 별도 `results/<dataset>-<model>-<timestamp>/`에 저장됩니다.

| File | Contents |
| --- | --- |
| `metrics.csv` | 조건별 c_v, c_npmi, TD, 실제 토픽 수, 이상치 비율 |
| `topics.json` | 조건별 토픽 상위 단어 |
| `summary.json` | 지표별 평균, 유효한 평가 수, 최적 요청 토픽 수 |
| `metadata.json` | 입력 옵션, 문서 해시, 데이터 fingerprint, 모델 commit, 실행 환경, 시간, 완료 여부 |

`nr_topics`는 축소 목표입니다. HDBSCAN이 만든 토픽 수보다 더 많은 토픽이 만들어지는 것은 아니므로 `n_clusters`와 함께 해석하세요. train/test는 논문처럼 합치며, 지표는 같은 코퍼스에서 계산합니다. 별도 test 성능으로 해석하지 않습니다. 평가할 토픽이 없으면 지표는 결측값(CSV 빈 값/JSON null)입니다.

## Repository layout

```text
src/topic_embedding/
  preprocessing.py   # 데이터 로드, 정제, 표제어화
  embeddings.py      # 7개 풀링, 선택적 청크 처리, 캐시 키
  modeling.py        # BERTopic / LDA 토픽 수별 실험
  evaluation.py      # c_v, c_npmi, TD
  cli.py             # 실행, 결과와 환경 기록
notebooks/
  run_experiments.ipynb      # 모듈 실행 예시
  original_sanitized.ipynb  # 출력·토큰을 제거한 원본 참고본
docs/REPRODUCIBILITY.md
tests/test_core.py
```

원본 참고본에는 중복 셀과 실행 순서 의존성이 남아 있으므로 실험 실행은 CLI 또는 `run_experiments.ipynb`를 사용합니다. 원본 실행 출력은 검증된 논문 결과로 배포하지 않습니다.

## Validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

정리 시 확인한 범위: Python 문법, 캐시 구분, 풀링 조건, 청크 경계, TD 계산, CLI 옵션, 노트북 형식, 공개 파일 내 토큰 패턴. 전체 의존성 설치·데이터 다운로드·GPU 임베딩·논문 수치 재현은 아직 검증하지 않았습니다. 전체 실행 후 `python -m pip freeze > results/environment-lock.txt`를 저장하고 metadata와 함께 보관하세요.

## Paper and reuse

논문 원문 PDF는 저장소에 포함하지 않았습니다. 출판 정보(학술지명·권호·쪽·DOI)와 공개 가능한 논문 링크가 확인되면 이 부분을 완성할 수 있습니다. 현재 제공된 제목·저자만으로 불완전한 서지나 DOI를 만들지 않았습니다.

코드의 재사용 라이선스는 저자 확인 전 지정하지 않았습니다. 데이터셋과 모델의 이용 조건은 각 제공처를 확인하세요.

## Documentation references

- [BERTopic parameter tuning](https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html)
- [BERTopic best practices](https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html)
- [Sentence Transformers custom models](https://sbert.net/docs/sentence_transformer/usage/custom_models.html)
