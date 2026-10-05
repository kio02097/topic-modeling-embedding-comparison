# 문서 임베딩 생성 방법에 따른 토픽 모델링 성능 비교

### S-BERT와 LLM 임베딩 모델 중심으로

**저자:** 박지원 · 양치복 · 이충권  
**소속:** 계명대학교 경영정보학과

## 연구 소개

이 연구는 문서를 벡터로 바꾸는 방식이 토픽 모델의 품질에 미치는 영향을 비교합니다. `all-MiniLM-L6-v2` 기반 S-BERT와 `Llama-3.2-1B` 임베딩을 BBC News, 20 Newsgroups(20NG), IMDB에 적용하고, CLS(C), max(X), mean(M) 풀링과 그 조합을 비교했습니다.

문서 임베딩 후 UMAP 차원 축소, HDBSCAN 군집화, c-TF-IDF 단어 추출을 수행하는 BERTopic을 사용했습니다. 토픽 일관성은 C_v와 NPMI로, 토픽 다양성은 상위 단어의 중복 정도로 평가했습니다.

## 논문에 보고된 최고 점수

각 지표에서 데이터셋·모델별 최고 결과입니다. C_v와 NPMI의 최고 조건은 서로 다를 수 있습니다.

| 데이터셋 | 모델 | 최고 C_v (토픽 수, 풀링) | 최고 NPMI (토픽 수, 풀링) |
| --- | --- | --- | --- |
| BBC News | S-BERT | 0.7893 (16, CX) | 0.2113 (16, CX) |
| BBC News | LLaMA | 0.7558 (20, CX) | 0.2221 (4, CX) |
| 20NG | S-BERT | 0.6502 (20, M) | 0.1246 (22, CX) |
| 20NG | LLaMA | 0.6158 (10, CM) | 0.0829 (24, CXM) |
| IMDB | S-BERT | 0.6498 (4, M) | 0.1638 (4, CXM) |
| IMDB | LLaMA | 0.6616 (6, M) | 0.1549 (6, M) |

논문은 mean pooling을 적용한 LLaMA가 비교적 안정적이었으며, CLS와 max를 결합한 S-BERT가 여러 조건에서 높은 일관성을 보였다고 보고합니다. 표 6의 단일 풀링/듀얼 풀링 시간은 S-BERT 11분 59초/13분 40초, LLaMA 47분 50초/48분 47초입니다. S-BERT는 L4 GPU, LLaMA는 A100 GPU에서 측정했으므로 시간은 하드웨어 조건과 함께 해석해야 합니다.

## 실험 설정

- **데이터:** Hugging Face SetFit의 BBC News, 20 Newsgroups, IMDB. 각 데이터의 train/test를 합쳐 코퍼스를 구성했습니다.
- **임베딩:** `sentence-transformers/all-MiniLM-L6-v2`, `meta-llama/Llama-3.2-1B`
- **풀링:** C(CLS), X(max), M(mean), CM, CX, XM, CXM
- **토픽 수:** 2부터 24까지 짝수 값
- **BERTopic:** UMAP `n_neighbors=15`, `n_components=5`, `min_dist=0`, `metric=cosine`; HDBSCAN `min_cluster_size=10`, `metric=euclidean`, `cluster_selection_method=eom`; c-TF-IDF 상위 단어 10개
- **평가:** Gensim C_v, c_npmi, Topic Diversity

전처리는 기호 정리, 소문자 변환, 영어 불용어 제거, spaCy 표제어화와 짧은 문서 제거를 포함합니다. 데이터별 예외 불용어와 세부 처리 순서는 논문과 원본 노트북을 함께 확인하세요.

## 코드와 재현 안내

이 저장소는 실험 노트북을 모듈로 정리한 연구 코드입니다. 위 표의 값은 논문 보고 결과입니다. 현재 코드가 논문 수치를 완전히 재현하는지는 검증하지 않았습니다. 실행 환경, 모델·데이터 revision, 일부 전처리 세부 정보가 원 실험과 다를 수 있습니다. 자세한 차이는 [재현성 기록](docs/REPRODUCIBILITY.md)을 참고하세요.

```text
src/topic_embedding/       데이터 준비, 임베딩, 토픽 모델링, 평가 코드
notebooks/                 실행 예시와 민감 정보 제거 원본 노트북
tests/                     핵심 테스트
docs/REPRODUCIBILITY.md    논문 설정과 코드의 차이
```

## 실행

Python 3.10–3.12 환경에서 설치합니다.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords

# 작은 입력으로 실행 흐름 확인
topic-experiment --dataset bbc --model sbert --pooling CX --topics 2 4 --limit 200 --umap-seed 42
```

위 예시는 입력을 200개로 제한하고 UMAP seed를 고정하므로 논문 실험 조건과 다릅니다. 논문 설정의 UMAP `random_state`는 `None`입니다. 전체 옵션은 `topic-experiment --help`에서 확인하세요. LLaMA 사용에는 Hugging Face 접근 승인이 필요할 수 있습니다. 인증 토큰을 코드나 저장소에 기록하지 마세요.

## 논문 정보

박지원, 양치복, 이충권. 「문서 임베딩 생성 방법에 따른 토픽 모델링 성능 비교: S-BERT와 LLM 임베딩 모델 중심으로」. 계명대학교 경영정보학과.

제공된 원문에서 공개 링크와 완전한 학술지 서지 정보를 확인하지 못해 DOI나 학술지 정보를 추정해 기재하지 않았습니다. 논문 PDF와 원 데이터는 저장소에 포함하지 않았습니다.
