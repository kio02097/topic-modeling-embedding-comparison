# 문서 임베딩 생성 방법에 따른 토픽 모델링 성능 비교

### S-BERT와 LLM 임베딩 모델 중심으로

**저자:** 박지원 · 양치복 · 이충권  
**소속:** 계명대학교 경영정보학과

## 연구 목적

토픽 모델링 결과는 문서를 어떤 벡터로 표현하느냐에 따라 달라질 수 있습니다. 이 논문은 **임베딩 모델의 종류와 토큰 임베딩을 문서 임베딩으로 모으는 풀링 방식이 토픽 품질에 어떤 영향을 주는지** 비교합니다.

특히 비교적 가벼운 문장 임베딩 모델인 S-BERT가 적절한 풀링 전략을 사용할 때 LLM 임베딩과 견줄 수 있는지, 그리고 풀링 방식을 결합했을 때 토픽 품질과 실행 시간은 어떻게 달라지는지를 살펴봅니다.

## 논문이 던지는 질문

- BBC News, 20 Newsgroups(20NG), IMDB처럼 성격이 다른 데이터셋에서 S-BERT와 LLM의 결과는 어떻게 달라지는가?
- CLS, max, mean 풀링과 이들의 조합 중 어떤 방식이 토픽 일관성에 유리한가?
- 여러 풀링 결과를 결합하면 토픽 다양성을 유지하면서 추가 실행 부담을 줄일 수 있는가?

## 논문의 핵심 메시지

**LLM 임베딩이 항상 더 좋은 토픽을 만드는 것은 아닙니다.** 임베딩 모델의 크기만으로 성능을 판단하기보다, 데이터셋에 맞는 풀링 방식을 함께 선택해야 합니다.

논문에서는 LLaMA의 mean pooling이 여러 조건에서 비교적 안정적인 결과를 보였습니다. 한편 S-BERT는 CLS와 max를 결합한 CX 풀링 등에서 높은 토픽 일관성을 보였고, 데이터셋과 평가 지표에 따라 LLaMA와 같거나 더 높은 점수를 기록했습니다. 따라서 적절한 풀링을 적용한 S-BERT도 일부 조건에서는 LLM 임베딩의 대안이 될 수 있다는 것이 연구의 주요 시사점입니다.

다만 우세한 모델과 풀링 방식은 데이터셋 및 지표에 따라 달라집니다. 이 결과는 S-BERT가 모든 상황에서 LLaMA보다 낫다는 뜻이 아니며, 아래 점수도 각 지표에서 따로 확인한 최고 조건입니다.

## 연구 방법

세 데이터셋에 두 임베딩 모델과 일곱 가지 풀링 전략을 적용한 뒤 BERTopic으로 토픽을 구성했습니다.

- **데이터:** Hugging Face SetFit의 BBC News, 20 Newsgroups, IMDB
- **임베딩 모델:** S-BERT sentence-transformers/all-MiniLM-L6-v2, LLM meta-llama/Llama-3.2-1B
- **풀링:** CLS(C), max(X), mean(M), 그리고 조합 CM, CX, XM, CXM
- **토픽 모델링:** 임베딩 → UMAP 차원 축소 → HDBSCAN 군집화 → c-TF-IDF 키워드 추출(BERTopic)
- **평가:** 토픽 일관성 C_v와 NPMI, 토픽 다양성(상위 토픽 단어의 중복 정도)
- **탐색한 토픽 수:** 2부터 24까지 짝수 값

## 논문에 보고된 최고 점수

아래는 각 데이터셋·모델에서 지표별로 가장 높은 점수입니다. C_v와 NPMI는 서로 다른 토픽 수나 풀링 방식에서 최고일 수 있으므로, 한 행의 두 결과를 하나의 설정에서 얻은 값으로 해석하면 안 됩니다.

| 데이터셋 | 임베딩 모델 | 최고 C_v (토픽 수, 풀링) | 최고 NPMI (토픽 수, 풀링) |
| --- | --- | --- | --- |
| BBC News | S-BERT | 0.7893 (16, CX) | 0.2113 (16, CX) |
| BBC News | LLaMA | 0.7558 (20, CX) | 0.2221 (4, CX) |
| 20NG | S-BERT | 0.6502 (20, M) | 0.1246 (22, CX) |
| 20NG | LLaMA | 0.6158 (10, CM) | 0.0829 (24, CXM) |
| IMDB | S-BERT | 0.6498 (4, M) | 0.1638 (4, CXM) |
| IMDB | LLaMA | 0.6616 (6, M) | 0.1549 (6, M) |

논문 표 6의 단일 풀링/듀얼 풀링 실행 시간은 S-BERT가 11분 59초/13분 40초, LLaMA가 47분 50초/48분 47초입니다. 다만 S-BERT는 L4 GPU, LLaMA는 A100 GPU에서 측정했으므로 두 모델의 시간을 직접적인 속도 비교로 볼 수 없습니다.

## 실험 설정

- 데이터별 train/test split을 합쳐 코퍼스로 구성
- UMAP: n_neighbors=15, n_components=5, min_dist=0, metric=cosine
- HDBSCAN: min_cluster_size=10, metric=euclidean, cluster_selection_method=eom
- c-TF-IDF: 상위 단어 10개 사용
- 전처리: 소문자화, 기호 정리, 영어 불용어 제거, spaCy 표제어화, 짧은 문서 제거 등

세부 전처리와 데이터별 예외 처리는 원 논문 및 노트북을 확인하세요. 원문 설명과 구현 사이에 차이가 있을 수 있는 부분은 [재현성 기록](docs/REPRODUCIBILITY.md)에 정리했습니다.

## 저장소 안내

이 저장소는 실험 노트북을 실행 가능한 코드 구조로 정리한 것입니다. 표의 점수는 논문에 보고된 결과이며, 현재 저장소의 코드로 논문 수치를 완전히 재현했음을 뜻하지는 않습니다. 모델·데이터 버전, 실행 환경, 일부 전처리 설정 차이를 고려해 주세요.

    src/topic_embedding/       데이터 준비, 임베딩, 토픽 모델링, 평가 코드
    notebooks/                 실행 예시와 민감 정보 제거 원본 노트북
    tests/                     핵심 테스트
    docs/REPRODUCIBILITY.md    논문 설정과 코드의 차이

## 실행

Python 3.10–3.12 환경에서 설치합니다.

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -e .
    python -m spacy download en_core_web_sm
    python -m nltk.downloader stopwords

작은 입력으로 실행 흐름을 확인하는 예시입니다.

    topic-experiment --dataset bbc --model sbert --pooling CX --topics 2 4 --limit 200 --umap-seed 42

이 예시는 입력을 200개로 제한하고 UMAP seed를 고정하므로 논문 실험 조건과 다릅니다. 전체 옵션은 topic-experiment --help에서 확인할 수 있습니다. LLaMA를 사용하려면 Hugging Face 접근 승인이 필요할 수 있습니다. 인증 토큰은 코드나 저장소에 기록하지 마세요.

## 논문 정보

박지원, 양치복, 이충권. 「문서 임베딩 생성 방법에 따른 토픽 모델링 성능 비교: S-BERT와 LLM 임베딩 모델 중심으로」. 계명대학교 경영정보학과.

제공된 원문에서 DOI와 완전한 학술지 서지 정보를 확인하지 못해 임의로 기재하지 않았습니다. 논문 PDF와 원 데이터는 저장소에 포함하지 않았습니다.
