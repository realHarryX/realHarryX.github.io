<div align="center">

# KBMR

### Beyond Visual Similarity: Entity-Aligned Retrieval for
### Knowledge-Based Visual Question Answering

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2608.21450-b31b1b.svg)](https://arxiv.org/abs/2608.21450)
[![Conference](https://img.shields.io/badge/ACM%20MM-2026-6f42c1.svg)](https://arxiv.org/abs/2608.21450)
[![Project Page](https://img.shields.io/badge/Project%20Page-Online-blue.svg)](https://realharryX.github.io/KBMR/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Backbone](https://img.shields.io/badge/Backbone-Qwen2--VL--7B-00B8D9.svg)](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
[![Task](https://img.shields.io/badge/Task-KB--VQA%20Retrieval-2ea44f.svg)](https://arxiv.org/abs/2608.21450)

*An MLLM-based retriever that moves KB-VQA retrieval from surface-level visual matching toward*

*entity-aligned semantic retrieval.*

<img src="assets/method.png" alt="KBMR method overview" width="100%">

<sub><b>KBMR overview.</b> EVA-CLIP-8B prefilters candidates, an MLLM Semantic Discriminator provides continuous entity-consistency weights, and Qwen2-VL-7B is trained with continuous semantic distillation.</sub>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Method](#method)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data and Model Preparation](#data-and-model-preparation)
- [Build Training Data](#build-training-data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Overview

Knowledge-Based Visual Question Answering (KB-VQA) requires retrieving external evidence for long-tail entities. Existing first-stage retrievers are commonly based on CLIP-style visual similarity. This is fragile when the same entity appears under different viewpoints, periods, or styles, and when different entities look visually similar.

KBMR replaces the first-stage CLIP retriever with an MLLM embedding retriever. Images are prompted with:

```text
<Image> Summary above image in one word:
```

The final-token hidden state is normalized and used as the image embedding. An MLLM Semantic Discriminator then produces continuous entity-consistency weights for query-candidate pairs. These weights guide hard-negative construction and supervise the retriever through a symmetric KL distillation objective.

<div align="center">
<img src="assets/motivation.png" alt="Motivation for entity-aligned retrieval" width="62%">
<br>
<sub><b>Motivation.</b> Surface similarity can retrieve a visually similar but incorrect entity, whereas KBMR targets entity-level semantic relevance.</sub>
</div>

---

## Highlights

- **The first MLLM-based embedding retriever tailored for KB-VQA**, shifting retrieval from surface-level visual similarity toward entity-aligned semantic matching.
- **Entity-aware supervision:** An MLLM-based Semantic Discriminator generates continuous entity-consistency weights for reliable hard-negative mining and fine-grained soft supervision.
- **Continuous Semantic Distillation:** KBMR aligns retrieval similarities with an entity-aware semantic prior, improving discrimination among visually similar yet semantically distinct entities.
- **Plug-and-play improvements:** Replacing the original retriever with KBMR boosts diverse KB-VQA pipelines by up to **+9.4 points** in end-to-end answer accuracy, without modifying their downstream reasoning or reranking components.
- **State-of-the-art performance:** KBMR achieves up to **+14.7 points in Recall@1** and VQA scores of **54.7**, **50.8**, and **79.3** on E-VQA, InfoSeek, and OK-VQA, respectively.
---

## Method

### Stage 1 — EVA-CLIP prefiltering

EVA-CLIP-8B encodes each query and knowledge-base image. After excluding candidates associated with the same Wikipedia entity, the 50 most similar images form the potential hard-negative set.

```text
potential negatives = Top-50 EVA-CLIP cosine neighbors excluding positives
```

### Stage 2 — Semantic relevance scoring

Qwen2.5-VL-7B receives a query image and candidate image with the following instruction:

```text
You need to determine whether the given Candidate and the Query refer to
the same entity. If they do, answer 'Yes'; otherwise, answer 'No'.
```

The entity-consistency weight is computed from the `Yes` and `No` token logits:

```text
w_i = exp(z_yes / gamma) / (exp(z_yes / gamma) + exp(z_no / gamma))
```


### Stage 3 — Hard-negative sampling

For positive weight `w_pos`, candidates with a weight above the margin threshold are removed:

```text
alpha = w_pos - beta
keep candidate i when w_i <= alpha
```

This release uses `beta = 0.01`. Remaining candidates are sorted by their discriminator weights, divided into four equal-frequency difficulty strata, and sampled two per stratum. Samples are duplicated when fewer than eight remain. A query is discarded when no candidate survives filtering.

### Continuous semantic distillation

For the target candidate and eight hard negatives, KBMR builds:

- a retriever posterior from cosine similarities;
- a semantic prior from discriminator weights.

Both distributions use temperature-scaled softmax. Training minimizes symmetric KL divergence:

```text
L_CSD = 0.5 * (KL(p_retriever || p_semantic) + KL(p_semantic || p_retriever))
```

The implementation is in [`kbmr/loss.py`](kbmr/loss.py).

---

## Results

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Retriever</th>
      <th colspan="4" align="center">E-VQA</th>
      <th colspan="4" align="center">InfoSeek</th>
    </tr>
    <tr>
      <th align="center">R@1 ↑</th>
      <th align="center">R@5 ↑</th>
      <th align="center">R@10 ↑</th>
      <th align="center">R@20 ↑</th>
      <th align="center">R@1 ↑</th>
      <th align="center">R@5 ↑</th>
      <th align="center">R@10 ↑</th>
      <th align="center">R@20 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Qwen2-VL-7B <em>(zero-shot)</em></td>
      <td align="center">6.1</td>
      <td align="center">14.7</td>
      <td align="center">18.5</td>
      <td align="center">21.9</td>
      <td align="center">17.5</td>
      <td align="center">31.4</td>
      <td align="center">35.1</td>
      <td align="center">37.6</td>
    </tr>
    <tr>
      <td align="left">EVA-CLIP-8B</td>
      <td align="center">13.3</td>
      <td align="center">31.3</td>
      <td align="center">41.0</td>
      <td align="center">48.8</td>
      <td align="center">45.6</td>
      <td align="center">67.1</td>
      <td align="center">73.0</td>
      <td align="center">77.9</td>
    </tr>
    <tr>
      <td align="left"><strong>KBMR (Qwen2-VL-7B)</strong></td>
      <td align="center"><strong>22.6</strong></td>
      <td align="center"><strong>43.1</strong></td>
      <td align="center"><strong>50.2</strong></td>
      <td align="center"><strong>53.5</strong></td>
      <td align="center"><strong>55.8</strong></td>
      <td align="center"><strong>71.1</strong></td>
      <td align="center"><strong>77.4</strong></td>
      <td align="center"><strong>83.2</strong></td>
    </tr>
  </tbody>
</table>

<sub><strong>Entity retrieval performance (%).</strong> Higher is better; the best result in each column is shown in bold.</sub>

</div>

---

## Repository Structure

```text
.
├── assets/
│   ├── method.png                  # Figure 2: complete KBMR pipeline
│   └── motivation.png              # Figure 1: motivation
├── kbmr/
│   ├── data.py                     # Original split parsing and positive selection
│   ├── eva_clip.py                 # EVA-CLIP-8B image encoder adapter
│   ├── io.py                       # Streaming JSON-array I/O
│   ├── judge.py                    # Qwen2.5-VL Semantic Discriminator
│   ├── loss.py                     # Symmetric KL distillation objective
│   ├── modeling.py                 # Attention backend selection
│   └── sampling.py                 # Filtering and stratified sampling
├── tools/
│   ├── build_eva_index.py          # Build the normalized FAISS corpus index
│   └── build_training_data.py      # Mine, score, and sample training records
├── scripts/
│   └── train_qwen2vl_7b.sh         # Qwen2-VL-7B training entry
├── tests/                          # Data, sampling, and loss tests
├── train.py                        # LoRA + continuous distillation training
├── evaluate.py                     # Entity-level retrieval evaluation
└── pyproject.toml
```

---

## Installation

```bash
git clone <your-repository-url> KBMR
cd KBMR

conda create -n kbmr python=3.10 -y
conda activate kbmr
pip install -e .
```

To install the optional test dependencies, use `pip install -e ".[dev]"` instead.

Model downloads may require accepting the corresponding Hugging Face licenses.

---

## Data and Model Preparation

### Required models

| Component | Model | Purpose |
|:--|:--|:--|
| Prefilter | EVA-CLIP-8B | Offline Top-50 candidate retrieval |
| Semantic Discriminator | `Qwen/Qwen2.5-VL-7B-Instruct` | Entity-consistency weights |
| Trainable retriever | `Qwen/Qwen2-VL-7B-Instruct` | Final KBMR retriever |

EVA-CLIP-8B checkpoints use different packaging conventions, so the code accepts a local model path or a compatible model-hub ID instead of assuming one distribution.

### Original training split

The data builder directly consumes the original post-split `train.json`; no intermediate query schema is introduced. The file is a top-level JSON array. KBMR uses `wikipedia_url` and `related_images` from each record.

The following is a field-level excerpt from the first record of the original file:

```json
[
  {
    "wikipedia_url": "https://en.wikipedia.org/wiki/Heracleum_mantegazzianum",
    "related_images": "iNaturalist_2021/train/06513_Plantae_Tracheophyta_Magnoliopsida_Apiales_Apiaceae_Heracleum_mantegazzianum/68bb3bbe-be28-4cdd-90a1-158af2d96b42.jpg"
  }
]
```

Other fields, including `wikipedia_title`, `question`, `answer`, `retrieval`, `dataset_name`, and `unique_id`, remain in the source split but are not used during hard-negative construction.

The original preprocessing behavior is preserved:

1. Deduplicate globally by `related_images`, keeping the first occurrence.
2. Group images by the exact `wikipedia_url` value.
3. Sort each entity's image paths lexicographically.
4. Select the next image with wraparound as the target image.
5. Use the same image as the target when an entity has only one image.

The source JSON is streamed with `ijson`; only the two required fields are retained in memory.

### Knowledge-base metadata

The knowledge-base metadata is another top-level JSON array. Each entry must contain at least:

```json
{
  "wikipedia_url": "https://en.wikipedia.org/wiki/Example_entity",
  "image_url": "https://upload.wikimedia.org/path/to/image.jpg"
}
```

A separate `url2image.json` object maps normalized image URLs to local relative paths:

```json
{
  "https://upload.wikimedia.org/path/to/image.jpg": "AToMiC-Images-v0.2/images/000/example.jpg"
}
```

### Build the EVA-CLIP index

```bash
python -m tools.build_eva_index \
  --metadata-json /path/to/knowledge_base/index.json \
  --url-to-image-json /path/to/knowledge_base/url2image.json \
  --image-root /path/to/knowledge_base/images \
  --model /path/to/EVA-CLIP-8B \
  --output-dir artifacts/eva_index
```

This creates:

```text
artifacts/eva_index/
├── index.faiss     # Normalized EVA-CLIP embeddings with inner-product search
├── corpus.json     # Index-aligned image paths and Wikipedia metadata
└── config.json     # Encoder and metric metadata
```

---

## Build Training Data

```bash
python -m tools.build_training_data \
  --train-json /path/to/original_split/train.json \
  --query-image-root /path/to/query/images \
  --candidate-image-root /path/to/knowledge_base/images \
  --index-dir artifacts/eva_index \
  --judge-model Qwen/Qwen2.5-VL-7B-Instruct \
  --output data/kbmr_train.json \
  --top-k 50 \
  --beta 0.01 \
  --gamma 1.1 \
  --num-negatives 8 \
  --image-size 336 \
  --seed 42
```

### Generated training format

The output is a top-level JSON array compatible with the original training loader. The example below is abridged to one negative; every generated record contains exactly eight negatives and eight scores.

```json
[
  {
    "query_text": "<|image_1|>\nRepresent the given image.\n",
    "query_image": "/path/to/query.jpg",
    "pos_text": "<|image_1|>\nRepresent the given image.\n",
    "pos_image": "/path/to/positive.jpg",
    "query_pos_scores": 0.95,
    "hard_negatives": [
      ["<|image_1|>\nRepresent the given image.\n", "/path/to/negative.jpg"]
    ],
    "hard_negatives_scores": [0.20]
  }
]
```

---

## Training

Edit the example paths at the top of [`scripts/train_qwen2vl_7b.sh`](scripts/train_qwen2vl_7b.sh), then run:

```bash
bash scripts/train_qwen2vl_7b.sh
```

The script defaults to the paper's eight-GPU setup and keeps the accumulated
query batch size at 1,024. On a four-GPU machine, run:

```bash
NUM_PROCESSES=4 bash scripts/train_qwen2vl_7b.sh
```

The training and evaluation entry points use FlashAttention 2 when it is installed and otherwise fall back to PyTorch SDPA.

---

## Evaluation

The evaluator consumes an original benchmark JSON array and the index-aligned `corpus.json` produced during EVA-CLIP indexing:

```bash
python evaluate.py \
  --adapter /path/to/output/kbmr-qwen2vl-7b/checkpoint-5000 \
  --index-dir artifacts/eva_index \
  --queries /path/to/original_split/test.json \
  --query-image-root /path/to/query/images \
  --candidate-image-root /path/to/knowledge_base/images \
  --recall-k 1 5 10 20
```

The evaluator reports entity-level recall by matching normalized Wikipedia URLs. For paper comparisons, use the official E-VQA test split and the complete InfoSeek validation split with the 100K Wikipedia knowledge base. Downstream QA follows the original benchmark protocols: BEM for E-VQA and VQA accuracy for InfoSeek.

---

## Acknowledgements

This implementation builds on the following open-source projects and models:

- [Qwen2-VL and Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for the retriever and Semantic Discriminator.
- [EVA-CLIP](https://github.com/baaivision/EVA) for offline candidate prefiltering.
- [FAISS](https://github.com/facebookresearch/faiss) for nearest-neighbor search.
- [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [Accelerate](https://github.com/huggingface/accelerate), and [DeepSpeed](https://github.com/microsoft/DeepSpeed) for model training.
- [Encyclopedic-VQA](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa), [InfoSeek](https://github.com/open-vision-language/infoseek), and [OK-VQA](https://okvqa.allenai.org/) for evaluation and training data.

Please follow the licenses and terms of each upstream model, dataset, and dependency.

---

## Citation

If you find KBMR useful, please cite:

```bibtex
@misc{xu2026kbmr,
      title={Beyond Visual Similarity: Entity-Aligned Retrieval for Knowledge-Based Visual Question Answering},
      author={Hangrui Xu and Zhengxian Wu and Yunyao Yu and Zhuohong Chen and Rui Cong and Xiangwen Deng and Zhifang Liu and Peng Jiao and Haoqian Wang},
      year={2026},
      eprint={2608.21450},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.21450},
}
```

<div align="center">
<sub>Entity-aligned retrieval for knowledge-based visual question answering.</sub>
</div>
