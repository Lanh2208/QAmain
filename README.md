# About this work
This is a work on a NLP course at our school, in particular, on open-domain question answering.

# Our approach
* The baseline is DrQA as suggested by the instructor. About DrQA, please refer to [DrQA's official repo](https://github.com/facebookresearch/DrQA) for more information (paper, intr, ciation, license, ...)
* Improve the retrieval stage with better schemes (researching)
* Leverage Huggingface transformers framework, with better models such as BERT.
* Apply the methods to our Vietnamese language.


# Report summary

### Pipeline: en

| Pipeline              | Open SQuAD-dev (EM/F1) |
|-----------------------|:-------------------:|
| DrQA-biLSTM           | 29.5 / -          |
| DrQA-transformers     | 31.9 / 36.9       |
| pyserini-transformers | 37.3 / 43.9       |

with transformers model being used as `distilbert-base-cased-distilled-squad`

### Vi readers
|       Data                    |     Model       | Params  | Throughput | vi-wiki-test |   MLQA-dev  |
|:-----------------------------:|:---------------:|:-------:|:----------:|:------------:|:-----------:|
| SQuAD-translate (~100k pairs) | PhoBERT-base    |   135M  |   17.6/s   |  45.0 / 63.6 | 37.6 / 57.2 |
|                               | XLM-R-base      |   270M  |   15.1/s   |  45.9 / 65.5 | 40.9 / 59.8 |
|   MLQA + XQuAD (~7000 pairs)  | XLM-R-base      |   270M  |   15.1/s   |  52.3 / 67.0 | 44.4 / 64.5 |
|                               | XLM-R-large     |   550M  |    4.9/s   |  60.4 / 73.9 | 51.1 / 70.4 |

# Installation
* Clone the repo & run: `python setup.py develop`
* Install [Java 11](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html) with $JAVA_HOME environment variable set up correctly, according to [pyserini](https://github.com/castorini/pyserini#package-installation)
* Data (db, models, index file...) (to be updated)
  * databases
  * [index files](https://drive.google.com/drive/folders/1iUyuBj-h335hY6_hjFMzjCHoo0sO1VYB?usp=sharing)
  * [xlm-roberta-base-mlqa-xquad-vi-wiki](https://drive.google.com/drive/folders/1oAnQ8sInUCIMlB00wXc22ThTvwEkVFXs?usp=sharing)
  * [xlm-roberta-large-mlqa-xquad-vi-wiki](https://drive.google.com/drive/folders/14JRCuFq7bEGqI1YnQpY_lzJyw1EUZ-GA?usp=sharing)

# Usage

### Interactive

pyserini-transformers: vietnamese
```
python scripts\pipeline_transformers\interactive.py  
  --reader-model <path to model folder or Huggingface model name> \
  --retriever pyserini-bm25 \
  --index-path <path to index folder> \
  --index-lan vi  \
  --num-workers 4 
```

### Web UI
At drqa-webui submodule

Showcase
![Sample question](report/sample.png)

# Future work
Still, there are a lot to improve, so many new novel methods and ideas to implement
* Better retriever scheme...
* Employ better QA models
* [UIT-VIQuAD dataset](https://www.aclweb.org/anthology/2020.coling-main.233/) for Vietnamese