- NLP를 전공한 석사생 수준까지는 아니지만 기본적으로 텍스트데이터를 다룰 수 있는 사람이 되는 것이 목표

<!-- TOC -->

- [Project 2022](#project-2022)
- [Project 2020](#project-2020)
  - [Basic](#basic)
  - [Text Classification](#text-classification)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Machine Translation](#machine-translation)
  - [Etc](#etc)
- [Paper Read](#paper-read)
- [Reference](#reference)

<!-- /TOC -->

# Project 2022

- 2022년에 하고 있는 실습 프로젝트, 논문 구현

# Project 2020

- 2020년에 했던 실습 프로젝트, 논문 구현

## Basic

<details>
  <summary>Deep Learning Basic</summary>

- pytorch, pytorch lighting, cnn, rnn
- [code](/Project%202020/%5BBasic%5D%20deep%20learning%20basic/)
</details>

<details>
  <summary>NLP basic</summary>
  
- tokenize, torchtext, woed2vec, Doc2vec
- [code](/Project%202020/%5BBasic%5D%20NLP%20basic/)
</details>

<details>
  <summary>torchtext tutorial</summary>

- [code](/Project%202020/%5BBasic%5D%20TorchText%20tutorial/)
</details>

<details>
  <summary>국민청원데이터 실습</summary>

- Text preprocessing, Bow, TF-IDF, word2vec, Topic Modeling(LDA)
- [code](/Project%202020/%5BBasic%5D%20%EA%B5%AD%EB%AF%BC%EC%B2%AD%EC%9B%90%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%A4%EC%8A%B5/)
- reference: [박조은님](https://github.com/corazzon)
</details>

## Text Classification

<details>
  <summary>Convolutional Neural Networks for Sentence Classification (Yoon 2014)</summary>

- 논문 구현
- [code](</Project%202020/%5BText%20Classification%5D%20CNN%20(2014%20YoonKim)/>)
</details>

<details>
  <summary>Fasttext</summary>

- [code](/Project%202020/%5BText%20Classification%5D%20fasttext/)
</details>

<details>
  <summary>문자단위 RNN으로 이름 분류하기</summary>

- [code](/Project%202020/%5BText%20Classification%5D%20%EB%AC%B8%EC%9E%90%EB%8B%A8%EC%9C%84%20RNN%EC%9C%BC%EB%A1%9C%20%EC%9D%B4%EB%A6%84%20%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0/)
- reference: [pytorch 홈페이지](https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial.html)
</details>

## Sentiment Analysis

<details>
  <summary>네이버 영화 댓글 감정 분석</summary>

- bert sentiment classification
- [code](/Project%202020/%5BSentiment%20Analysis%5D%20%5BKobert%5D%20%EB%84%A4%EC%9D%B4%EB%B2%84%20%EC%98%81%ED%99%94%20%EB%8C%93%EA%B8%80%20%EA%B0%90%EC%A0%95%EB%B6%84%EC%84%9D/)
</details>

## Machine Translation

<details>
  <summary>Seq2Seq, Transformer 구현 연습</summary>

- [code](/Project%202020/%5BMachine%20Translation%5D%20Seq2Seq%2C%20transformer/)
- reference : [Aladdin Persson Youtube](https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ)
</details>

## Etc

<details>
  <summary>image captioning</summary>

- reference : [Aladdin Persson Youtube](https://github.com/AladdinPerzon/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning)
</details>

# Paper Read

- Efficient Estimation of Word Representations in Vector Space
  - [`Paper Link`](https://arxiv.org/abs/1301.3781) | `My Summary` | [`My Code`]
  - `Key Word` : word2vec
- Distributed Representations of Words and Phrases and their Compositionality (NIPS 2013)
  - [`Paper Link`](https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html) | `My Summary` | `My Code`
  - `Key Word` : word2vec, negative sampling
- Convolutional Neural Networks for Sentence Classification (EMNLP 2014)
  - [`Paper Link`](https://www.aclweb.org/anthology/D14-1181/) | `My Summary` | [`My Code`](</Project%202020/Paper/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification%20(EMNLP%202014)/>)
  - `Key Word` : cnn, text classification
- Sequence to Sequence Learning with Neural Networks (2014)
  - [`Paper Link`](https://arxiv.org/abs/1409.3215) | `My Summary` | `My Code`
  - `Key Word` : seq2seq
- Attention is All You Need (NIPS 2017)
  - [`Paper Link`](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) | `My Summary` | `My Code`
  - `Key Word` : transformer, self-attention
- Self-Supervised Learning for Contextualized Extractive Summarization (arXiv 2019)
  - [`Paper link`](https://arxiv.org/abs/1906.04466) | `My Summary` | `My Code`
  - `Key Word` : extractive summarization, self-supervised

# Reference

- CS224n
- 고려대학교 강필성 교수님 Unstructured Data Analysis (Graduate)
- 밑바닥부터 시작하는 딥러닝 2
- 김기현의 자연어 처리 딥러닝 캠프 파이토치편
- 한국어 임베딩
- [KoGPT2-fine tuning](https://github.com/gyunggyung/KoGPT2-FineTuning)
- [Beomi/KcBERT](https://github.com/Beomi/KcBERT)
- [CMU NLP 강의](http://demo.clab.cs.cmu.edu/NLP/)
- [AdapterHub Documentation](https://docs.adapterhub.ml/index.html)
- [Allennlp](https://github.com/allenai/allennlp)
- [kangpilsung/text analytics](https://github.com/pilsung-kang/Text-Analytics)
- [sk planet Bert 강의](https://www.youtube.com/watch?v=qlxrXX5uBoU&list=PL9mhQYIlKEhcIxjmLgm9X5BUtW5jMLbZD)
- [The super duper NLP repo](https://notebooks.quantumstat.com/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter)
- [kss : korean sentence splitter](https://github.com/hyunwoongko/kss)
- [SKT-AI/KoBART](https://github.com/SKT-AI/KoBART)
- [skt KoBART를 transformers로](https://github.com/hyunwoongko/kobart-transformers)
- [Korpora](https://github.com/ko-nlp/Korpora)
- [한국어 악성댓글 데이터셋](https://github.com/ZIZUN/korean-malicious-comments-dataset)
- [박찬준님의 다양한 강의노트](https://github.com/Parkchanjun)
