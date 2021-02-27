# Lecture, Book
- CS224n
- 고려대학교 강필성 교수님 Unstructured Data Analysis (Graduate)
- 밑바닥부터 시작하는 딥러닝 2
- 김기현의 자연어 처리 딥러닝 캠프 파이토치편, 패스트캠퍼스 강의

# NLP tutorial
- [Deep Learning Basic](/deep%20learning%20basic)
- [NLP basic](/NLP%20basic)

# Practice, Project

## Basic
- 국민청원데이터 실습
    - Text preprocessing | Bow, TF-IDF | word2vec | Topic Modeling(LDA)
    - `참고링크` : [박조은님](https://github.com/corazzon)
- torchtext tutorial 

## Text Classification
- 문자단위 RNN으로 이름 분류하기 
    - `참고링크` : [pytorch 홈페이지](https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial.html)
- Convolutional Neural Networks for Sentence Classification (Yoon 2014)
- Fasttext

## Sentiment Analysis 
- 네이버 영화 댓글 감정 분석 
    - `참고링크` : [KoBert colab](https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb) , [Bert colab](https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP#scrollTo=DbsNMA8Idc3K)
- Using `transformers` bert sentiment classification and flask demo

## Machine Translation
- Seq2Seq, Transformer 구현
    - `참고링크` : [Aladdin Persson Youtube](https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ)
- 한국어-영어 번역 Seq2Seq, Transformer 구현 프로젝트 (`ing`)
  -  Seq2Seq model
  - Transformer model
  - DataLoader, Trainer 
  - Automatic mixed precision, Gradient Accumulator
  - Inference, BLEU
  - Beam Search

## Text Summarization

## Etc 
- image captioning 
    - `참고링크` : [Aladdin Persson Youtube](https://github.com/AladdinPerzon/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning)

# Paper Read
- Efficient Estimation of Word Representations in Vector Space
  - [`Paper Link`](https://arxiv.org/abs/1301.3781) | `My Summary` | [`My Code`](./Paper/My%20Code)
  - `Key Word` : word2vec
- Distributed Representations of Words and Phrases and their Compositionality (NIPS 2013)
  - [`Paper Link`](https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html) | `My Summary` | `My Code`
  - `Key Word` : word2vec, negative sampling
- Convolutional Neural Networks for Sentence Classification (EMNLP 2014)
  - [`Paper Link`](https://www.aclweb.org/anthology/D14-1181/) | `My Summary` | [`My Code`](./Paper/My%20Code)
  - `Key Word` : cnn, text classification
- Sequence to Sequence Learning with Neural Networks (2014)
  - [`Paper Link`](https://arxiv.org/abs/1409.3215) | `My Summary` | `My Code`
  - `Key Word` : seq2seq
- Attention is All You Need (NIPS 2017)
  - [`Paper Link`](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) | `My Summary` | `My Code`
  - `Key Word` : transformer, self-attention, no more rnn
- Self-Supervised Learning for Contextualized Extractive Summarization (arXiv 2019)
  - [`Paper link`](https://arxiv.org/abs/1906.04466) | `My Summary` | `My Code`
  - `Key Word` : extractive summarization, self-supervised

# Reference Link
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