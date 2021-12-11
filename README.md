# CourseProject 

The goal of this project is to develop a means of easily comparing topic modeling methods, such as [LDA](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) and [Top2Vec](https://arxiv.org/abs/2008.09470).  
This was implemented through a web app hosted [here](https://share.streamlit.io/wujameszj/courseproject/main/main.py).  A demo video is available [here](https://youtu.be/3oj7M-j5vPs).

![](data/windows-2pass500it120topic-short.PNG)


## Install

The web app instance hosted online has limited CPU and RAM resources.  
For heavy testing, it is recommended to run this app locally.  

1. Set up your environment `conda create -n myapp python=3.8` and `conda activate myapp`
1. Clone this repo and switch to project directory
1. Install dependencies `pip install -r requirements.txt`
1. Launch app in browser `streamlit run main.py`  


## Usage

The app has two components: 
- a sidebar for user input and control parameters
  1. choose dataset
  2. set parameters such as number of topics 
  3. search topic models with a keyword
- the main pane for displaying results
  - each algorithm has a dedicated column, lined up side-by-side for ease of comparison
  - topics are shown via wordclouds, where word size corresponds to term weight
  - documents returned from keyword search are displayed in height-adjustable boxes

Currently supported algorithms are LDA and Top2Vec.  A simplified overview and comparison of the two is available in this tech review [note](https://github.com/wujameszj/tech_review/blob/main/techreview.pdf).


## Reflection and Future Work

Although many features were planned for this app, a decision was made to make the first version simple, not overly cluttered with dozens of parameters and customization options. 

Ideas for future releases:
- expand available datasets for testing
  - load dataset from local directory or URL
  - scrape sites such as wikipedia and reddit based on user-defined timeframe and/or theme
- phrase/multi-term search
- show and compare time taken to train topic models and perform search 
- add options for lemmatization and word n-grams in vocabulary
- add more algorithms for comparison  
  
- provide users to more parameters for fine-tuning models
- offer customizable result display:
  - number of documents to show
  - default height of document display box
  - number of wordclouds
  - number of words per wordcloud


## Reference

LDA is implemented via [gensim](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html) while Top2Vec via [top2vec](https://github.com/ddangelov/Top2Vec).  Both Python packages are available via pip.