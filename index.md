# Using AutoPhrase for Text Mining in 8-K reports

Our 3 main targets are data preparation, feature encoding and a test target. The data preparation target scrapes, cleans, and consolidates companies' 8-K documents. Furthermore, it curates features such as EPS as well as price movements for the given companies. The feature encoding target creates encoded text vectors for each 8-K: both unigrams and quality phrases outputed by the AutoPhrase method. Users can configure parameters for both these targets in the ./config files.

## Motivation

Stock market is one of the most popular markets that the investors like to put their money in. There are millions of investors who participate in the stock market investment directly or indirectly, such as by mutual fund, defined-benefit plan. Certainly, there are many people who research on the stock market, and they all know the information takes an important role in the decision making. According to the Strong Form Market Efficiency Theorem, the stock price is only determined by the new information; otherwise, it will be a random walk. However, there are thousands of information in this market everyday, and the investors can only pay attention to few of them. Therefore, the investors, especially the individual investors without the help of professional financial analyst, can only get the parts of the whole information, so his investment decisions may be biased. In this project, we want to solve this real-world problem to apply the Phrase mining technique to forecast the stock price and help the investors to make the decision.

## Datasets

* [8-K Reports Database](https://nlp.stanford.edu/projects/lrec2014-stock/8K.tar.gz) 1.4G
* [8-K Reports (RAR)](https://www.dropbox.com/s/pu08xl15b8y7jvu/8K.rar?dl=0) 701M
* [8-K Reports With Info in JSON](https://www.dropbox.com/s/f7hxtruvkbu8ke9/8k.json?dl=0) 6.1G
* [Merged Dataset with Price label](https://www.dropbox.com/s/872mfi57vygyhbw/merged_all_data.csv?dl=0) 618M
* [Financial Quality Terms Web Mining from Investopedia](https://www.dropbox.com/s/ms1kh6kftrbpjz0/finance_quality.txt?dl=0)


## Data Prep

* `data_dir` is the file path to download files: 8-K's, EPS, etc.
* `raw_dir` is the directory to the raw data
* `raw_8k_fp` is the file path with newly downloaded 8-K's (should be the same as to_dir)
* `raw_eps_fp` is the file path with newly downloaded EPS information (should be the same as to_dir)
* `processed_dir` is the directory to the processed data
* `testing` is the status of whether we are doing testing (by default is false)


## Feature Encoding

* `data_file` is the file path to outputed .csv file from the data prep target
* `phrase_file` is the file path to the quality phrases outputted by AutoPhrase
* `n_unigrams` sets the top n unigrams to be encoded based on PMI
* `threshhold` takes quality phrases with a quality score above it to be encoded
* `train_split` is the desired size of the training data
* `test_split` is the desired size of the test data (the validation_split is equal to 1 - train_split - test_split)


## Test

* all the targets are the same as those in `Data Prep`, except `testing = true`

![Sicherung vorbereiten](/ROC.png)
