# DSC 180B Project

Our 3 main targets are data preparation, feature encoding and a test target. The data preparation target scrapes, cleans, and consolidates companies' 8-K documents. Furthermore, it curates features such as EPS as well as price movements for the given companies. The feature encoding target creates encoded text vectors for each 8-K: both unigrams and quality phrases outputed by the AutoPhrase method. Users can configure parameters for both these targets in the ./config files.

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
