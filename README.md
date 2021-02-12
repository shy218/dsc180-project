# DSC 180B Project

Our 3 main targets are data preparation, feature encoding and a test target. The data preparation target scrapes, cleans, and consolidates companies' 8-K documents. Furthermore, it curates features such as EPS as well as price movements for the given companies. The feature encoding target creates encoded text vectors for each 8-K: both unigrams and quality phrases outputed by the AutoPhrase method. Users can configure parameters for both these targets in the ./config files.


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
* `n_unigrams` sets the top n unigrams to be encoded
* `threshhold` takes quality phrases with a quality score above it to be encoded
* `train_split` is the desired size of the training data
* `test_split` is the desired size of the test data (the validation_split is equal to 1 - train_split - test_split)


## Test

* all the targets are the same as those in `Data Prep`, except `testing = true`
