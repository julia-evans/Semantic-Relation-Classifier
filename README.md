# Semantic Relation Classifier

This semantic relation classifier was built by Julia Evans for the class Semantic Relations in Knowledge Repositories and in Texts at the IMS, University of Stuttgart.

## Packages

This classifier requires the following packages:
- re
- numpy
- sklearn
- nltk
- spacy


## Additional Files

The following files are also required:
- index.sense
- norms.dat

## Usage

The classifier can be access through the classifier.py file.

The bottom of the file is where the various options can be specified.

```python
if __name__ == '__main__':
    reps = 100
    train_file = 'relation-1-train.txt'
    perceptual = False
```

**reps**: the number of iterations over which the average score is computed

**train_file**: name of file to be used for training

**perceptual**: toggle between True and False to use perceptual features or not


```python
    # Get predictions
    test_file = 'relation-1-test.txt'
```

**test_file**: name of file to be used for testing (getting predictions)


```python
    outfile_name = 'relation-1'
    output_labels(test_file, svm_predicted_labels, outfile_name)
```

**outfile_name**: the name of the file containing the predictions ('-predictions' will be appended to this)

There is code to get an accuracy score using cross-validation and to output a file containing predictions.  After specifying the options described above, simply running the code will do both.