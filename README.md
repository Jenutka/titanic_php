# Rubix ML - Titanic - Machine Learning from Disaster

![Titanic - Machine Learning from Disaster](/images/img_titanic.jpg)

## Content

- [Installation](#installation)
- [Requirements](#requierments)
- [Recommended](#recommended)
- [Turorial](#tutorial)
- [Introduction](#introduction)
- [Extracting the Data](#extracting-the-data)

An example Rubix ML project that predicts which passengers survived the Titanic shipwreck using a Random Forest clasiffier and a very famous dataset from a [Kaggle competition] (https://www.kaggle.com/competitions/titanic). In this tutorial, you'll learn about classification and advanced preprocessing techniques. By the end of the tutorial, you'll be able to submit your own predictions to the Kaggle competition.

- **Difficulty:** Medium
- **Training time:** Minutes

From Kaggle:

> This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML competitions
>
> The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
> 
> In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled train.csv and the other is titled test.csv.
>
> Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.
>
> The test.csv dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.
>
> Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.


## Installation
Clone the project locally using [Composer](https://getcomposer.org/):
```sh
$ composer create-project jenutka/titanic_php
```

## Requirements
- [PHP](https://php.net) 7.4 or above

#### Recommended
- [Tensor extension](https://github.com/RubixML/Tensor) for faster training and inference
- 1G of system memory or more

## Tutorial

### Introduction
[Kaggle](https://www.kaggle.com) is a platform that allows you to test your data science skills by engaging with contests. This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works. The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. The sinking of the Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

We'll choose [Random Forest](https://docs.rubixml.com/2.0/classifiers/random-forest.html) as our learner since it offers good performance and is capable of handling both categorical and continuous features.

> **Note:** The source code for this example can be found in the [train.php](https://github.com/jenutka/titanic_php/train.php) and in [predict.php](https://github.com/jenutka/titanic_php/predict.php) file in project root.

### Script desription

The script is separated into two parts:

- **[train.php](https://github.com/jenutka/titanic_php/train.php)** For extracting training data from csv, feature transformation, training and saving predicting model
- **[predict.php](https://github.com/jenutka/titanic_php/predict.php)** For
  loading trained predicting model and for making and exporting predictions from unlabeled dataset 

The training data are given to us in `train.csv` which has features and labels for training the model. We train the model from the whole dataset, because our testing data `test.csv` are unlabeled, so in this case we can only validate predictions with Kaggle competition.

### Extracting the Data

Each feature is defined by column in `train.csv`. For our purpose we only
choose preferable features with the most informative value for our model. These
are continuos and categorical. For extraction from `train.csv` to dataset object we use [Column
Picker](https://docs.rubixml.com/latest/extractors/column-picker.html). As the
last extracted feature we name our target (label) feature `Survived`.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;

$extractor = new ColumnPicker(new CSV('train.csv', true), [
    'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked', 'Survived',
]);
```

### Preprocessing the Data

As in the `*.csv` file are missing values, we need to preprocess them for use
with
[MissingDataImputer](https://docs.rubixml.com/2.0/transformers/missing-data-imputer.html).
For this purpose we use
[LambdaFunction](https://docs.rubixml.com/2.0/transformers/lambda-function.html)
in which we pass mapping function `$toPlaceholder`.

```php
use Rubix\ML\Transformers\LambdaFunction;

$toPlaceholder = function (&$sample, $offset, $types) {
    foreach ($sample as $column => &$value) {
        if (empty($value) && $types[$column]->isContinuous()) {
            $value = NAN;
        }
        else if (empty($value) && $types[$column]->isCategorical()) {
            $value = '?';
        }
    }
};
```

The target values in `train.csv` are `0` and `1`. Our training model can handle
it as floating number so we should map these as categorical variable `Dead` and
`Survived`.

```php
$transformLabel = function ($label) {
    return $label == 0 ? 'Dead' : 'Survived';
};
```

For numerical variables we transform data with
[MinMaxNormalize](https://docs.rubixml.com/2.0/transformers/min-max-normalizer.html).For
categorical variable we use
[OneHotEncoder](https://docs.rubixml.com/2.0/transformers/one-hot-encoder.html).
For these two transformers and for
[MissingDataImputer](https://docs.rubixml.com/2.0/transformers/missing-data-imputer.html)
we instantiate new objects.

```php
$minMaxNormalizer = new MinMaxNormalizer();
$oneHotEncoder = new OneHotEncoder();
$imputer = new MissingDataImputer();
```

Finally we create the
[Labaled](https://docs.rubixml.com/2.0/datasets/labeled.html) dataset and fit
with our preprocessing functions.

```php
$dataset = Labeled::fromIterator($extractor)
    ->apply(new NumericStringConverter())
    ->transformLabels($transformLabel);

$dataset->apply(new LambdaFunction($toPlaceholder, $dataset->types()))
    ->apply($imputer)
    ->apply($minMaxNormalizer)
    ->apply($oneHotEncoder);
```

### Saving serializers

Now because we want to apply the same fitted preprocessing on testing dataset
`test.csv` and predicting part will be realized with separated script
`predict.php`, we need to save our fitted transformers into serialized objects.

```php
$serializer->serialize($imputer)->saveTo(new Filesystem('imputer.rbx'));
$serializer->serialize($minMaxNormalizer)->saveTo(new Filesystem('minmax.rbx'));
$serializer->serialize($oneHotEncoder)->saveTo(new Filesystem('onehot.rbx'));
```


