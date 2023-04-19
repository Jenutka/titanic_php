# Rubix ML - Titanic - Machine Learning from Disaster

![Titanic - Machine Learning from Disaster](/images/img_titanic.jpg)

## Content

- [Installation](#installation)
- [Requirements](#requierments)
- [Recommended](#recommended)
- [Turorial](#tutorial)
- [Introduction](#introduction)
- [Extracting the training Data](#extracting-the-training-data)
- [Preprocessing the training Data](#preprocessing-the-training-data)
- [Saving transformers](#saving-transformers)
- [Model training](#model-training)
- [Saving the estimator](#saving-the-estimator)
- [Extracting the test Data](#extracting-the-test-data)
- [Loading transformers](#loading-transformers)
- [Preprocessing the test Data](#preprocessing-the-test-data)
- [Loading estimator](#loading-estimator)
- [Making predictions](#making-predictions)
- [Saving predictions](#saving-predictions)
- [Conclusion](#conclusion)


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
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;

$extractor = new ColumnPicker(new CSV('train.csv', true), [
    'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked', 'Survived',
]);
```

### Preprocessing the training Data

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
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\MissingDataImputer;

$minMaxNormalizer = new MinMaxNormalizer();
$oneHotEncoder = new OneHotEncoder();
$imputer = new MissingDataImputer();
```

Finally we create the
[Labeled](https://docs.rubixml.com/2.0/datasets/labeled.html) dataset and fit
with our preprocessing functions.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($extractor)
    ->apply(new NumericStringConverter())
    ->transformLabels($transformLabel);

$dataset->apply(new LambdaFunction($toPlaceholder, $dataset->types()))
    ->apply($imputer)
    ->apply($minMaxNormalizer)
    ->apply($oneHotEncoder);
```

### Saving Transformers 

Now because we want to apply the same fitted preprocessing on testing dataset
`test.csv` and predicting part will be realized with separated script
`predict.php`, we need to save our fitted transformers into serialized objects.
For this purpose we create new
[Filesystem](https://docs.rubixml.com/2.0/persisters/filesystem.html) objects
with using [RBX](https://docs.rubixml.com/2.0/serializers/rbx.html) file
format.

```php
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Serializers\RBX;

$serializer->serialize($imputer)->saveTo(new Filesystem('imputer.rbx'));
$serializer->serialize($minMaxNormalizer)->saveTo(new Filesystem('minmax.rbx'));
$serializer->serialize($oneHotEncoder)->saveTo(new Filesystem('onehot.rbx'));
```

### Model training

After we have prepared our data, we can train our predicting model. As
estimator we use
[RandomForest](https://docs.rubixml.com/2.0/classifiers/random-forest.html)
which is an ensemble of
[ClassificationTrees](https://docs.rubixml.com/2.0/classifiers/classification-tree.html)
which is good suited for our relatively small dataset.

```php
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new RandomForest(new ClassificationTree(10), 500, 0.8, false);

$estimator->train($dataset);
```

### Saving the estimator

Finally we save our predicting model for use with `predict.php` script. As in
case with transformers we use [Filesystem](https://docs.rubixml.com/2.0/persisters/filesystem.html) object with using [RBX](https://docs.rubixml.com/2.0/serializers/rbx.html) file format again. But now instead of serializing we use for saving predictive model [PersistentModel](https://docs.rubixml.com/2.0/persistent-model.html) object. For secure of overwriting existing model, we ask user for saving the new trained model.

```php
use Rubix\ML\PersistentModel;

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator = new PersistentModel($estimator, new Filesystem('model.rbx'));

    $estimator->save();

    $logger->info('Model saved as model.rbx');
}
```


Now we have finished our training part `train.php`, which we execute by calling
it from the command line.

```sh
$ php train.php
```

Now we can move on creating predicting part `predict.php`

### Extracting the test Data

For predicting part we need to extract our test data which don't contain
labels. By extracting we name the same features as for training set, but we
omit the target `Survived`.

```php
use Rubix\ML\Extractors\ColumnPicker;

$extractor = new ColumnPicker(new CSV('test.csv', true), [
    'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked',
]);
```

### Loading transformers

For transforming our test dataset we need to use the transformation fitted on
our training dataset. So we load and deserialize our previously saved
persistors.

```php
$persister_imputer = new Filesystem('imputer.rbx', true, new RBX());

$imputer = $persister_imputer->load()->deserializeWith(new RBX);

$persister_minMax = new Filesystem('minmax.rbx', true, new RBX());

$minMaxNormalizer = $persister_minMax->load()->deserializeWith(new RBX);

$persister_oneHot = new Filesystem('onehot.rbx', true, new RBX());

$oneHotEncoder = $persister_oneHot->load()->deserializeWith(new RBX);
```

### Preprocessing the test Data

For testing data we need to create new [Unlabeled](https://docs.rubixml.com/2.0/datasets/unlabeled.html) dataset object in which we pass our `$extractor`. As we have loaded our fitted transformers, we can apply them on this dataset object. As in case of training data we use function `$toPlaceholder` to map our missing values so [MissingDataImputer](#https://docs.rubixml.com/2.0/transformers/missing-data-imputer.html) can handle missing values.

```php
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\LambdaFunction;

$dataset = Unlabeled::fromIterator($extractor)
    ->apply(new NumericStringConverter());

$dataset->apply(new LambdaFunction($toPlaceholder, $dataset->types()))
    ->apply($imputer)
    ->apply($minMaxNormalizer)
    ->apply($oneHotEncoder);
```

### Loading estimator

Now we can load our persisted
[RandomForest](https://docs.rubixml.com/2.0/classifiers/random-forest.html)
estimator into our script using the static `load()` method.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Serializers\RBX;

$estimator = PersistentModel::load(new Filesystem('model.rbx'));
```

### Making predictions

For making predictions on our testing unlabeled dataset we call the `predict()`
method on our loaded estimator. We store our predicted classes under
`$predictions` variable.

```php
$predictions = $estimator->predict($dataset);
```

### Saving predictions

Now we need to prepare our stored predictions into required format so we can
submit it to Kaggle competition.

Firstly we map back our labels into `1` and
`O`. For this we create function `bin_mapper` which we pass as parameter into
built-in php function `array_map`.

```php
$predictions = $estimator->predict($dataset);

function bin_mapper($v)
{
    if ($v==="Survived") {
        return "1";
    } else {
        return "0";
    }
}

$predictions_mapped = array_map('bin_mapper', $predictions);
```

Now we extract `PassengerId` column from `test.csv`. Now we create array `$ids` for column `PassengerId`. We apply `array_unshift` function on both columns. Next we instantiate [CSV](https://docs.rubixml.com/2.0/extractors/csv.html) file `predictions.csv` and finaly export our two columns of data into it with `array_transpose` function.

```php
$predictions_mapped = array_map('bin_mapper', $predictions);

$logger->info('Saving predictions to csv');

$extractor = new ColumnPicker(new CSV('test.csv', true), ['PassengerId']);

$ids = array_column(iterator_to_array($extractor), 'PassengerId');

array_unshift($ids, 'PassengerId');
array_unshift($predictions_mapped, 'Survived');

$extractor = new CSV('predictions.csv');

$extractor->export(array_transpose([$ids, $predictions_mapped]));
```

Now we can our prediction script by calling it from the command line.

```sh
$ php predict.php
```
### Conclusion


