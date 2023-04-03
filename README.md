# Rubix ML - Titanic - Machine Learning from Disaster

![Titanic - Machine Learning from Disaster](/images/img_titanic.jpg)

## Content

- [Installation](#installation)
- [Requirements](#requierments)
- [Recommended](#recommended)
- [Turorial](#tutorial)
- [Introduction](#introduction)
- [Extracting the Data(#extracting-the-data)

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

> **Note:** The source code for this example can be found in the [train.php](https://github.com/jenutka/titanic_php/train.php) file in project root.

### Extracting the Data

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;

$extractor = new ColumnPicker(new CSV('dataset.csv', true), [
]);

$dataset = Labeled::fromIterator($extractor);
```

