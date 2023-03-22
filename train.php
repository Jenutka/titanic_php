<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$extractor_num = new ColumnPicker(new CSV('train_im.csv', true), [
    'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived',
]);

$extractor_cat = new ColumnPicker(new CSV('train_im.csv', true), [
    'Sex', 'Embarked', 
]);


$logger->info('Processing numerical features');

$dataset_num = Labeled::fromIterator($extractor_num)
    ->apply(new NumericStringConverter());

$transformer_num = new MinMaxNormalizer();

$serializer_num = new RBX();

$transformer_num->fit($dataset_num);

$serializer_num->serialize($transformer_num)->saveTo(new Filesystem('trans_num.rbx'));

$dataset_num->apply($transformer_num);


$logger->info('Processing categorical features');

$dataset_cat = Unlabeled::fromIterator($extractor_cat);

$transformer_cat = new OneHotEncoder();

$serializer_cat = new RBX();

$transformer_cat->fit($dataset_cat);

$serializer_cat->serialize($transformer_cat)->saveTo(new Filesystem('trans_cat.rbx'));

$dataset_cat->apply($transformer_cat);


$logger->info('Joining features into one dataset');

$dataset = $dataset_num->join($dataset_cat);

$logger->info('Training and validating model');

$estimator = new RandomForest(new ClassificationTree(10), 300, 0.2, false);

$estimator->train($dataset);

use Rubix\ML\CrossValidation\Metrics\Accuracy;

$metric = new Accuracy();

$predictions = $estimator->predict($dataset);

$score = $metric->score($predictions, $dataset->labels());
$logger->info("Accuracy is $score");

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator = new PersistentModel($estimator, new Filesystem('model.rbx'));

    $estimator->save();

    $logger->info('Model saved as model.rbx');
}

?>
