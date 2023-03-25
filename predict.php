<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;
use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$extractor_num = new ColumnPicker(new CSV('test_im.csv', true), [
    'Pclass', 'Age', 'Fare', 'SibSp', 'Parch',
]);

$extractor_cat = new ColumnPicker(new CSV('test_im.csv', true), [
    'Sex', 'Embarked', 
]);


$logger->info('Processing numerical features');

$dataset_num = Unlabeled::fromIterator($extractor_num)
    ->apply(new NumericStringConverter());

$persister = new Filesystem('trans_num.rbx', true, new RBX());

$transformer_num = $persister->load()->deserializeWith(new RBX);

$dataset_num->apply($transformer_num);


$logger->info('Processing categorical features');

$dataset_cat = Unlabeled::fromIterator($extractor_cat)
    ->apply(new NumericStringConverter());

$persister = new Filesystem('trans_cat.rbx', true, new RBX());

$transformer_cat = $persister->load()->deserializeWith(new RBX);

$dataset_cat->apply($transformer_cat);


$logger->info('Joining features into one dataset');

$dataset = $dataset_num->join($dataset_cat);


$logger->info('Loading estimator into memory');

$estimator = PersistentModel::load(new Filesystem('model.rbx'));


$logger->info('Making predictions');

$predictions = $estimator->predict($dataset);


$logger->info('Saving predictions to csv');

$extractor = new ColumnPicker(new CSV('test_im.csv', true), ['PassengerId']);

$ids = array_column(iterator_to_array($extractor), 'PassengerId');

array_unshift($ids, 'PassengerId');
array_unshift($predictions, 'Survived');

$extractor = new CSV('predictions.csv');

$extractor->export(array_transpose([$ids, $predictions]));

?>
