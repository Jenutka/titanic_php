<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Transformers\LambdaFunction;
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

$extractor = new ColumnPicker(new CSV('test.csv', true), [
    'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked',
]);

$logger->info('Processing features');

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

$dataset = Unlabeled::fromIterator($extractor)
    ->apply(new NumericStringConverter());

$persister_imputer = new Filesystem('imputer.rbx', true, new RBX());

$imputer = $persister_imputer->load()->deserializeWith(new RBX);

$persister_minMax = new Filesystem('minmax.rbx', true, new RBX());

$minMaxNormalizer = $persister_minMax->load()->deserializeWith(new RBX);

$persister_oneHot = new Filesystem('onehot.rbx', true, new RBX());

$oneHotEncoder = $persister_oneHot->load()->deserializeWith(new RBX);

$dataset->apply(new LambdaFunction($toPlaceholder, $dataset->types()))
    ->apply($imputer)
    ->apply($minMaxNormalizer)
    ->apply($oneHotEncoder);

$logger->info('Loading estimator into memory');

$estimator = PersistentModel::load(new Filesystem('model.rbx'));

$logger->info('Making predictions');

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

$logger->info('Saving predictions to csv');

$extractor = new ColumnPicker(new CSV('test.csv', true), ['PassengerId']);

$ids = array_column(iterator_to_array($extractor), 'PassengerId');

array_unshift($ids, 'PassengerId');
array_unshift($predictions_mapped, 'Survived');

$extractor = new CSV('predictions.csv');

$extractor->export(array_transpose([$ids, $predictions_mapped]));

?>
