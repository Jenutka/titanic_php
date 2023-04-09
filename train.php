<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

ini_set('memory_limit', '-1');

$logger = new Screen();
$serializer = new RBX();

$logger->info('Loading data into memory');

$extractor = new ColumnPicker(new CSV('train.csv', true), [
    'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked', 'Survived',
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

$transformLabel = function ($label) {
    return $label == 0 ? 'Dead' : 'Survived';
};

$dataset = Labeled::fromIterator($extractor)
    ->apply(new NumericStringConverter())
    ->transformLabels($transformLabel);

$minMaxNormalizer = new MinMaxNormalizer();
$oneHotEncoder = new OneHotEncoder();
$imputer = new MissingDataImputer();

$dataset->apply(new LambdaFunction($toPlaceholder, $dataset->types()))
    ->apply($imputer)
    ->apply($minMaxNormalizer)
    ->apply($oneHotEncoder);

$serializer->serialize($imputer)->saveTo(new Filesystem('imputer.rbx'));
$serializer->serialize($minMaxNormalizer)->saveTo(new Filesystem('minmax.rbx'));
$serializer->serialize($oneHotEncoder)->saveTo(new Filesystem('onehot.rbx'));

$logger->info('Training and validating model');

$estimator = new RandomForest(new ClassificationTree(10), 500, 0.8, false);

$estimator->train($dataset);

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
