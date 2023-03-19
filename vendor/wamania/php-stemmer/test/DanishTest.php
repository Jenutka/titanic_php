<?php
namespace Wamania\Snowball\Tests;

use Wamania\Snowball\Stemmer\Danish;

class DanishTest extends \PHPUnit_Framework_TestCase
{
    /**
     * @dataProvider load
     */
    public function testStem($word, $stem)
    {
        $o = new Danish();

        $snowballStem = $o->stem($word);

        $this->assertEquals($stem, $snowballStem);
    }

    public function load()
    {
        return new CsvFileIterator('test/files/dk.txt');
    }
}
