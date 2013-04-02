import unittest

from pade.main import *

class MainTest(unittest.TestCase):

    def test_fix_newlines(self):
        self.assertEquals(
            "a\n", fix_newlines("a"))
        self.assertEquals(
            "a b\n", fix_newlines("a\nb"))
        self.assertEquals(
            "a b\nc d\n", fix_newlines("a\nb\n\nc\nd"))
        foo = """\
                   The schema file \"{}\" already exists. If you want to
                   overwrite it, use the --force or -f argument.""".format("foo")
        self.assertEquals("The schema file \"foo\" already exists. If you want to overwrite it, use\nthe --force or -f argument.\n",
                          fix_newlines(foo))

    def test_validate_settings(self):
        schema = Schema(
            ['id', 'a', 'b', 'c', 'd'],
            ['feature_id', 'sample', 'sample', 'sample', 'sample'])

        schema.add_factor('treated', [False, True])
        schema.add_factor('dose', ['high', 'medium', 'low'])
        schema.add_factor('gender', ['male', 'female'])
        
        settings = Settings(
            stat='f',
            condition_variables=['treated'],
            block_variables=['gender'])

        validate_settings(schema, settings)

        with self.assertRaisesRegexp(UsageException, 'foo.*"dose", "gender", and "treated"'):
            settings = Settings(
                stat='f',
                condition_variables=['foo'])
            validate_settings(schema, settings)

    def test_quote_and_join(self):
        self.assertEquals('"a"', quote_and_join(['a']))
        self.assertEquals('"a" and "b"', quote_and_join(['a', 'b']))

if __name__ == '__main__':
    unittest.main()
