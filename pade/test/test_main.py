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

if __name__ == '__main__':
    unittest.main()
