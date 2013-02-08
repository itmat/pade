import unittest

from pade.main import *

class ResultTableTest(unittest.TestCase):

    def setUp(self):
        stats = np.array([np.arange(100) for x in range(4)])
        self.table = ResultTable(
            means=np.arange(400).reshape(100, 4),
            coeffs=np.arange(400).reshape(100, 4),
            stats=stats,
            feature_ids=np.arange(100),
            scores=stats/100.0
            )

    def test_filter(self):
        table = self.table
        filtered = table.filter_by_score(0.745)
        self.assertEquals(len(filtered),  25)
        self.assertEquals(len(table),    100)
        
    def test_pages(self):
        pages = list(self.table.pages(rows_per_page=30))
        self.assertEquals(len(pages), 4)
        self.assertEquals(len(pages[0]), 30)
        self.assertEquals(len(pages[1]), 30)
        self.assertEquals(len(pages[2]), 30)
        self.assertEquals(len(pages[3]), 10)
        
