import unittest
import sys
import os
#hack to import from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessor import DataPreprocessor

class TestSentimentLogic(unittest.TestCase):
    def setUp(self):
        self.processor = DataPreprocessor("dummy.csv")

    def test_clean_html(self):
        #checking if HTML tags are removed
        raw = "Movie was good <br /> very good"
        clean = self.processor.clean_text(raw)
        self.assertEqual(clean.strip(), "movie was good   very good")

    def test_clean_punctuation(self):
        #checking if punctuation is removed
        raw = "Bad movie!!!"
        clean = self.processor.clean_text(raw)
        self.assertEqual(clean.strip(), "bad movie")

if __name__ == '__main__':
    unittest.main()