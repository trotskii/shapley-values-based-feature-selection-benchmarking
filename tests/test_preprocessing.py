from cgi import test
import src.preprocessing.text_preprocessing as tp 

class TestExpandContractions:
    def test_expand_contractions_lowercase(self):
        test_str = "i've got the power"
        expected = "i have got the power"
        assert tp.expand_contractions(test_str)  == expected
    
    def test_expand_contractions_uppercase(self):
        test_str = "I've got the power It's great"
        expected = "i have got the power it is great"
        assert tp.expand_contractions(test_str) == expected

    def test_expand_contractions_mixedcase(self):
        test_str = "I've got the power it's great"
        expected = "i have got the power it is great"
        assert tp.expand_contractions(test_str) == expected