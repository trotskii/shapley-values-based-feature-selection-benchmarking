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

class TestTextNormalization:
    def test_tokenize(self):
        test_str = "It would be unfair, to, demand. that people cease pirating files."
        expected = ['It', 'would', 'be', 'unfair', ',', 
            'to', ',', 'demand', '.', 'that', 'people', 'cease', 'pirating', 'files', '.']
        assert tp.tokenize(test_str) == expected
    
    def test_remove_punctuation(self):
        test_tokens = ['It', 'would', 'be', 'unfair', ',', 
            'to', ',', 'demand', '.', 'that', 'people', 'cease', 'pirating', 'files', '.']
        expected = ['It', 'would', 'be', 'unfair', 
            'to', 'demand', 'that', 'people', 'cease', 'pirating', 'files']
        assert tp.remove_punctuation(test_tokens) == expected
        
    def test_lemmatize(self):
        test_tokens = ['It', 'would', 'be', 'unfair', 
            'to', 'demand', 'that', 'people', 'cease', 'pirating', 'files']
        expected = ['It', 'would', 'be', 'unfair', 
            'to', 'demand', 'that', 'people', 'cease', 'pirating', 'file']
        assert tp.lemmatize(test_tokens) == expected
    
    def test_normalize_text(self):
        test_str = "It would be unfair, to, demand. that people cease pirating files."
        expected = "it would be unfair to demand that people cease pirating file"
        assert tp.normalize_text(test_str) == expected
        