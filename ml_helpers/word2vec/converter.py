import gensim
import numpy as np
import re
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize, TweetTokenizer
import collections

URL_STRING = '__url_string__'

DIGIT_SYMBOL = '##'
SPECIAL_SYMBOLS = [
    ' ',
    ',',
    '.',
    '!',
    '?',
    ':',
    '<',
    '>',
    '-',
    '_',
    ')',
    '(',
    '*',
    '\'',
    '\"',
    '`',
    '#',
    ';',
    '/',
    '\\',
    '|',
    '~',
    ']',
    '['
]
WORD_TO_EXLUDE = [
    'a',
    'and',
    'of',
    'to',
    'an',
    *SPECIAL_SYMBOLS
]

TOKEN_SYMBOLS = [
    *SPECIAL_SYMBOLS
]

WORDS_TO_REPLACE = {
    ":)": "smile",
    ";)": "smile",
    ";(": "sadness",
    ":(": "sadness",
    "usl": "fighting",
    URL_STRING: "url"
}

LIST_OF_KNOWN_WORDS = ['url', DIGIT_SYMBOL]

class TokenizerType(enumerate):
    split_by_bag_of_symbols = 0
    word_tokenize = 1
    tweet_tokenizer = 2

class Converter:
    def __init__(self, tokenizer_type=TokenizerType.word_tokenize):
        self.tokenizer_type = tokenizer_type
        self.clear_statistic()
        self.spell_checker = SpellChecker()

        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            'helpers/word2vec/GoogleNews-vectors-negative300.bin',
            binary=True,
            datatype=np.float16)

    def tokenize(self, sentence):
        if self.tokenizer_type == TokenizerType.tweet_tokenizer:
            return TweetTokenizer().tokenize(sentence)
        elif self.tokenizer_type == TokenizerType.word_tokenize:
            return word_tokenize(sentence.lower())
        elif self.tokenizer_type == TokenizerType.split_by_bag_of_symbols:
            return re.split('|'.join(TOKEN_SYMBOLS), sentence.lower())

    def clear_statistic(self):
        self.all_input_words = set()
        self.all_output_words = set()

        self.unknown_words = collections.defaultdict(int)
        self.known_words = collections.defaultdict(int)
        pass

    def convert_word(self, word):
        self.all_input_words.add(word)

        if word.startswith('http'):
            word = URL_STRING

        if word.isdigit():
            word = DIGIT_SYMBOL

        word = WORDS_TO_REPLACE[word] if word in WORDS_TO_REPLACE else word

        word = self.reduce_lengthening(word)

        if word not in LIST_OF_KNOWN_WORDS:
            word = self.spell_checker.correction(word)

        if (word in WORD_TO_EXLUDE):
            return None, None

        self.all_output_words.add(word)

        vec = None
        try:
            vec = self.word2vec.get_vector(word)
            self.known_words[word] += 1
        except:
            self.unknown_words[word] += 1

        return vec, word

    def convert_words(self, words):
        tmp = [self.convert_word(word) for word in words]

        vectors = [i[0] for i in tmp]
        converted_words = [i[1] for i in tmp]

        return vectors, converted_words

    def convert_sentence(self, sentence):
        sentence = self.paint_urls(sentence)
        #sentence = self.paint_digits(sentence)

        tokens = self.tokenize(sentence.lower())

        converted_words = self.convert_words(tokens)

        return converted_words

    def convert_sentences(self, sentences):
        tmp = sentences.apply(self.convert_sentence)
        vectors = [i[0] for i in tmp]
        converted_sentences = [i[1] for i in tmp]

        return vectors, converted_sentences

    # Заменяем все слова которые начинаются на http на URL_STRING
    def paint_urls(self, sentence):
        splited = sentence.split(' ')
        for i in range(len(splited)):
            if splited[i].startswith("http"):
                splited[i] = URL_STRING
        return ' '.join(splited)

    # Заменяем все цифры на #
    def paint_digits(self, sentence):
        return re.sub('\d', '#', sentence)

    def reduce_lengthening(self, text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

