import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# QWERTY neighbors for typo simulation
qwerty_neighbors = {
        "a": "qwsz",
        "b": "vghn",
        "c": "xdfv",
        "d": "serfcx",
        "e": "wsdr",
        "f": "drtgv",
        "g": "ftyhbv",
        "h": "gyujnb",
        "i": "ujko",
        "j": "huikm",
        "k": "jiolm",
        "l": "kop",
        "m": "njk",
        "n": "bhjm",
        "o": "iklp",
        "p": "ol",
        "q": "wa",
        "r": "edft",
        "s": "wedxza",
        "t": "rfgy",
        "u": "yhji",
        "v": "cfgb",
        "w": "qase",
        "x": "zsdc",
        "y": "tghu",
        "z": "asx"
    }


def custom_transform(example):
   
    ################################
    ##### YOUR CODE BEGINGS HERE ###
   
    text = example["text"]
    words = word_tokenize(text)
    detok = TreebankWordDetokenizer()

    new_words = []
    for w in words:
        original = w
        #replace like 20% with synonyms using wordnet
        if random.random()< 0.2:
            synonyms = wordnet.synsets(w)
            lemmas = []

            for synonym in synonyms:
                for lemma in synonym.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != w.lower():
                        lemmas.append(name)

            if len(lemmas) > 0:
                w = random.choice(lemmas)

        #introduce typos aruond 10% prob
        if random.random() < 0.1:
            #choose a letter position to corrupt if possible
            chars = list(w)
            typo_indices = [i for i, c in enumerate(chars) if c.lower() in qwerty_neighbors]

            if len(typo_indices) > 0:
                idx = random.choice(typo_indices)
                c = chars[idx].lower()
                replacement = random.choice(qwerty_neighbors[c])
                chars[idx] = replacement  # replace letter with neighbor
                w = "".join(chars)

        new_words.append(w)

    ##now we take new words and reformat them to normal text
    transformed_text = detok.detokenize(new_words)

    example["text"] = transformed_text

    ##### YOUR CODE ENDS HERE ######

    return example
