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


def custom_transform(example):
   
    ################################
    ##### YOUR CODE BEGINGS HERE ###
    text = example["text"]

    words = word_tokenize(text)
    new_words = []

    for w in words:
        # 20% chance of replacement
        if random.random() < 0.2:
            syns = wordnet.synsets(w)
            lemmas = []

            for syn in syns:
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != w.lower():
                        lemmas.append(name)

            if len(lemmas) > 0:
                new_words.append(random.choice(lemmas))
                continue

        new_words.append(w)

    transformed_text = TreebankWordDetokenizer().detokenize(new_words)

    
    return {
        "text": transformed_text,
        "label": example["label"]
    }
    ##### YOUR CODE ENDS HERE ######

    return example
