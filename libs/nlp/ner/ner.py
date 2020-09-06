import spacy
import random
import os
from spacy.util import compounding
from spacy.util import minibatch
from tqdm import tqdm

from pathlib import Path

class NER:
    def __init__(self):
        self.new_model = True
        self.nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    def load_model(self, path):
        self.new_model = False
        self.nlp = spacy.load(path)  # load existing spaCy model
        print("Loaded model '%s'" % path)

    def save_model(self, path):
        path = Path(path)
        if not os.path.exists(path.parent):
            os.mkdirs(path.parent)

        self.nlp.meta["name"] = path.name
        self.nlp.to_disk(path)
        print("Saved model to", path)

    #
    #  train_data = [(
    #       "some_text",
    #       {
    #           "entities": [(start, end, 'lable_name'), ...]
    #       }
    #  ), ...]
    #
    def train(self, train_data, n_iter=20, drop=0.5):
        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.create_pipe("ner")
            self.nlp.add_pipe(ner, last=True)
        # otherwise, get it so we can add labels
        else:
            ner = self.nlp.get_pipe("ner")

        # add labels
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):  # only train NER
            # sizes = compounding(1.0, 4.0, 1.001)
            # batch up the examples using spaCy's minibatch
            if self.new_model:
                self.nlp.begin_training()
            else:
                self.nlp.resume_training()

            for itn in tqdm(range(n_iter)):
                random.shuffle(train_data)
                batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=drop,  # dropout - make it harder to memorise data
                        losses=losses,
                    )

                print("Losses", losses)