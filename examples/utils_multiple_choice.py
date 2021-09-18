# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function


import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
from typing import List
from transformers import PreTrainedTokenizer
import random
# from mctest import parse_mc
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None, nsp_label=None, context_sents=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label,
                 pq_end_pos=None

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
        self.pq_end_pos=pq_end_pos


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, 'train/high')
        middle = os.path.join(data_dir, 'train/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, 'dev/high')
        middle = os.path.join(data_dir, 'dev/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, 'test/high')
        middle = os.path.join(data_dir, 'test/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (k, data_raw) in enumerate(lines):
            # if k >10:
            #     break
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw['answers'][i]) - ord('A'))
                question = data_raw['questions'][i]
                options = data_raw['options'][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article], # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth))
        return examples

class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != 'label':
            raise ValueError(
                "For training, the input file must contain a label column."
            )

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts = [line[4], line[4], line[4], line[4]],
                endings = [line[7], line[8], line[9], line[10]],
                label=line[11]
            ) for line in lines[1:]  # we skip the line with the column names
        ]

        return examples



class DreamProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def __init__(self):
        self.data_pos={"train":0,"dev":1,"test":2}
        self.D = [[], [], []]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "test")
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""

        if len(self.D[self.data_pos[type]])==0:
            random.seed(42)
            for sid in range(3):
                with open([data_dir + "/" + "train.json", data_dir + "/"  + "dev.json",
                           data_dir + "/" + "test.json"][sid], "r") as f:
                    data = json.load(f)
                    if sid == 0:
                        random.shuffle(data)
                    for i in range(len(data)):
                        for j in range(len(data[i][1])):
                            d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                            for k in range(len(data[i][1][j]["choice"])):
                                d += [data[i][1][j]["choice"][k].lower()]
                            d += [data[i][1][j]["answer"].lower()]
                            self.D[sid] += [d]
        data=self.D[self.data_pos[type]]
        examples = []
        for (i, d) in enumerate(data):
            for k in range(3):
                if data[i][2 + k] == data[i][5]:
                    answer = str(k)

            label = answer
            guid = "%s-%s-%s" % (type, i, k)

            text_a = data[i][0]

            text_c = data[i][1]
            examples.append(
                InputExample(example_id=guid,contexts=[text_a,text_a,text_a],question=text_c,endings=[data[i][2],data[i][3],data[i][4]],label=label))

        return examples

class MctestProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def __init__(self):
        self.data_pos={"train":0,"dev":1,"test":2}
        self.D = [[], [], []]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "test")
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""
        if "mc500" in data_dir:
            pref="mc500."
        if "mc160" in data_dir:
            pref="mc160."
        article, question, ct1, ct2, ct3, ct4, y, q_id = parse_mc(os.path.join(data_dir,pref+type+".tsv"), os.path.join(data_dir,pref+type+".ans"))
        examples = []
        for i, (s1, s2, s3, s4, s5, s6, s7, s8), in enumerate(zip(article, question, ct1, ct2, ct3, ct4, y, q_id)):
            examples.append(InputExample(example_id=s8,contexts=[s1,s1,s1,s1],question=s2,endings=[s3,s4,s5,s6],label=str(s7)))
        return examples

def read_race(path):
    with open(path, 'r', encoding='utf_8') as f:
        data_all = json.load(f)
        article = []
        question = []
        st = []
        ct1 = []
        ct2 = []
        ct3 = []
        ct4 = []
        y = []
        q_id = []
        for instance in data_all:

            ct1.append(' '.join(instance['options'][0]))
            ct2.append(' '.join(instance['options'][1]))
            ct3.append(' '.join(instance['options'][2]))
            ct4.append(' '.join(instance['options'][3]))
            question.append(' '.join(instance['question']))
            # q_id.append(instance['q_id'])
            q_id.append(0)
            art = instance['article']
            l = []
            for i in art: l += i
            article.append(' '.join(l))
            # article.append(' '.join(instance['article']))
            y.append(instance['ground_truth'])
        return article, question, ct1, ct2, ct3, ct4, y, q_id

class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            return lines


    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        #There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id = id,
                        question=question,
                        contexts=[options[0]["para"].replace("_", ""), options[1]["para"].replace("_", ""),
                                  options[2]["para"].replace("_", ""), options[3]["para"].replace("_", "")],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth))

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    truncation_strategy='longest_first'
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm.tqdm(examples, desc="convert examples to features")):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        # record end positions of two parts which need interaction such as Passage and Question, for later separating them
        pq_end_pos=[]
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):

            text_a=context
            text_b = example.question + " " + ending

            special_tok_len=3 # [CLS] [SEP] [SEP]
            sep_tok_len=1 # [SEP]
            t_q_len=len(tokenizer.tokenize(example.question))
            # t_o_len=len(tokenizer.tokenize(ending)) # 直接计算会出错，单独对option分词和拼接question再分词，结果不同
            t_o_len=len(tokenizer.tokenize(text_b))-t_q_len
            context_max_len=max_length-special_tok_len-t_q_len-t_o_len
            t_c_len=len(tokenizer.tokenize(context))
            if t_c_len>context_max_len:
                t_c_len=context_max_len

            assert(t_q_len+t_o_len+t_c_len<=max_length)

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation_strategy=truncation_strategy
            )


            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            assert(len(input_ids[t_c_len+t_q_len+t_o_len:])==special_tok_len)

            t_pq_end_pos=[1 + t_c_len - 1, 1 + t_c_len + sep_tok_len + t_q_len + t_o_len - 1] # [CLS] CONTEXT [SEP] QUESTION OPTION [SEP]

            pq_end_pos.append(t_pq_end_pos)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            pad_token=tokenizer.pad_token_id

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))


        label = label_map[example.label]

        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
                pq_end_pos=pq_end_pos
            )
        )


    return features




processors = {
    "race": RaceProcessor,
    "swag": SwagProcessor,
    "arc": ArcProcessor,
    "dream": DreamProcessor,
    "mctest": MctestProcessor
}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "race", 4,
    "swag", 4,
    "arc", 4,
    "dream", 3,
    "mctest", 4
}
