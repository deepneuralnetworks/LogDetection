from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
import torch
import os
import pickle
import pandas as pd
import numpy as np


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        self.is_pretrain = args.is_pretrain

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        # Use Pandas .read_csv
        data = pd.read_csv(file_path, encoding="utf-8")
        text = data['message'].apply(lambda x: x[:-1]).tolist() # 마침표 제거
        self.examples = tokenizer.batch_encode_plus(text, add_special_tokens=True, max_length=block_size)["input_ids"]

        # Dataset for fine-tuning 
        if not self.is_pretrain:
            type_code = {
                'DELL LC ':0, 'xid':1, 'sxid':2,
                'Infiniband switch':3, 'HPE IML':4, 'LSF':5, 'syslog':6
                }
            type_label = {
                's0': 0, 's1': 1, 's2': 2, 's3': 3
                }
            self.logtype = data['logtype'].apply(lambda x: type_code[x])
            self.label = data['severity'].apply(lambda x: type_label[x])
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # Dataset for fine-tuning 
        if not self.is_pretrain:
            elem = {
                'example': torch.tensor(self.examples[item], dtype=torch.long),
                'logtype': torch.tensor([self.logtype[item]], dtype=torch.long),
                'label': torch.tensor(self.label[item], dtype=torch.long)
                }
            return elem

        # Dataset for pre-training
        else:
            return torch.tensor(self.examples[item], dtype=torch.long)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)