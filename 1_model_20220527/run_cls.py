from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import glob
from typing import Dict

from task.trainer import TrainCLS
from data.textdataset import load_and_cache_examples
from utils.misc import set_seed
from utils.args import roberta_cls_parser

def main(args):

    # Batch Collector
    def collate(data: Dict):

        examples = []
        logtypes = []
        labels = []
        
        for elem in data:
            examples.append(elem['example'])
            logtypes.append(elem['logtype'])
            labels.append(elem['label'])

        if tokenizer._pad_token is None:
            return {
                'example': pad_sequence(examples, batch_first=True),
                'logtype': torch.stack(logtypes, dim=0),
                'label': torch.stack(labels, dim=0)
                }

        return {
            'example': pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id),
            'logtype': torch.stack(logtypes, dim=0),
            'label': torch.stack(labels, dim=0)
            }

    # Fix seed for reproducibility
    set_seed(args)

    # CUDA
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # RoBERTa default block_size is 512
    args.block_size = 512

    # Get pretrained encoder & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,  num_labels=4).to(device)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Get dataset
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    # Define batch size
    args.train_batch_size = args.per_gpu_train_batch_size 
    args.eval_batch_size = args.per_gpu_eval_batch_size

    # Get dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # Define trainer (including evaluation)
    trainer = TrainCLS(args, model, optimizer, criterion, train_dataloader, eval_dataloader, tokenizer)

    # RUN!
    num_epochs = args.num_train_epochs
    for epoch in range(num_epochs):
        trainer.run(epoch)



if __name__ == "__main__":

    for seed in [1011,2022,3033,4044,5055]:
        # Define experiment
        exp_number = len(glob.glob('./result_*'))
        
        # Define parser
        parser = roberta_cls_parser(exp_number, seed)
        args = parser.parse_args()
        args.model_name_or_path = 'allenai/cs_roberta_base'    

        main(args)

