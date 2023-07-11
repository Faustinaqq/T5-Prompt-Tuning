from util.utils import get_logger
import torch
from util.dataset import CustomDataset
from torch.utils.data import DataLoader
import os
from torch.optim import AdamW
from tqdm import tqdm
from transformers import default_data_collator
from transformers import get_linear_schedule_with_warmup
import numpy as np


def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')


class T5BoolQTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.learning_rate = args.lr
        self.num_epochs = args.epochs
        self.target_max_length = args.target_max_length
        self.eval_epoch = args.eval_epoch
        self.logger = get_logger(os.path.join('./log', args.log_file))
        train_dataset = CustomDataset(self.train_dataset, self.tokenizer, self.max_length, self.target_max_length)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )
        eval_dataset = CustomDataset(self.eval_dataset, self.tokenizer, self.max_length)
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=default_data_collator
        )
        print_args(self.logger, args)
        
        with torch.no_grad():
            total_params = sum(p.numel() for p in self.model.get_parameters())
            self.logger.info("param: {}".format(total_params))

    def train(self):
        result = self.evaluate()
        self.logger.info(f"init result: {result}")

        t_total = len(self.train_dataloader) * self.num_epochs
        optimizer = AdamW(self.model.get_parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

        for epoch in range(1, self.num_epochs + 1):
            total_loss = 0
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

                loss = self.model(input_ids=input_ids, target_ids=target_ids).loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            self.logger.info(f"Epoch {epoch} - Avg Loss: {total_loss / len(self.train_dataloader)}")
            if epoch % self.eval_epoch == 0:
                result = self.evaluate()
                self.logger.info(f"result: {result}")

    def evaluate(self):
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                generate_ids = self.model.generate(input_ids=input_ids, max_target_length=self.target_max_length)
                predictions = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
                all_predictions.extend(predictions)
                gold_labels = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
                all_labels.extend(gold_labels)

        accuracy = np.mean([prediction == label for prediction, label in zip(all_predictions, all_labels)])
        return {'accuracy': accuracy}