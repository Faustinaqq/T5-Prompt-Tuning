from transformers import T5Tokenizer
from datasets import load_dataset
from util.utils import setup_seed
from argparse import ArgumentParser
from model.model import T5FineTuningModel, T5PromptModel
from util.trainer import T5BoolQTrainer


def parse_args():
    parser = ArgumentParser(description='prompt tuning of T5 on BoolQ')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate')
    parser.add_argument('--epochs', type=int, default=300, help='training epoch')
    parser.add_argument('--seed', type=int, default=111, help='seed')
    parser.add_argument('--log_file', type=str, default='log.txt', help='log file')
    parser.add_argument('--model', type=str, default='t5-base', choices=['t5-small', 't5-base', 't5-large', 't5-3B'])
    parser.add_argument('--eval_epoch', type=int, default=1, help='evaluate every eval_epoch')
    parser.add_argument('--max_length', type=int, default=512, help='tokenizer input max length')
    parser.add_argument('--target_max_length', type=int, default=4, help='t5 generate max length')
    parser.add_argument('--prompt_len', type=int, default=20, help='prompt length')
    parser.add_argument('--ft_way', type=str, default=20, help='fine tuning way', choices=['fine_tune', 'prompt'])
    parser.add_argument('--changed', type=int, default=0, help='prompt tuning changed or not', choices=[0, 1])
    parser.add_argument('--init', type=str, default='class_label', help='prompt parameter initialize', choices=['random', 'random_vocab', 'class_label'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    tokenizer = T5Tokenizer.from_pretrained(args.model, model_max_length=args.max_length)
    
    train_dataset = load_dataset('boolq', split='train')
    eval_dataset = load_dataset('boolq', split='validation')
    
    setup_seed(args.seed)

    if args.ft_way == 'fine_tune':
        model = T5FineTuningModel(args.model)
    elif args.ft_way == 'prompt':
        model = T5PromptModel(args.model, tokenizer, args.prompt_len, init=args.init, changed=args.changed)
    
    trainer = T5BoolQTrainer(model, tokenizer, train_dataset, eval_dataset, args)
    trainer.train()
