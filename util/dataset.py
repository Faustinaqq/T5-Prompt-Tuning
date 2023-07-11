from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, target_max_length=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length
        self.ll = 0
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        input_ids, target_ids = self.preprocess(example)
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
        }

    def preprocess(self, example):
        question = example['question']
        passage = example['passage']
        # boolq input for t5
        input_text = f"question: {question}\npassage: {passage}"
        
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        answer = str(example['answer'])
        
        target_ids = self.tokenizer.encode_plus(
            answer,
            max_length=self.target_max_length,
            pad_to_max_length=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        target_ids = target_ids['input_ids'].squeeze()
        return input_ids, target_ids