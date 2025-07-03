from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from dep_tree_gen import subword_dep_matrix
import torch

# Glue Datasets: MRPC, QQP, STS-B, MNLI, RTE

class DEPI_dataset():
    def __init__(self, sequence_length = 512):
        
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
        self.dep_matrix_fn = subword_dep_matrix 
        self.sequence_length = sequence_length
        self.datasets = self.create_dataset()
    
    def create_dataset(self):
        dataset_names = ["mrpc", "qqp", "stsb", "mnli", "rte"]

        field_map = {
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
        "rte": ("sentence1", "sentence2"),
        }

        dataset_dict = {}


        for dataset_name in dataset_names:
            data = load_dataset("glue", dataset_name)
            field1, field2 = field_map[dataset_name]
            tmp_dict = {}

            for type in data:
                input_ids, token_type_ids, attention_mask, labels, dep_matrices = [], [], [], [], []
                for sample in data[type]:
                    s1 = sample[field1]
                    s2 = sample[field2]
                    label = sample['label']

                    encoded = self.tokenizer(s1, s2, 
                                            padding="max_length", truncation=True, max_length=self.sequence_length, return_tensors="pt")
                    
                    dep_matrix = self.dep_matrix_fn(s1, s2, alpha=2, seq_len=self.sequence_length)

                    input_ids.append(encoded["input_ids"].squeeze(0))
                    token_type_ids.append(encoded["token_type_ids"].squeeze(0))
                    attention_mask.append(encoded["attention_mask"].squeeze(0))
                    labels.append(torch.tensor(label, dtype=torch.long))
                    dep_matrices.append(torch.tensor(dep_matrix, dtype=torch.float))  # [seq_len, seq_len]

                tmp_dict[type] = {
                    "input_ids": torch.stack(input_ids),
                    "token_type_ids": torch.stack(token_type_ids),
                    "attention_mask": torch.stack(attention_mask),
                    "labels": torch.stack(labels),
                    "dependency_matrix": torch.stack(dep_matrices), 
                }

            dataset_dict[dataset_name] = DatasetDict(tmp_dict)
        
        return dataset_dict
        
    
    def get_dataset(self, name):
        return self.datasets[name]
