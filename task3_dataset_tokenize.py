import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from typing import Optional, Tuple
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision.transforms import transforms
from transformers import BertTokenizer
import pickle
import json
from tqdm import tqdm
class textDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 512,
        max_caption_len: int = 200,
        do_lower_case: bool = True,
        use_word_cut = True,  # 词级别分词
        text_tokenizer: str = '/data/lzp/code/dialogSum/bert-base-chinese/'
    ):
        super().__init__()

        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.tokenizer = BertTokenizer.from_pretrained(text_tokenizer)
        self.datas = []
        
        with open(self.data_path) as f:
            for line in f:
                self.datas.append(json.loads(line))
        
        tokenized = []
        temp = {}
        
        for i in tqdm(range(len(self.datas))):
            temp = {}
            text = 'p1#' + "".join(self.datas[i]['p1_persona']) +'*p2#' + "".join(self.datas[i]['p2_persona']) + '*'
            temp['content'] = text
            temp['summary'] = self.datas[i]['conversation'][0]
            tokenized.append(temp)
            for j in range(len(self.datas[i]['conversation'])-1):
                text = text + self.datas[i]['conversation'][j]
                temp = {}
                temp['content'] = text
                temp['summary'] = self.datas[i]['conversation'][j+1]
                tokenized.append(temp)
                
        with open(f'{self.data_path[:-5]}_aug.json','w') as writers:
            for item in tokenized:
                json.dump(item,writers,ensure_ascii=False)
                writers.write('\n')
        # for i in tqdm(range(len(self.datas))):
        #     text = 'p1#' + "".join(self.datas[i]['p1_persona']) +'*p2#' + "".join(self.datas[i]['p2_persona']) + '*'
        #     temp['input_text'] = self._preprocess_title(text)
        #     temp['input_summary'] = self._preprocess_title(self.datas[i]['conversation'][0])
        #     tokenized.append(temp)
        #     for j in range(len(self.datas[i]['conversation'])-1):
        #         temp['input_text'] = self._preprocess_title(text + self.datas[i]['conversation'][j])
        #         temp['input_summary'] = self._preprocess_title(self.datas[i]['conversation'][j+1])
        #         tokenized.append(temp)
        
        # f = open(f'{self.data_path[:-5]}.pkl','wb')
        # pickle.dump(tokenized, f, -1)
        #print(1)

    def _preprocess_title(self, textlist):
        """
        title
        """
        input = ''
        for text in textlist:
            input = input + text
        inputdata = self.tokenizer.encode_plus(text=input, truncation=True,max_length=self.max_seq_len,pad_to_max_length=True,  return_attention_mask=True)
        inputdata_input_ids = torch.LongTensor(inputdata['input_ids'])
        inputdata_attention_mask = torch.LongTensor(inputdata['attention_mask'])
        return {'input_ids':inputdata_input_ids, 'attention_mask':inputdata_attention_mask}

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index:int):
        summary_input_ids = torch.LongTensor(self.datas[index]['input_summary']['input_ids'])
        summary_attention_mask = torch.LongTensor(self.datas[index]['input_summary']['attention_mask'])
        inputdata_input_ids = torch.LongTensor(self.datas[index]['input_text']['input_ids'])
        inputdata_attention_mask = torch.LongTensor(self.datas[index]['input_text']['attention_mask'])
        return (inputdata_input_ids, inputdata_attention_mask, summary_input_ids, summary_attention_mask)


class textSumDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir:str,
        train_data_dir: str, 
        test_data_dir: str, 
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_seq_len: int = 512,
        max_caption_len: int = 200,
        text_tokenizer: str = '/data/lzp/code/dialogSum/bert-base-chinese/vocab.txt',
        train_val_test_split: Tuple[int, int, int] = (4844, 600, 600),
        train_size = 4844,
        dev_size = 600,
        test_size = 600
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        self.data_dir = data_dir
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.train_size = train_size
        self.dev_size = dev_size
        self.test_size = test_size
        self.tokenizer = BertTokenizer.from_pretrained(text_tokenizer)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train:
            trainset = textDataset(self.hparams.train_data_dir, self.hparams.max_seq_len,
                                        self.hparams.max_caption_len, self.hparams.text_tokenizer)
            testset = textDataset(self.hparams.test_data_dir, self.hparams.max_seq_len,
                                        self.hparams.max_caption_len, self.hparams.text_tokenizer)

            self.data_train = trainset
            self.data_test = testset
        print(1)
            

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
if __name__ == '__main__':
    data_loader = textSumDataModule(
        data_dir='./data/DuLeMon_faithfulmatch_v2/DuLeMon_faithfulmatch_v2',
        train_data_dir='./data/DuLeMon_faithfulmatch_v2/DuLeMon_faithfulmatch_v2/train.json',
        test_data_dir='./data/DuLeMon_faithfulmatch_v2/DuLeMon_faithfulmatch_v2/dev.json'
    ).setup()
    print(data_loader)