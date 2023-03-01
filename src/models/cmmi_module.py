from ntpath import join
from typing import Any, List
import re
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.text.rouge import ROUGEScore
from rouge import Rouge
from transformers import BertTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

#model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

class textSumLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.tokenizer = BertTokenizer.from_pretrained('/data/lzp/code/dialogSum/bert-base-chinese/vocab.txt')
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.max_seq_len = 512

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        self.train_rog = Rouge()
        self.val_rog = Rouge()
        self.test_rog = Rouge()
        
        # for logging best so far train roguracy
        self.train_rog_1_best = MaxMetric()
        self.train_rog_2_best = MaxMetric()
        self.train_rog_L_best = MaxMetric()

        # for logging best so far validation roguracy
        self.val_rog_1_best = MaxMetric()
        self.val_rog_2_best = MaxMetric()
        self.val_rog_L_best = MaxMetric()

        # for logging best so far test roguracy
        self.test_rog_1_best = MaxMetric()
        self.test_rog_2_best = MaxMetric()
        self.test_rog_L_best = MaxMetric()

    def forward(self, x: torch.Tensor, y: torch.Tensor, task_type: str):
        return self.net(x,y,task_type)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_rog_1_best.reset()
        self.val_rog_2_best.reset()
        self.val_rog_L_best.reset()

    def model_step(self, batch: Any):
        # x_input_ids, x_attention_mask, y_input_ids, y_attention_mask = batch
        content, summary = batch
        if content[0][:2] == '类型':
            task_type = 'adv'
        elif content[0][:2] == 'p1':
            task_type = 'dialog'
        else:
            task_type = 'sum'
        inputdata = self.tokenizer(content, truncation=True,max_length=self.max_seq_len,pad_to_max_length=True,  return_attention_mask=True, return_tensors="pt")
        x_input_ids = torch.LongTensor(inputdata['input_ids']).cuda()
        x_attention_mask = torch.LongTensor(inputdata['attention_mask']).cuda()
        
        inputdata = self.tokenizer(summary, truncation=True,max_length=self.max_seq_len,pad_to_max_length=True,  return_attention_mask=True, return_tensors="pt")
        y_input_ids = torch.LongTensor(inputdata['input_ids']).cuda()
        y_attention_mask = torch.LongTensor(inputdata['attention_mask']).cuda()
        
        logits, enc_self_attns, dec_self_attns, dec_enc_attns = self.forward(x_input_ids, y_input_ids, task_type)
        loss = self.Ccriterion(logits.reshape(x_input_ids.size()[0],x_input_ids.size()[1],-1), y_input_ids, y_attention_mask)
        preds = torch.reshape(logits.data.max(1, keepdim=True)[1], y_input_ids.shape)
        return loss, preds, y_input_ids

    def Ccriterion(self,logits,target_ids,attention_mask):
        logits = torch.softmax(logits, dim=2)
        text_ntokens = torch.sum(attention_mask, dim=1)-1
        for i in range(len(text_ntokens)):
            gold_probs = torch.gather(logits[i][:text_ntokens[i], :], 1, target_ids[i][1:text_ntokens[i]+1].unsqueeze(1)).squeeze()
            text_batch_loss = -torch.log(gold_probs + 1e-7)

            if i == 0:
                text_batch_losses = text_batch_loss
            else:
                text_batch_losses = torch.cat((text_batch_losses, text_batch_loss), dim=0)
            
                # pdb.set_trace()
        loss = torch.mean(text_batch_losses)
        return loss
    
    def greedy_decoder(self, max_len, enc_input, task_type, start_symbol=101):
        """
        For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
        target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
        Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
        :param model: Transformer Model
        :param enc_input: The encoder input
        :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
        :return: The target input
        """
        enc_outputs, enc_self_attns = self.net.encoder(enc_input,task_type)
        dec_input = torch.zeros(1, 0).type_as(enc_input.data)
        terminal = False
        next_symbol = start_symbol
        while not terminal:
            if dec_input.shape[1] >= max_len:
                return dec_input         
            dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],-1)
            dec_outputs, _, _ = self.net.decoder(dec_input, enc_input, enc_outputs, task_type)
            projected = self.net.projection(dec_outputs)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word
            if int(next_symbol) == 102:
                terminal = True
            #print(next_word)            
        return dec_input
    
    """def greedy_decoder(self, max_len, enc_input, start_symbol=101):
        # enc_outputs, enc_self_attns = model.encoder(enc_input)
        dec_input = torch.zeros(1, 0).type_as(enc_input.data)
        terminal = False
        next_symbol = start_symbol
        while not terminal:
            if dec_input.shape[1] >= max_len:
                return dec_input
            dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],-1)
            dec_outputs, _, _, _  = self.forward(enc_input, dec_input)
            # dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
            # projected = model.projection(dec_outputs)
            prob = dec_outputs.max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word
            if int(next_symbol) == 102:
                terminal = True
            #print(next_word)            
        return dec_input"""

    def training_step(self, batch: Any, batch_idx: int):
        # print('strat train')
        # x_input_ids, x_attention_mask, y_input_ids, y_attention_mask = batch
        content, summary = batch
        loss, preds, targets = self.model_step(batch)

        R_1 = 0
        R_2 = 0
        R_L = 0
        for i in range(len(summary)):
            targets_token = summary[i]
            preds_token = "".join(self.tokenizer.convert_ids_to_tokens(preds[i], skip_special_tokens=True))

            if preds_token == '':
                preds_token = '空'
            # print('---------train-----------')
            # print(targets_token)
            # print(preds_token)
            # log val metrics
            rog = self.train_rog.get_scores(' '.join(list(targets_token)), ' '.join(list(preds_token)[:500]))
            #rog = rog[0]['rouge-1']['f']
            R_1 = R_1 + rog[0]['rouge-1']['f']
            R_2 = R_2 + rog[0]['rouge-2']['f']
            R_L = R_L + rog[0]['rouge-l']['f']
        
        # log val metrics
        rog_1 = R_1 / len(summary)
        rog_2 = R_2 / len(summary)
        rog_L = R_L / len(summary)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(batch[0]))
        self.log("train/rouge_1", rog_1, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch[0]))
        self.log("train/rouge_2", rog_2, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch[0]))
        self.log("train/rouge_L", rog_L, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch[0]))

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds_token, "targets": targets_token, "rouge_1": rog_1, "rouge_2": rog_2, "rouge_L": rog_L}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        R_1 = 0
        R_2 = 0
        R_L = 0
        for i in range(len(outputs)):
            R_1 = R_1 + outputs[i]['rouge_1']
            R_2 = R_2 + outputs[i]['rouge_2']
            R_L = R_L + outputs[i]['rouge_L']
        
        #rog = R / len(outputs)
        rog_1 = R_1 / len(outputs)
        rog_2 = R_2 / len(outputs)
        rog_L = R_L / len(outputs)
        self.train_rog_1_best.update(rog_1)
        self.train_rog_2_best.update(rog_2)
        self.train_rog_L_best.update(rog_L)
        self.log("train/rouge_1_best", self.train_rog_1_best.compute(), on_epoch=True, prog_bar=True)
        self.log("train/rouge_2_best", self.train_rog_2_best.compute(), on_epoch=True, prog_bar=True)
        self.log("train/rouge_L_best", self.train_rog_L_best.compute(), on_epoch=True, prog_bar=True)
        print("train/rouge_1_best", self.train_rog_1_best.compute())
        print("train/rouge_2_best", self.train_rog_2_best.compute())
        print("train/rouge_L_best", self.train_rog_L_best.compute())
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        print('strat valid')
        #loss, preds, targets = self.model_step(batch)
        
        # x_input_ids, x_attention_mask, y_input_ids, y_attention_mask = batch
        content, summary = batch
        
        if content[0][:2] == '类型':
            task_type = 'adv'
        elif content[0][:2] == 'p1':
            task_type = 'dialog'
        else:
            task_type = 'sum'
        
        inputdata = self.tokenizer(content, truncation=True,max_length=self.max_seq_len,pad_to_max_length=True,  return_attention_mask=True, return_tensors="pt")
        x_input_ids = torch.LongTensor(inputdata['input_ids']).cuda()
        x_attention_mask = torch.LongTensor(inputdata['attention_mask']).cuda()
        
        inputdata = self.tokenizer(summary, truncation=True,max_length=self.max_seq_len,pad_to_max_length=True,  return_attention_mask=True, return_tensors="pt")
        y_input_ids = torch.LongTensor(inputdata['input_ids']).cuda()
        y_attention_mask = torch.LongTensor(inputdata['attention_mask']).cuda()
        
        R_1 = 0
        R_2 = 0
        R_L = 0
        predicts = []
        for i in range(1):
            targets_token = summary[i]
            greedy_dec_input = self.greedy_decoder(max_len=250, enc_input=x_input_ids[i].view(1, -1), task_type=task_type, start_symbol=101)
            #print("".join(self.tokenizer.convert_ids_to_tokens(greedy_dec_input[0], skip_special_tokens=False)))
            predict, _, _, _ = self.net(x_input_ids[i].view(1, -1), greedy_dec_input, task_type)
            predicts.extend(predict)
        
            predict = predict.data.max(1, keepdim=True)[1]
            # targets_token = "".join(self.tokenizer.convert_ids_to_tokens(y_input_ids[i], skip_special_tokens=True))
            # targets_CLS_index = targets_token.find('[CLS]')
            # targets_SEP_index = targets_token.find('[SEP]')
            # targets_token = targets_token[targets_CLS_index+5:targets_SEP_index]
            preds_token = "".join(self.tokenizer.convert_ids_to_tokens(predict, skip_special_tokens=True))
            
            if preds_token == '':
                preds_token = '空'
            
            print('---------val-----------')
            print(targets_token)
            print(preds_token)
            rog = self.val_rog.get_scores(' '.join(list(targets_token)), ' '.join(list(preds_token)[:500]))
            #rog = rog[0]['rouge-1']['f']
            R_1 = R_1 + rog[0]['rouge-1']['f']
            R_2 = R_2 + rog[0]['rouge-2']['f']
            R_L = R_L + rog[0]['rouge-l']['f']
            
        # log val metrics
        rog_1 = R_1 / len(y_input_ids)
        rog_2 = R_2 / len(y_input_ids)
        rog_L = R_L / len(y_input_ids)

        #loss = self.Ccriterion(torch.Tensor(predicts).reshape(x_input_ids.size()[0],x_input_ids.size()[1],-1), y_input_ids, y_attention_mask)
        loss = 0.1
        self.log("val/rouge_1", rog_1, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch[0]))
        self.log("val/rouge_2", rog_2, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch[0]))
        self.log("val/rouge_L", rog_L, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch[0]))
        

        return {"loss": loss, "preds": preds_token, "targets": targets_token, "rouge_1": rog_1, "rouge_2": rog_2, "rouge_L": rog_L}

    def validation_epoch_end(self, outputs: List[Any]):
        R_1 = 0
        R_2 = 0
        R_L = 0
        for i in range(len(outputs)):
            R_1 = R_1 + outputs[i]['rouge_1']
            R_2 = R_2 + outputs[i]['rouge_2']
            R_L = R_L + outputs[i]['rouge_L']
        
        #rog = R / len(outputs)
        rog_1 = R_1 / len(outputs)
        rog_2 = R_2 / len(outputs)
        rog_L = R_L / len(outputs)
        self.val_rog_1_best.update(rog_1)
        self.val_rog_2_best.update(rog_2)
        self.val_rog_L_best.update(rog_L)
        self.log("val/rouge_1_best", self.val_rog_1_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/rouge_2_best", self.val_rog_2_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/rouge_L_best", self.val_rog_L_best.compute(), on_epoch=True, prog_bar=True)
        print("val/rouge_1_best", self.val_rog_1_best.compute())
        print("val/rouge_2_best", self.val_rog_2_best.compute())
        print("val/rouge_L_best", self.val_rog_L_best.compute())

    def test_step(self, batch: Any, batch_idx: int):
        #loss, preds, targets = self.model_step(batch)
        # x_input_ids, x_attention_mask = batch
        content = batch
        
        if content[0][:2] == '类型':
            task_type = 'adv'
        elif content[0][:2] == 'p1':
            task_type = 'dialog'
        else:
            task_type = 'sum'
        
        inputdata = self.tokenizer(content, truncation=True,max_length=self.max_seq_len,pad_to_max_length=True,  return_attention_mask=True, return_tensors="pt")
        x_input_ids = torch.LongTensor(inputdata['input_ids']).cuda()
        x_attention_mask = torch.LongTensor(inputdata['attention_mask']).cuda()

        for i in range(len(x_input_ids)):
            greedy_dec_input = self.greedy_decoder(max_len=250, enc_input=x_input_ids[i].view(1, -1), task_type=task_type, start_symbol=101)
            predict, _, _, _ = self.net(x_input_ids[i].view(1, -1), greedy_dec_input, task_type)
            predict = predict.data.max(1, keepdim=True)[1]
            preds_token = "".join(self.tokenizer.convert_ids_to_tokens(predict, skip_special_tokens=True))
            if preds_token == '':
                preds_token = '空'
            print(preds_token)
            with open('./test/cmmi_5epoch.txt', 'a+') as f:
                f.write(preds_token+'\n')
            

        return {"preds": preds_token}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
