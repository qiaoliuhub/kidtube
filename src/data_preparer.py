import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from torch.utils.data import random_split, DataLoader, Dataset
import glob
import pandas as pd
import os
import pickle
import argparse
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn as nn
import pdb
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score


class VideoCategoryData:

    video_details_df = None
    video_text_and_label_df = None

    def __init__(self, video_details_dir, video_transcript_dir):
        '''
        Prepare the dataframe ## video ## text ## category ##
        '''

        self.video_details_dir = video_details_dir
        self.video_transcript_dir = video_transcript_dir
        self.__prepare_video_details_df()
        self.__build_video_cat_df()
        self.__label_category()

    def __prepare_video_details_df(self):
        '''
        read in all the video detail csv file in video details csv folder
        '''
        if self.video_details_df is None:
            all_df = []
            for csv_file in glob.glob(self.video_details_dir + '/*.csv'):
                cur_df = pd.read_csv(csv_file)
                all_df.append(cur_df)
            self.video_details_df = pd.concat(all_df, ignore_index = True)

    def __build_video_cat_df(self):
        
        if self.video_text_and_label_df is None:
            self.video_text_and_label_df = pd.DataFrame(columns = ['video_id', 'video_transcript', 'video_category'])
            for i, video_file in enumerate(glob.glob(self.video_transcript_dir + '/*.p')):
                video_id = os.path.basename(video_file).split('.')[0]
                video_transcript = self.__merge_all_text_for(video_file)
                video_category = self.video_details_df.loc[self.video_details_df['video'] == video_id, 'category'].values
                if len(video_category) < 1:
                    print(video_id)
                    continue
                self.video_text_and_label_df.loc[i] = pd.Series({'video_id': video_id, 'video_transcript': video_transcript, 'video_category': video_category[0]})
    
    def __label_category(self):
        ### 2. make the str type categorical information numerical like a label
        label_encoder = LabelEncoder()
        self.video_text_and_label_df['label'] = label_encoder.fit_transform(self.video_text_and_label_df['video_category'])
        np.save('./label_encoder_cls.npy', label_encoder.classes_)
        
    def __merge_all_text_for(self, video_file):

        video_transcript_segments = pickle.load(open(video_file, 'rb'))
        ### tips to imporve: may need to add on punctunation
        ### some text was saved in unicode form which can not be used, should encode to utf-8 and save
        return ' '.join([x['text'].encode('unicode-escape').replace(b'\\\\', b'\\').decode('unicode-escape') for x in video_transcript_segments])
    
    def get_video_cat_df(self):
        return self.video_text_and_label_df
    
    def get_num_of_classes(self):
        print(set(self.video_text_and_label_df.label))
        return len(set(self.video_text_and_label_df.video_category))

class TranscriptCategoryDataset(Dataset):
    '''
    Transfer a dataframe to dataset
    '''

    def __init__(self, tokenized_results):
        '''
        :tokenized_results: dict or dict-list type, like transformers.tokenization_utils_base.BatchEncoding
        '''
        super(TranscriptCategoryDataset, self).__init__()
        # assert 'video_transcript' in video_cat_df.columns and 'label' in video_cat_df.columns, "video_transcript and video_category columns should be in video category dataframe"
        # assert 'input_ids' in video_cat_df.columns and 'attention_mask' in video_cat_df.columns, "input_ids and attention_mask columns should be in video category dataframe"

        assert 'input_ids' in tokenized_results and 'attention_mask' in tokenized_results and 'label' in tokenized_results, "input_ids, label and attention_mask columns should be in video category dataframe"
        self.video_category_data = tokenized_results

    def __len__(self):
        return len(self.video_category_data['label'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.video_category_data['input_ids'][idx], self.video_category_data['label'][idx], self.video_category_data['attention_mask'][idx]

class KidTubeModel(pl.LightningModule):
    '''
    the classification model built on top of Albert for kids video category classification
    '''
    def __init__(self, number_of_classes):
        super(KidTubeModel, self).__init__()
        self.save_hyperparameters()
        self.bert_model = AutoModelForMaskedLM.from_pretrained("albert-base-v2")
        self.fc1 = nn.Linear(768, 50)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.f2 = nn.Linear(50, number_of_classes)
        self.down_stream = nn.Sequential(self.fc1, self.relu, self.dropout, self.f2)
        self.softmax = nn.Softmax(dim = -1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids, attention_mask):
        ''' The inference step
        :token_ids: tensor of token_id (barch * length) 
        :return: tensor (batch * num_of_cls)
        '''
        hidden_states = self.bert_model(input_ids = token_ids, attention_mask = attention_mask, output_hidden_states=True)
        hidden_cls = hidden_states['hidden_states'][-1][:,0,:]
        return self.softmax(self.down_stream(hidden_cls))

    def training_step(self, batch, batch_idx):
        '''
        Training step and log the losses every step
        :batch: dataloader(tokenids, label, mask)
        :batch_idx: 
        :return: dict with loss
        '''
        x, y, mask = batch
        hidden_states = self.bert_model(input_ids = x, attention_mask = mask, output_hidden_states=True)
        hidden_cls = hidden_states['hidden_states'][-1][:,0,:]
        output = self.down_stream(hidden_cls)
        pred_probs = self.softmax(output)
        loss = self.loss(output, y)
        self.log('train_loss_step', loss, prog_bar=True)
        return {'loss': loss, 'train_labels': y, 'train_pred_probs': pred_probs}

    # def training_step_end(...)

    def training_epoch_end(self, outputs):
        '''
        :outputs: dict: the output from validation step:
        :return: dict: 'log': what ever want to be logged in dict format
        '''
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        pred_probs = torch.cat([x['train_pred_probs'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['train_labels'] for x in outputs]).detach().cpu().numpy()
        roc_auc = roc_auc_score(labels, pred_probs, multi_class='ovr')
        pr_auc = roc_auc_score(labels, pred_probs, multi_class='ovr')
        to_log = {'train_loss_epoch': train_loss, 'train_rocauc': roc_auc, 'train_prauc': pr_auc}
        self.log_dict(to_log, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        hidden_states = self.bert_model(input_ids = x, attention_mask = mask, output_hidden_states=True)
        hidden_cls = hidden_states['hidden_states'][-1][:,0,:]
        output = self.down_stream(hidden_cls)
        pred_probs = self.softmax(output)
        val_loss = self.loss(output, y)
        self.log('val_loss_step', val_loss, prog_bar=True)
        ## the first val_loss will be used in the validation_epoch_step
        return {'val_loss_step': val_loss, 'val_labels': y, 'val_pred_probs': pred_probs}

    def validation_epoch_end(self, outputs):
        '''
        :outputs: dict: the output from validation step:
        :return: dict: 'log': what ever want to be logged in dict format
                    'val_loss': trigger automatic checkpoint
        '''
        val_loss = torch.stack([x['val_loss_step'] for x in outputs]).mean()
        pred_probs = torch.cat([x['val_pred_probs'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['val_labels'] for x in outputs]).detach().cpu().numpy()
        roc_auc = roc_auc_score(labels, pred_probs, multi_class='ovr')
        pr_auc = roc_auc_score(labels, pred_probs, multi_class='ovr')
        to_log = {'val_loss_epoch': val_loss, 'val_rocauc': roc_auc, 'val_prauc': pr_auc, 'val_loss': val_loss}
        self.log_dict(to_log, prog_bar=True)
        ### val_loss is the default metrics used by checkpoint callback function, when this values is smaller, the model will be automatically saved
        #return {'val_loss': val_loss}

    # def validation_step_end(...)

    # def test_step(...)

    # def test_step_end(...)

    # def test_epoch_end(...)
    # def setup(self, stage):
    #     if stage == 'fit':
    #         # Get dataloader by calling it - train_dataloader() is called after setup() by default
    #         train_loader = self.train_dataloader()

    #         # Calculate total steps
    #         self.total_steps = (
    #             (len(train_loader.dataset) // (16 * max(1, 1))) *
    #             float(5)
    #         )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        models = [self.bert_model, self.down_stream]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for model in models for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0001,
            },
            {
                "params": [p for model in models for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        return optimizer

    # def any_extra_hook(...)


class KidTubeDataModule(pl.LightningDataModule):

    def __init__(self, video_category_df, batch_size = 32):
        super(KidTubeDataModule, self).__init__()
        self.batch_size = batch_size
        self.video_category_df  = video_category_df
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    def prepare_data(self):
        '''
        Use this method to do things that might write to disk or that need to be \
            done only from a single GPU in distributed settings.
        how to download(), tokenize, etc
        prepare_data is called from a single GPU. Do not use it to assign state (self.x = y).
        '''
        AutoTokenizer.from_pretrained("albert-base-v2")

    def build_transcript_dataset(self):
        
        ### 1. tokenize the vedio transcript to token ids

        ### may need to change text list to real list
        assert 'video_transcript' in self.video_category_df.columns, "video_transcript column should be in video category dataframe"
        tokenized_results = self.tokenizer(list(self.video_category_df.video_transcript), 
                                    return_tensors = 'pt',
                                    padding='max_length',
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    truncation = True)
        tokenized_results['label'] = torch.tensor(self.video_category_df.label.values)
        # self.video_category_df['input_ids'] = tokenized_results['input_ids']
        # self.video_category_df['attention_mask'] = tokenized_results['attention_mask']
        return tokenized_results

    def setup(self, stage = None):
        '''
        how to split, etc…
        '''
        tokenized_results = self.build_transcript_dataset()
        ### 2. build the dataset class
        self.transcript_category_dataset = TranscriptCategoryDataset(tokenized_results)
        data_size = len(self.transcript_category_dataset)
        train_data_size, val_data_size = int(data_size * 0.8), data_size - int(data_size * 0.8)
        self.transcript_cat_train, self.transcript_cat_val = random_split(self.transcript_category_dataset, [train_data_size, val_data_size], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.transcript_cat_train, batch_size=self.batch_size, num_workers = 5)

    def val_dataloader(self):
        return DataLoader(self.transcript_cat_val, batch_size=self.batch_size, num_workers = 5)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

if __name__ == '__main__':

    USE_wandb = False
    if USE_wandb:
        wandb_logger = WandbLogger(name='kidtube' + datetime.now().strftime('_%m-%d-%H-%M-%S'), project='pytorchlightning')
        # wandb.init(project="DeepCE_AE_loss")
    else:
        wandb_logger = WandbLogger(offline = True)
        # import logging
        # logging.basicConfig(filename = 'kidtube_log', 
        #                                     level = logging.DEBUG,
        #                                     format='%(asctime)-15s %(name)s %(levelname)s %(message)s')
        # wandb_logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--video_details_dir', help = 'The folder to save all the video details csv file')
    parser.add_argument('--video_transcript_dir', help = 'The folder to save all the video transcripts file')

    args = parser.parse_args()
    video_details_dir = args.video_details_dir
    video_transcript_dir = args.video_transcript_dir

    ### prepare the video detail dataframe
    video_category_data_preparer = VideoCategoryData(video_details_dir=video_details_dir, video_transcript_dir=video_transcript_dir)
    video_category_df = video_category_data_preparer.get_video_cat_df()
    num_of_classes = video_category_data_preparer.get_num_of_classes()

    kidtube_video_data_module = KidTubeDataModule(video_category_df=video_category_df, batch_size=16)
    kidtube_video_data_module.prepare_data()
    kidtube_video_data_module.setup()

    kidtube_model = KidTubeModel(number_of_classes = num_of_classes)
    trainer = pl.Trainer.from_argparse_args(args, fast_dev_run = False, logger= wandb_logger)
    trainer.fit(kidtube_model, kidtube_video_data_module)