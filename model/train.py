""" Training Config & Helper Classes  """

import os
import json
from typing import NamedTuple
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score
from random import randint
from pytorchtools import EarlyStopping


class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 128
    lr: int = 1e-3 # learning rate
    n_epochs: int = 100 # the number of epoch
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, dataName,stopNum,model,model2, data_iter,data_iter2, data_iter3,dataset_dev, optimizer, optimizer2, device, kkk):
        self.cfg = cfg # config for training : see class Config
        self.dataName = dataName
        self.stopNum = stopNum
        self.model = model
        self.model2 = model2
        self.data_iter = data_iter # iterator to load data
        self.data_iter2 = data_iter2
        self.data_iter3 = data_iter3
        self.data_iter_temp = data_iter3
        self.dataset_dev = dataset_dev
        self.optimizer = optimizer
        self.optimizer2 =optimizer2
        self.device = device # device name
        self.kkk = kkk

    
    def train(self, get_loss_CNN, get_loss_Attn_LSTM, evalute_CNN_SSL, pseudo_labeling,evalute_Attn_LSTM,evalute_CNN,evalute_Attn_LSTM_SSL, generating_lexiocn, data_parallel=True):
     
        """ Train Loop """
        self.model.train() # train mode
        self.model2.train() # train mode
        model = self.model.to(self.device)
        model2 = self.model2.to(self.device)
        t =  self.kkk
        
        if(self.dataName == 'IMDB'):
            rnn_save_name = "./IMDB_model_save/checkpoint_RNN"+str(t)+".pt"
            cnn_save_name = "./IMDB_model_save/checkpoint_CNN"+str(t)+".pt"
            result_name = "./result/result_IMDB.txt"
            pseudo_name = "./result/pseudo_train_set_IMDB.txt"
        elif(self.dataName == "AGNews"):
            rnn_save_name = "./AGNews_model_save/checkpoint_RNN"+str(t)+".pt"
            cnn_save_name = "./AGNews_model_save/checkpoint_CNN"+str(t)+".pt"
            result_name = "./result/result_AGNews.txt"
            pseudo_name = "./result/pseudo_train_set_AGNews.txt"
        elif(self.dataName == "dbpedia"):
            rnn_save_name = "./DBpedia_model_save/checkpoint_RNN"+str(t)+".pt"
            cnn_save_name = "./DBpedia_model_save/checkpoint_CNN"+str(t)+".pt"
            result_name = "./result/result_DBpedia.txt"
            pseudo_name = "./result/pseudo_train_set_DBpedia.txt"
        elif(self.dataName == "yahoo"):
            rnn_save_name = "./yahoo_model_save/checkpoint_RNN"+str(t)+".pt"
            cnn_save_name = "./yahoo_model_save/checkpoint_CNN"+str(t)+".pt"
            result_name = "./result/result_yahoo.txt"
            pseudo_name = "./result/pseudo_train_set_yahoo.txt"

        
        
        num_a=0
        global_step = 0 # global iteration steps regardless of epochs
        global_step3 = 0

        before = -50
        curTemp=0
        print("self.cfg.n_epochs#:", self.cfg.n_epochs)
        ddf = open(result_name,'a', encoding='UTF8')
        ddf.write("############################################"+str(t)+": ramdom_samplimg###########################################"+'\n')
        ddf.close()
        
        ddf = open(pseudo_name,'a', encoding='UTF8')
        ddf.write("############################################"+str(t)+": ramdom_samplimg###########################################"+'\n')
        ddf.close()
                
        for e in range(self.cfg.n_epochs):
            if(e==0):
                temp=987654321
                early_stopping = EarlyStopping(patience=30, verbose=True)
                valid_losses = []
                
                while(1):
                    model.train()
                    loss_sum = 0.
                    global_step3 = 0
                    iter_bar3 = tqdm(self.data_iter3, desc='Iter (loss=X.XXX)')
                    for i, batch in enumerate(iter_bar3):
                        batch = [t.to(self.device) for t in batch]
                        loss = get_loss_CNN(model, batch, global_step3).mean() # mean() for Data Parallelism
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        global_step3 += 1
                        loss_sum += loss.item()
                        iter_bar3.set_description('Iter (loss=%5.3f)'%loss.item())
                        
                        if global_step3 % self.cfg.save_steps == 0: # save
                            self.save(global_step3)

                        if self.cfg.total_steps and self.cfg.total_steps < global_step3:
                            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                            print('The Total Steps have been reached.')
                            self.save(global_step3) # save and finish when global_steps reach total_steps
                            return

                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    model.eval()
                    loss_sum = 0.
                    global_step3 = 0
                    iter_bar_dev = tqdm(self.dataset_dev, desc='Iter (loss=X.XXX)')
                    for i, batch in enumerate(iter_bar_dev):
                        batch = [t.to(self.device) for t in batch]
                        loss = get_loss_CNN(model, batch, global_step3).mean() # mean() for Data Parallelism
                        valid_losses.append(loss.item())
                        global_step3 += 1
                        loss_sum += loss.item()
                        iter_bar_dev.set_description('Iter (loss=%5.3f)'%loss.item())

                        if global_step3 % self.cfg.save_steps == 0: # save
                            self.save(global_step3)

                        if self.cfg.total_steps and self.cfg.total_steps < global_step3:
                            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                            print('The Total Steps have been reached.')
                            self.save(global_step3) # save and finish when global_steps reach total_steps
                            return

                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    valid_loss = np.average(valid_losses)
                    loss_min=early_stopping(valid_loss, model,"./model_save/checkpoint_CNN_real.pt")
                    valid_losses = []

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

 
                model.load_state_dict(torch.load("./model_save/checkpoint_CNN_real.pt"))
                print("Early stopping")
                model.eval()# evaluation mode
                loss_total = 0
                total_sample = 0
                acc_total = 0
                correct = 0
                global_step3=0
                
                with torch.no_grad():
                    iter_bar = tqdm(self.data_iter2, desc='Iter (f1-score=X.XXX)')
                    for batch in iter_bar:
                        batch = [t.to(self.device) for t in batch]
                        input_ids, segment_ids, input_mask, label_id,seq_lengths = batch
                        targets, outputs =  evalute_CNN(model, batch,global_step3,len(iter_bar)) # accuracy to print
                        loss = get_loss_CNN(model, batch, global_step3).mean() # mean() for Data Parallelism
                        
                        _, predicted = torch.max(outputs.data, 1)
                        
                        correct += (np.array(predicted.cpu()) ==
                                    np.array(targets.cpu())).sum()
                        loss_total += loss.item() * input_ids.shape[0]
                        total_sample += input_ids.shape[0]
                        iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                        
                        
                acc_total = correct/total_sample
                loss_total = loss_total/total_sample
                ddf = open(result_name,'a', encoding='UTF8')
                ddf.write(str(t)+": "+ str(num_a)+"aucr: "+str(acc_total)+'\n')
                ddf.close()
                num_a+=1
                        
                
                    
                temp=987654321
                early_stopping = EarlyStopping(patience=30, verbose=True)
                valid_losses = []
                while(1):
                    model2.train()
                    loss_sum = 0
                    global_step3 = 0
                    iter_bar3 = tqdm(self.data_iter3, desc='Iter (loss=X.XXX)')
                    for i, batch in enumerate(iter_bar3):
                        batch = [t.to(self.device) for t in batch]
                        loss = get_loss_Attn_LSTM(model2, batch, global_step3).mean() # mean() for Data Parallelism
                        self.optimizer2.zero_grad()
                        loss.backward()
                        self.optimizer2.step()
                        global_step3 += 1
                        loss_sum += loss.item()
                        iter_bar3.set_description('Iter (loss=%5.3f)'%loss.item())

                        if global_step3 % self.cfg.save_steps == 0: # save
                            self.save(global_step3)

                        if self.cfg.total_steps and self.cfg.total_steps < global_step3:
                            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                            print('The Total Steps have been reached.')
                            self.save(global_step3) # save and finish when global_steps reach total_steps
                            return
                        
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    model2.eval()
                    loss_sum = 0.
                    global_step3 = 0
                    iter_bar_dev = tqdm(self.dataset_dev, desc='Iter (loss=X.XXX)')
                    for i, batch in enumerate(iter_bar_dev):
                        batch = [t.to(self.device) for t in batch]
                        loss = get_loss_Attn_LSTM(model2, batch, global_step3).mean() # mean() for Data Parallelism
                        valid_losses.append(loss.item())
                        global_step3 += 1
                        loss_sum += loss.item()
                        iter_bar_dev.set_description('Iter (loss=%5.3f)'%loss.item())

                        if global_step3 % self.cfg.save_steps == 0: # save
                            self.save(global_step3)

                        if self.cfg.total_steps and self.cfg.total_steps < global_step3:
                            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                            print('The Total Steps have been reached.')
                            self.save(global_step3) # save and finish when global_steps reach total_steps
                            return

                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    valid_loss = np.average(valid_losses)
                    loss_min=early_stopping(valid_loss, model2,"./model_save/checkpoint_LSTM_real.pt")
                    valid_losses = []
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    
                model2.eval()# evaluation mode
                loss_total = 0
                total_sample = 0
                acc_total = 0
                correct = 0
                global_step3=0
                
                with torch.no_grad():
                    iter_bar = tqdm(self.data_iter2, desc='Iter (f1-score=X.XXX)')
                    for batch in iter_bar:
                        batch = [t.to(self.device) for t in batch]
                        input_ids, segment_ids, input_mask, label_id,seq_lengths = batch
                        targets, outputs = evalute_Attn_LSTM(model2, batch, global_step3,len(iter_bar))# accuracy to print
                        loss = get_loss_Attn_LSTM(model2, batch, global_step3).mean() # mean() for Data Parallelism
                        
                        _, predicted = torch.max(outputs.data, 1)
                        
                        correct += (np.array(predicted.cpu()) ==
                                    np.array(targets.cpu())).sum()
                        loss_total += loss.item() * input_ids.shape[0]
                        total_sample += input_ids.shape[0]
                        iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                        
                print('total#####:', total_sample)
                print("correct#:#", correct)
                acc_total = correct/total_sample
                loss_total = loss_total/total_sample
                ddf = open(result_name,'a', encoding='UTF8')
                ddf.write(str(t)+": "+ str(num_a)+"aucr: "+str(acc_total)+'\n')
                ddf.close()
                num_a+=1
                
            elif(e%2==1):
                global_step1 = 0
                model2.eval()
                model.eval()
                labell=[]
                iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
                for batch in iter_bar:
                    batch = [t.to(self.device) for t in batch]
                    with torch.no_grad(): # evaluation without gradient calculation
                        label_id, y_pred1 = generating_lexiocn(model,model2, batch,global_step1,len(iter_bar),e) # accuracy to print
                        global_step1+=1
                
                global_step1 = 0
                model.eval()
                model2.eval()
                sen = []
                labell=[]
                iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
                for batch in iter_bar:
                    batch = [t.to(self.device) for t in batch]
                    with torch.no_grad(): # evaluation without gradient calculation
                        label_id, y_pred1,result_label,result3,data_temp,data_iter_temp_na = pseudo_labeling(model, model2,batch,global_step1,len(iter_bar),e) # accuracy to print
                        global_step1+=1
        
                self.data_iter_temp = data_temp
                self.data_iter = data_iter_temp_na
                #print(result3)
                num_good=0
                num_label=0
                num_label1=0
                ddf = open(pseudo_name,'a', encoding='UTF8')
                
                for i in range(0, len(result3)):
                    sen.append(result3[i])
                
                num_label=0
                num_label1=0
                num_good = 0
                for i in range(0, len(result3)):
                    if(result3[i] != -1):
                        num_good +=1
                        if(result3[i] == result_label[i]):
                            num_label+=1
                            
                ddf.write(str(t)+": "+"number of good :"+str(num_good)+" ")
                ddf.write("number of label :"+str(num_label)+" ")
                ddf.write("\n")
                
                ddf.close()
                if(num_good  < self.stopNum):
                    curTemp+=1
                else:
                    curTemp=0
                if(curTemp>=2):
                    break

                    

            elif(e%2==0 ):
                early_stopping = EarlyStopping(patience=10, verbose=True)
                valid_losses = []
                while(1):
                    model.train()
                    l=0
                    l_sum=0
                    loss_sum = 0.
                    global_step3 = 0
                    iter_bar3 = tqdm(self.data_iter_temp, desc='Iter (loss=X.XXX)')
                    for i, batch in enumerate(iter_bar3):
                        batch = [t.to(self.device) for t in batch]
                        loss = get_loss_CNN(model, batch, global_step3).mean() # mean() for Data Parallelism
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        global_step3 += 1
                        loss_sum += loss.item()
                        iter_bar3.set_description('Iter (loss=%5.3f)'%loss.item())

                    model.eval()
                    loss_sum = 0.
                    global_step3 = 0
                    iter_bar_dev = tqdm(self.dataset_dev, desc='Iter (loss=X.XXX)')
                    for i, batch in enumerate(iter_bar_dev):
                        batch = [t.to(self.device) for t in batch]
                        loss = get_loss_CNN(model, batch, global_step3).mean() # mean() for Data Parallelism
                        valid_losses.append(loss.item())
                        global_step3 += 1
                        loss_sum += loss.item()
                        iter_bar_dev.set_description('Iter (loss=%5.3f)'%loss.item())


                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))

                    valid_loss = np.average(valid_losses)
                    loss_min=early_stopping(valid_loss, model,cnn_save_name)
                    valid_losses = []

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                model.load_state_dict(torch.load(cnn_save_name))
                model.eval()# evaluation mode
                loss_total = 0
                total_sample = 0
                acc_total = 0
                correct = 0
                global_step3=0
                
                with torch.no_grad():
                    iter_bar = tqdm(self.data_iter2, desc='Iter (f1-score=X.XXX)')
                    for batch in iter_bar:
                        batch = [t.to(self.device) for t in batch]
                        input_ids, segment_ids, input_mask, label_id,seq_lengths = batch
                        targets, outputs =  evalute_CNN(model, batch,global_step3,len(iter_bar)) # accuracy to print
                        loss = get_loss_CNN(model, batch, global_step3).mean() # mean() for Data Parallelism
                        
                        _, predicted = torch.max(outputs.data, 1)
                        
                        correct += (np.array(predicted.cpu()) ==
                                    np.array(targets.cpu())).sum()
                        loss_total += loss.item() * input_ids.shape[0]
                        total_sample += input_ids.shape[0]
                        iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                        
                        
                acc_total = correct/total_sample
                loss_total = loss_total/total_sample
                ddf = open(result_name,'a', encoding='UTF8')
                ddf.write(str(t)+": "+ str(num_a)+"aucr: "+str(acc_total)+'\n')
                ddf.close()
                num_a+=1
                            
                 
                valid_losses = []            
                temp = 987654321
                early_stopping = EarlyStopping(patience=10, verbose=True)
                while(1):
                    model2.train()
                    l=0
                    l_sum=0
                    loss_sum = 0
                    global_step3 = 0
                    iter_bar3 = tqdm(self.data_iter_temp, desc='Iter (loss=X.XXX)')
                    for i, batch in enumerate(iter_bar3):
                        batch = [t.to(self.device) for t in batch]
                        loss = get_loss_Attn_LSTM(model2, batch, global_step3).mean() # mean() for Data Parallelism
                        self.optimizer2.zero_grad()
                        loss.backward()
                        self.optimizer2.step()
                        global_step3 += 1
                        loss_sum += loss.item()
                        iter_bar3.set_description('Iter (loss=%5.3f)'%loss.item())

                     
                        
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    model2.eval()
                    loss_sum = 0.
                    global_step3 = 0
                    iter_bar_dev = tqdm(self.dataset_dev, desc='Iter (loss=X.XXX)')
                    for i, batch in enumerate(iter_bar_dev):
                        batch = [t.to(self.device) for t in batch]
                        loss = get_loss_Attn_LSTM(model2, batch, global_step3).mean() # mean() for Data Parallelism
                        valid_losses.append(loss.item())
                        global_step3 += 1
                        loss_sum += loss.item()
                        iter_bar_dev.set_description('Iter (loss=%5.3f)'%loss.item())

                       

                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    valid_loss = np.average(valid_losses)
                    loss_min=early_stopping(valid_loss, model2,rnn_save_name)
                    valid_losses = []

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                model2.load_state_dict(torch.load(rnn_save_name))   
                model2.eval()# evaluation mode
                loss_total = 0
                total_sample = 0
                acc_total = 0
                correct = 0
                global_step3=0
                
                with torch.no_grad():
                    iter_bar = tqdm(self.data_iter2, desc='Iter (f1-score=X.XXX)')
                    for batch in iter_bar:
                        batch = [t.to(self.device) for t in batch]
                        input_ids, segment_ids, input_mask, label_id,seq_lengths = batch
                        targets, outputs = evalute_Attn_LSTM(model2, batch, global_step3,len(iter_bar))# accuracy to print
                        loss = get_loss_Attn_LSTM(model2, batch, global_step3).mean() # mean() for Data Parallelism
                        
                        _, predicted = torch.max(outputs.data, 1)
                        
                        correct += (np.array(predicted.cpu()) ==
                                    np.array(targets.cpu())).sum()
                        loss_total += loss.item() * input_ids.shape[0]
                        total_sample += input_ids.shape[0]
                        iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                        
                        
                acc_total = correct/total_sample
                loss_total = loss_total/total_sample
                ddf = open(result_name,'a', encoding='UTF8')
                ddf.write(str(t)+": "+ str(num_a)+"aucr: "+str(acc_total)+'\n')
                ddf.close()
                num_a+=1

         

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        
    def load2(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model2.load_state_dict(torch.load(model_file))

       

    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))
    def save2(self, i):
        """ save current model """
        torch.save(self.model2.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))
        
class Eval(object):
    """Training Helper Class"""
    def __init__(self, cfg, model,model2, data_iter,save_dir,device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.model2 = model2
        self.data_iter = data_iter # iterator to load data
        self.save_dir = save_dir
        self.device = device # device name


    def eval(self, evalute_CNN_SSL, evalute_Attn_LSTM_SSL,  data_parallel=True):
        """ Evaluation Loop """
  
        self.model.eval() # train mode
        self.model2.eval() # train mode
        model = self.model.to(self.device)
        model2 = self.model2.to(self.device)
        num_a=0
        global_step = 0 # global iteration steps regardless of epochs
        global_step3 = 0
        
        
        
        model.load_state_dict(torch.load("./model_save/checkpoint_CNN.pt"))
        model.eval()# evaluation mode
        p=[]
        l=[]
        p3=[]

        iter_bar4 = tqdm(self.data_iter, desc='Iter (f1-score=X.XXX)')
        for batch in iter_bar4:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                label_id, y_pred1 = evalute_CNN_SSL(model, batch) # accuracy to print
                _, y_pred3 = y_pred1.max(1)

                p2=[]
                l2=[]

                for i in range(0,len(y_pred3)):
                    p3.append(np.ndarray.flatten(y_pred3[i].data.cpu().numpy()))
                    l.append(np.ndarray.flatten(label_id[i].data.cpu().numpy()))
                    p2.append(np.ndarray.flatten(y_pred3[i].data.cpu().numpy()))
                    l2.append(np.ndarray.flatten(label_id[i].data.cpu().numpy()))
            p2 = [item for sublist in p2 for item in sublist]
            l2 = [item for sublist in l2 for item in sublist]

            result2  = f1_score(l2, p2,average='micro')
            iter_bar4.set_description('Iter(roc=%5.3f)'%result2)
        p3 = [item for sublist in p3 for item in sublist]
        l = [item for sublist in l for item in sublist]
        p=np.array(p)
        l=np.array(l)
        results2  = accuracy_score(l, p3)
        F1score = f1_score(l,p3,average='micro')
        ddf = open("./result/result.txt",'a', encoding='UTF8')

        ddf.write(str(num_a)+"aucr: "+str(results2)+"f1-score: "+str(F1score)+'\n')
        ddf.close()
        num_a+=1
        print("model1_accuracy: ", results2,"model1_f1score: ", F1score) 
        
        
        model2.load_state_dict(torch.load("./model_save/checkpoint_LSTM.pt"))   
        model2.eval()# evaluation mode
        p=[]
        l=[]
        p3=[]

        iter_bar4 = tqdm(self.data_iter, desc='Iter (f1-score=X.XXX)')
        for batch in iter_bar4:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                label_id, y_pred1 = evalute_Attn_LSTM_SSL(model2, batch) # accuracy to print
                _, y_pred3 = y_pred1.max(1)
                p2=[]
                l2=[]

                for i in range(0,len(y_pred3)):
                    p3.append(np.ndarray.flatten(y_pred3[i].data.cpu().numpy()))
                    l.append(np.ndarray.flatten(label_id[i].data.cpu().numpy()))
                    p2.append(np.ndarray.flatten(y_pred3[i].data.cpu().numpy()))
                    l2.append(np.ndarray.flatten(label_id[i].data.cpu().numpy()))
            p2 = [item for sublist in p2 for item in sublist]
            l2 = [item for sublist in l2 for item in sublist]

            result2  = f1_score(l2, p2,average='micro')
            iter_bar4.set_description('Iter(roc=%5.3f)'%result2)
        p3 = [item for sublist in p3 for item in sublist]
        l = [item for sublist in l for item in sublist]
        p=np.array(p)
        l=np.array(l)
        results2  = accuracy_score(l, p3)
        F1score = f1_score(l,p3,average='micro')
        ddf = open("./result/result.txt",'a', encoding='UTF8')
        ddf.write(str(num_a)+"aucr: "+str(results2)+"f1-score: "+str(F1score)+'\n')
        ddf.close()
        num_a+=1
        print("model2_accuracy: ", results2,"model2_f1score: ", F1score) 
        
