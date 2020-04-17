import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from util import *
from tensorboardX import SummaryWriter
from sklearn.metrics import recall_score, f1_score

writer = SummaryWriter()

tqdm.monitor_interval = 0


class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()
        print("use_gpu:",self.state['use_gpu'])
        if self._state('image_size') is None:
            self.state['image_size'] = 224 #112
        print('image size:',self.state['image_size'])
        if self._state('batch_size') is None:
            self.state['batch_size'] = 32
        print('batch size:',self.state['batch_size'])
        if self._state('workers') is None:
            self.state['workers'] = 25
        print('num_workers:',self.state['workers'])
        
        if self._state('evaluate') is True:
            #self.state['evaluate'] = False
            print('### evaluate mode ###')
            print('use model ',self.state['resume'])
            
        else:
            print("### train mode ###")
            save_path = self.state['save_model_path']
            #if ( os.path.isdir(save_path)):
             #   shutil.rmtree(save_path)
                #os.system('rm -rf save_path')
                #raise AssertionError ("model-saving folder exists! ")
            #os.makedirs(self.state['save_model_path'])
            print('model will be saved in:',self.state['save_model_path']) #checkpoint/casme/
      
        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        # meters (measure a range of different measures,)
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True ##
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, au_criterion,data_loader, optimizer, display=True):
        self.state['correct'] = 0
        self.state['target_recall_score'] = 0
        self.state['f1_score'] = 0
        self.state['length'] = 0
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()
        self.state['mean_acc'] = 0

    def on_end_epoch(self, training, model, criterion, au_criterion,data_loader, optimizer, display=True):
        loss,_ = self.state['meter_loss'].value() # average loss among the whole epoch
        accuracy = self.state['correct']/self.state['length']
        target_recall_score = self.state['target_recall_score']/self.state['length']
        target_f1_score = self.state['f1_score']/self.state['length']
        # write loss value and acc into file for final visulization 

        #print(self.state['correct'],self.state['length'])
        #mean_acc += accuracy

        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'ACC {acc:.4f}\t'
                      'F1 Score {f1_score:.4f}\t'
                      'Recall score {recall:.4f}'.format(self.state['epoch'], loss=loss,acc=accuracy, f1_score=target_f1_score,recall=target_recall_score))

                writer.add_scalars('data/train_loss', {'Total Loss': loss},
                                  #'ME Loss': self.state['me_loss'].item(),
                                 #'AU Loss': self.state['au_loss'].item()},
                                 self.state['epoch'])

                if self.state['best_score'] < accuracy:
                    print("#########BEST MODEL###########")
                    self.state['best_score'] = accuracy

                    # type2: Saving a General Checkpoint for Inference and/or Resuming Training
                    torch.save({'epoch':self.state['epoch'],
                                    'model_state_dict':model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': loss, }, './checkpoint/casme_general/'+'model_%f.pkl'%float(accuracy))
                    
            else:
                print('Test Loss {loss:.4f} \t Test correct :{corr}\t Test ACC {acc:.4f}\t Test F1 Score {f1_score:.4f}\t Test Recall {recall:.4f}'.format(corr=self.state['correct'],loss=loss, acc=accuracy,f1_score=target_f1_score,recall=target_recall_score))
                if self.state['evaluate']==False:
                    writer.add_scalars('data/validate_loss', {'Total Loss': loss,
                                        },
                                 # 'ME Loss': self.state['me_loss'].item(),
                                 #'AU Loss': self.state['au_loss'].item()},
                                 self.state['epoch'])
                    writer.add_scalars('data/validate_acc', {
                                        'validate_acc': accuracy},
                                 # 'ME Loss': self.state['me_loss'].item(),
                                 #'AU Loss': self.state['au_loss'].item()},
                                 self.state['epoch'])
                    img_batch = np.zeros((16, 3, 100, 100))
                    for i in range(16):
                        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
                        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

                    writer.add_images('my_image_batch', img_batch, 0)
                    #writer.close()
               
                    if self.state['best_score'] < accuracy:
                        print("#########BEST MODEL###########")
                        self.state['best_score'] = accuracy
                        
                        # type1: Saving model for inference
                        torch.save(model.state_dict(), self.state['save_model_path']+'model_%f.pkl'%float(accuracy))

                      
        return loss, accuracy


    def on_end_batch(self, training, model, criterion,au_criterion, data_loader, optimizer, display=True):
        # record loss
        self.state['loss_batch'] = self.state['loss'].item()

        self.state['meter_loss'].add(self.state['loss_batch'])  # accumulate loss
    def one_hot_embedding(self,labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes) 
        return y[labels] 

    def on_forward(self, training, model, criterion, au_criterion, data_loader, optimizer, display=True):

        feature_var = torch.autograd.Variable(self.state['feature']).float()
        #target_var = torch.autograd.Variable(self.state['target']).float()
        target_var = torch.autograd.Variable(self.state['target']).long() # for classification
        #print("target_var:",target_var)

        target_onehot=self.one_hot_embedding(target_var,3).long() #num_classes

        #print(target_onehot)
        #assert 0

        au_target_var = torch.autograd.Variable(self.state['au_target']).long() # for au classification [19]
        #print(au_target_var)
        #assert 0


        au_target_var = au_target_var.view(1,18) #onehot format already
        

       
        inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot
        #inp_var = torch.autograd.Variable(self.state['input']).float()

        '''if not training:
            feature_var.volatile = True
            target_var.volatile = True
            inp_var.volatile = True'''

        # compute output
        '''if training:
            optimizer.zero_grad()'''
        
        self.state['output'], self.state['au_output'] = model(feature_var, inp_var)
        au_target_var = au_target_var.type_as(self.state['au_output'])
        #print(self.state['au_output'])
        
        #self.state['au_output'] = self.state['au_output'][0,:] # [19]!
        #print("self.state['au_output'].size():",self.state['au_output'].size()) #without previous layer->[1, 19]!
        ##print("self.state['au_output']:",self.state['au_output'])
        ##assert 0
        ##self.state['output'] = torch.autograd.Variable(self.state['output']).long()
        
        #print(len(self.state['output']))

        prediction = torch.argmax(self.state['output'],1)
        predict_onehot=self.one_hot_embedding(prediction,3) # num_classes

        
        target_onehot_cpu = target_onehot.cpu().detach().numpy()
        predict_onehot_cpu = predict_onehot.cpu().detach().numpy()

        self.state['target_recall_score'] += recall_score(predict_onehot_cpu, target_onehot_cpu, average='weighted') # tp / (tp + fn)
        self.state['f1_score'] += f1_score(predict_onehot_cpu, target_onehot_cpu, average='weighted') #F1 = 2 * (precision * recall) / (precision + recall)
        self.state['correct'] += (prediction==target_var).sum().float()

        self.state['me_loss'] = criterion(self.state['output'], target_var)# should add au loss
        
        #_, au_target_var = au_target_var.max(dim=0,keepdim=True)  #[1]
        #print(self.state['au_output'].long().dtype)


        #print(self.state['au_output'].dtype)

        self.state['au_loss'] = au_criterion(self.state['au_output'], au_target_var)
        
        self.state['loss'] = 0.6*self.state['me_loss'] +  0.4*self.state['au_loss']   # combine emotion loss with au loss
        


        if training:
            loss_history = []
            optimizer.zero_grad()
            self.state['loss'].backward()
            loss_history.append(self.state['loss'].item())
            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()


    def learning(self, model, criterion, au_criterion,train_dataset, val_dataset, optimizer=None):

        #self.init_learning(model, criterion)
        self.state['best_score'] = 0
        
        
        
        # data loading 
        if self.state['evaluate'] == False:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.state['batch_size'], shuffle=True,
                                                       num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])


        
        if self.state['use_gpu']:
            print ("############ using gpu ################")
            if self.state['evaluate'] == False:
                train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True
            model.cuda()
            criterion = criterion.cuda()
            au_criterion = au_criterion.cuda()

        if self.state['evaluate']:
            model.load_state_dict((torch.load(self.state['resume'])))
            self.validate(val_loader, model, criterion,au_criterion)
            return

        acc_epoch_add = 0
        len_epoch = self.state['max_epochs']-self.state['start_epoch']

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)

            # train for one epoch
            self.train(train_loader, model, criterion, au_criterion,optimizer, epoch)
            
            # evaluate on validation set
            #prec1 = self.validate(val_loader, model, criterion)
            acc_epoch = self.validate(val_loader, model, criterion,au_criterion).item()


            #print("acc_epoch for every epoch:", acc_epoch)

            acc_epoch_add = acc_epoch_add + acc_epoch
            #print("acc_epoch_add", acc_epoch_add)


            
        acc_avg = float(acc_epoch_add / len_epoch)
        print("average acc of current subject in total -{}- epoch is: {}".format(len_epoch, acc_avg))
        return acc_avg

    def train(self, data_loader, model, criterion,au_criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, au_criterion,data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            self.state['length'] += len(target)
            
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])
            self.state['target'] = target[0,0:1].cuda() #NEW: target->target[0,0:1]
            self.state['au_target'] = target[0,1:].cuda()
            #print("au target size is: ",target[0,1:].size()) # 19#
            #print("emotion target size is:", target[0,0:1].size()) # 1#
   
            #assert 0
            self.state['feature'] = input[0].cuda() # seq_data
            self.state['input'] = input[2].cuda() # self.inp
            self.on_forward(True, model, criterion, au_criterion,data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion,au_criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, au_criterion,data_loader, optimizer)

        #print("accuracy for every epoch",acc_test)


    def validate(self, data_loader, model, criterion,au_criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, au_criterion,data_loader, optimizer=None)

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            self.state['length'] += len(target)
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])
            self.state['target'] = target[0,0:1].cuda()#NEW: target->target[0,0:1]
            self.state['feature'] = input[0].cuda()
            self.state['input'] = input[2].cuda()

            self.on_forward(False, model, criterion, au_criterion,data_loader,optimizer=None)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, au_criterion,data_loader, optimizer=None)


        _, acc_epoch = self.on_end_epoch(False, model, criterion, au_criterion,data_loader, optimizer=None)
        #torch.save(model.state_dict(), self.state['save_model_path']+'model_%f.pkl'%float(acc_epoch))
        return acc_epoch


    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)
