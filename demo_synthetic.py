import argparse
import numpy as np
from casme_engine_synthetic import *
from models_synthetic import *
from casme import *
from tensorboardX import SummaryWriter
import time

parser = argparse.ArgumentParser(description='MERGCN Training')
#parser.add_argument('--data', metavar='DIR',
#                    default='./CASME/Cropped',help='path to dataset (e.g. CASME/Cropped)')
#parser.add_argument('--data_list_path',type=str,
#                    default='CASME/CASME2-coding-20190701.xlsx',help='path to the list of path and label (e.g. CASME/CASME2-coding-20190701.xlsx)')
#parser.add_argument('--inp_name',type=str, default='./CASME/casme_au_one_hot_19.pkl',
#                    help='path to the pkl of one-hot au (e.g. CASME/casme_au_one_hot_19.pkl)')
parser.add_argument('--data', metavar='DIR',
                    default='./synthetic_casme',help='path to dataset (e.g. CASME/Cropped)')
parser.add_argument('--data_list_path',type=str,
                    default='synthetic_label.xlsx',help='path to the list of path and label (e.g. CASME/CASME2-coding-20190701.xlsx)')
parser.add_argument('--inp_name',type=str, default='./synthetic_adj_18_new.pkl',
                    help='path to the pkl of one-hot au (e.g. CASME/casme_au_one_hot_19.pkl)')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 112)')
parser.add_argument('--max_length', default=141, type=int,
                    metavar='N', help='max sequence length (default: 141)')
parser.add_argument('--num_AU', default=18, type=int,
                    metavar='N', help='number of AU (CASMEII default: 19)')
parser.add_argument('--num_classes', default=3, type=int,
                    metavar='N', help='number of emotion classes (default: 7)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run(default: 100)')
parser.add_argument('--epoch_step', default=[5], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1, for different length)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume_for_evaluate', default='', type=str, metavar='PATH',
                    help='path to model for evaluation (e.g. checkpoint/casme/model_0.490196.pkl)')
parser.add_argument('--model_save_path', default='checkpoint/synthetic/', type=str, metavar='PATH',
                    help='path to save best model (default: checkpoint/casme/)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resnet', dest='resnet', action='store_true',
                    help='use only resnet')
parser.add_argument('--eva_protocol',default='loso',help='loso or lovo')


def get_seq_path_and_label_lovo(dir_path,data_list_path,max_length,evaluate,video_out_idx):
    # LOSO validation
    if not evaluate:

        MAX_LENGTH = max_length
        file = data_list_path
        # load data_file to df
        df = pd.read_excel(file).drop(['Unnamed: 2','Unnamed: 6'], axis=1)

        # NEW: build AU label dictionary !!
        import itertools
        from collections import Counter
        import ast
        from sklearn.preprocessing import MultiLabelBinarizer
        # calculate occurance of each AU

        au_list = [str(i) for i in df['Action Units'].values.tolist()]        
        au_list = [str(i).split("+") for i in au_list] #remove "+"

        #au_list =  [str(s).replace(cha,'') for cha in ['L','R'] for s in au_list]# remove "L","R"
        au_list_ =  [str(s).replace('L','') for s in au_list]# remove "L","R"
        au_list_ =  [str(s).replace('R','') for s in au_list_]# remove "L","R"

        #print("len of au_list",len(au_list_)) #255
        for i in range(len(au_list_)):
            au_list_[i] = ast.literal_eval(au_list_[i])
            au_list_[i] = [int(i) for i in au_list_[i]]

 
        au_onehot = MultiLabelBinarizer().fit_transform(au_list_)
        au_nums = dict(Counter(i for i in list(itertools.chain.from_iterable(au_list_))))


        # build emotion label dictionary 
        #emotion_np = pd.Series.as_matrix(df['Estimated Emotion'])#pandas version problem
        #emotion_np = pd.Series.to_numpy(df['Estimated Emotion'])
        #emotion_set = set(emotion_np)
        #emotion_list = list(emotion_set)

        emotion_list = ['positive','negative','surprise']

        emotion_dic = {}
        for i in range(3):
            emotion_dic[emotion_list[i]] = i
        # build sequence of path 
        train_seq = []
        # randomly pick validation sequences out of 255, k-fold validation
        val_seq = []
        vid_length = 0
        for index, row in df.iterrows():
            
            sub = row['Subject']
            seq_name = row['Filename']
            seq_length =row["OffsetFrame"] - row["OnsetFrame"] + 1
            #front_padding = int(np.ceil((MAX_LENGTH - seq_length)/2) -1)
            label = row['Estimated Emotion'] #str


            if label == 'others' or label=='repression':
                continue
            elif label == 'happiness':
                vid_length += 1
                label = 'positive'
                seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
                label = emotion_dic[label]
                au_label = au_onehot[index]
            elif label == 'sadness' or label =='fear' or label =='disgust':
                vid_length += 1
                #print(label)
                label = 'negative'
                seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
                label = emotion_dic[label]
                au_label = au_onehot[index]
            elif label == 'surprise':
                vid_length += 1
                label = 'surprise'
                seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
                label = emotion_dic[label]
                au_label = au_onehot[index]
                
                
            label_ = []  
            label_.append(label) #convert class int to class list
            label_ = np.asarray(label_) # list->ndarray
            label = np.concatenate((label_, au_label),axis =0) # concat emotion label and au label 
            #print("label for {}-{} is:{}".format(sub,seq_name, label))
            #assert 0
            #print("vid_length:",vid_length)
            if vid_length == video_out_idx:
                #print("this is validation sequences")
                val_seq.append((seq_path,label))
                
            else:
                train_seq.append((seq_path,label))
    #print(video_out_idx)
    #print("train sequences:",len(train_seq))
    #print("val sequences:",len(val_seq))
    return train_seq, val_seq, emotion_dic 

                        

def get_seq_path_and_label_loso(dir_path,data_list_path,max_length,evaluate,subject_out_idx):
    # LOSO validation
    if not evaluate:

        MAX_LENGTH = max_length
        file = data_list_path
        # load data_file to df
        #df = pd.read_excel(file).drop(['Unnamed: 2','Unnamed: 6'], axis=1)
        df = pd.read_excel(file)

        #print(df.head())
        #assert 0

        # NEW: build AU label dictionary !!
        import itertools
        from collections import Counter
        import ast
        from sklearn.preprocessing import MultiLabelBinarizer
        # calculate occurance of each AU

        au_list = [str(i) for i in df['Action Units'].values.tolist()]        
        au_list = [str(i).split("+") for i in au_list] #remove "+"

        #au_list =  [str(s).replace(cha,'') for cha in ['L','R'] for s in au_list]# remove "L","R"
        au_list_ =  [str(s).replace('L','') for s in au_list]# remove "L","R"
        au_list_ =  [str(s).replace('R','') for s in au_list_]# remove "L","R"

        #print("len of au_list",len(au_list_)) #255
        for i in range(len(au_list_)):
            au_list_[i] = ast.literal_eval(au_list_[i])
            au_list_[i] = [int(i) for i in au_list_[i]]

 
        au_onehot = MultiLabelBinarizer().fit_transform(au_list_)
        au_nums = dict(Counter(i for i in list(itertools.chain.from_iterable(au_list_))))


        # build emotion label dictionary 
        #emotion_np = pd.Series.as_matrix(df['Estimated Emotion'])#pandas version problem
        emotion_np = pd.Series.to_numpy(df['Estimated Emotion'])
        emotion_set = set(emotion_np)
        emotion_list = list(emotion_set)

        emotion_list = ['positive','negative','surprise']

        emotion_dic = {}
        for i in range(args.num_classes):
            emotion_dic[emotion_list[i]] = i
        # build sequence of path 
        train_seq = []
        # randomly pick validation sequences out of 255, k-fold validation
        val_seq = []
        
        for index, row in df.iterrows():
           

            sub = row['Subject']

            if sub == subject_out_idx:

                seq_name = row['Filename']
                seq_length =row["OffsetFrame"] - row["OnsetFrame"] + 1

                #front_padding = int(np.ceil((MAX_LENGTH - seq_length)/2) -1)
                label = row['Estimated Emotion'] #str


                if label == 'others' or label=='repression':
                    continue
                elif label == 'happiness':
                    label = 'positive'
                    seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
                    label = emotion_dic[label]
                    au_label = au_onehot[index]
                elif label == 'sadness' or label =='fear' or label =='disgust':
                    #print(label)
                    label = 'negative'
                    seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
                    label = emotion_dic[label]
                    au_label = au_onehot[index]
                elif label == 'surprise':
                    label = 'surprise'
                    seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
                    label = emotion_dic[label]
                    au_label = au_onehot[index]
                
                
                label_ = []   
     
                label_.append(label) #convert class int to class list
                label_ = np.asarray(label_) # list->ndarray
                label = np.concatenate((label_, au_label),axis =0) # concat emotion label and au label 
                val_seq.append((seq_path,label))

         

 


            else:
                seq_name = row['Filename']
                seq_length =row["OffsetFrame"] - row["OnsetFrame"] + 1
                #front_padding = int(np.ceil((MAX_LENGTH - seq_length)/2) -1)
                label = row['Estimated Emotion'] #str


                if label == 'others' or label=='repression':
                    continue
                elif label == 'happiness':
                    label = 'positive'
                    seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
                    label = emotion_dic[label]
                    au_label = au_onehot[index]
                elif label == 'sadness' or label =='fear' or label =='disgust':
                    #print(label)
                    label = 'negative'
                    seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
                    label = emotion_dic[label]
                    au_label = au_onehot[index]
                elif label == 'surprise':
                    label = 'surprise'
                    seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
                    label = emotion_dic[label]
                    au_label = au_onehot[index]                
                
                label_ = []   
     
                label_.append(label) #convert class int to class list
                label_ = np.asarray(label_) # list->ndarray
                label = np.concatenate((label_, au_label),axis =0) # concat emotion label and au label 
                train_seq.append((seq_path,label))


    else:
        assert 0
        MAX_LENGTH = max_length
        file = data_list_path
        # load data_file to df
        df = pd.read_excel(file).drop(['Unnamed: 2','Unnamed: 6'], axis=1)
         # NEW: build AU label dictionary !!
        import itertools
        from collections import Counter
        import ast
        from sklearn.preprocessing import MultiLabelBinarizer
        # calculate occurance of each AU

        au_list = [str(i) for i in df['Action Units'].values.tolist()]        
        au_list = [str(i).split("+") for i in au_list] #remove "+"

        #au_list =  [str(s).replace(cha,'') for cha in ['L','R'] for s in au_list]# remove "L","R"
        au_list_ =  [str(s).replace('L','') for s in au_list]# remove "L","R"
        au_list_ =  [str(s).replace('R','') for s in au_list_]# remove "L","R"

        #print("len of au_list",len(au_list_)) #255
        for i in range(len(au_list_)):
            au_list_[i] = ast.literal_eval(au_list_[i])
            au_list_[i] = [int(i) for i in au_list_[i]]

 
        au_onehot = MultiLabelBinarizer().fit_transform(au_list_)
        #au_list = ast.literal_eval(au_list)
        #au_list = [int(i) for i in au_list]

        #print("shape of au_onehot(should be 19#):",au_onehot.shape)


        au_nums = dict(Counter(i for i in list(itertools.chain.from_iterable(au_list_))))
        #print("AU numbers:",len(au_nums))
        #assert 0

        # build emotion label dictionary 
        emotion_np = pd.Series.as_matrix(df['Estimated Emotion'])
        emotion_set = set(emotion_np)
        emotion_list = list(emotion_set)
        emotion_dic = {}
        for i in range(args.num_classes):
            emotion_dic[emotion_list[i]] = i
        
        # build sequence of path 
        train_seq = []
        val_seq = []

        for index, row in df.iterrows():
            sub = row['Subject']
            seq_name = row['Filename']
            seq_length =row["OffsetFrame"] - row["OnsetFrame"] + 1
            label = row['Estimated Emotion']
            #front_padding = int(np.ceil((MAX_LENGTH - seq_length)/2) -1)

            seq_path = os.path.join(dir_path,'sub'+str(sub).zfill(2),seq_name)
            label = emotion_dic[label]
            au_label = au_onehot[index]

          
            label_ = []
            label_.append(label)
            label_ = np.asarray(label_) # list->ndarray
            label = np.concatenate((label_, au_label),axis =0)
           

            val_seq.append((seq_path,label))
            
    return train_seq, val_seq, emotion_dic 

def main_casme():
    start = time.time()
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    # use gpu
    use_gpu = torch.cuda.is_available()
    
    # define dataset
    if args.eva_protocol == 'loso':
        #LOSO 
        acc_avg_all=0
        sub_num = 0 
        for i in range(26):

            _train,_val,emo_dic = get_seq_path_and_label_loso(args.data,args.data_list_path,args.max_length,args.evaluate,subject_out_idx=i+1) 

            #print(len(_val))
            if len(_val)==0:
                continue
            sub_num += 1
            train_dataset = casme_dataset(root=args.data, split='train',inp_name=args.inp_name,max_length=args.max_length,\
                                    sequences=_train,dic=emo_dic,data_list=args.data_list_path,\
                                    transform=None, adj=None)
            val_dataset = casme_dataset(root=args.data, split='test',inp_name=args.inp_name,max_length=args.max_length,\
                                            sequences= _val ,dic=emo_dic,data_list=args.data_list_path,\
                                            transform=None, adj=None)

            num_classes = args.num_classes
            num_AU = args.num_AU
            # load model
            if args.resnet==True:
                model = resnet3d(num_classes=num_classes)
                print("### using only resnet3d for training ###")
                assert 0
            model = gcn_resnet3d(num_classes=num_classes, num_AU=num_AU, t=0, pretrained=True, adj_file=args.inp_name)
            #assert 0
            # define loss function (criterion)
            criterion = nn.CrossEntropyLoss()
            au_criterion = nn.BCEWithLogitsLoss()
            #writer = SummaryWriter()

            # define optimizer
            optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

            # load saved model
            #chk = torch.load('./checkpoint/casme_general/model_0.567797.pkl')
            #model.load_state_dict(chk['model_state_dict'])


            state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
                        'evaluate': args.evaluate,'resume': args.resume_for_evaluate, 'num_classes':num_classes}
            state['difficult_examples'] = True
            state['save_model_path'] = args.model_save_path
            state['workers'] = args.workers
            state['epoch_step'] = args.epoch_step
            state['lr'] = args.lr
            # start training
            engine = Engine(state)    
            acc_avg = engine.learning(model, criterion, au_criterion,train_dataset, val_dataset, optimizer)
            f = open("loso_acc.txt",'a')
            #print("i:",str(i))
            #print("acc:",str(acc_avg))
            f.write(str(i)+' ')
            f.write(str(acc_avg))
            f.write('\n')
            #assert 0
            acc_avg_all = acc_avg + acc_avg_all

        acc_avg_all = float(acc_avg_all/sub_num)
        print("final acc_avg on total -{}- subjects is: -{}-:".format(sub_num, acc_avg_all) )

        #print(len(_train))
        #print(len(_val))
    else:
        #LOVO
        
        acc_avg_all=0
        for i in range(129):

            _train,_val,emo_dic = get_seq_path_and_label_lovo(args.data,args.data_list_path,args.max_length,args.evaluate,video_out_idx=i+1) # video_out_idx can change

            train_dataset = casme_dataset(root=args.data, split='train',inp_name=args.inp_name,max_length=args.max_length,\
                                    sequences=_train,dic=emo_dic,data_list=args.data_list_path,\
                                    transform=None, adj=None)
            val_dataset = casme_dataset(root=args.data, split='test',inp_name=args.inp_name,max_length=args.max_length,\
                                            sequences= _val ,dic=emo_dic,data_list=args.data_list_path,\
                                            transform=None, adj=None)

            num_classes = args.num_classes
            num_AU = args.num_AU
            # load model
            if args.resnet==True:
                model = resnet3d(num_classes=num_classes)
                print("### using only resnet3d for training ###")
                assert 0
            model = gcn_resnet3d(num_classes=num_classes, num_AU=num_AU, t=0, pretrained=True, adj_file=args.inp_name)
            #assert 0
            # define loss function (criterion)
            criterion = nn.CrossEntropyLoss()
            au_criterion = nn.BCEWithLogitsLoss()
            #writer = SummaryWriter()

            # define optimizer
            optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)


            # load saved model
            chk = torch.load('./checkpoint/casme_general/model_0.460938.pkl')
            model.load_state_dict(chk['model_state_dict'])
            #optimizer.load_state_dict(chk['optimizer_state_dict'])
            #epoch = chk['epoch']
            #loss = chk['loss']
            #model.eval()
            #assert 0


            state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
                        'evaluate': args.evaluate,'resume': args.resume_for_evaluate, 'num_classes':num_classes}
            state['difficult_examples'] = True
            state['save_model_path'] = args.model_save_path
            state['workers'] = args.workers
            state['epoch_step'] = args.epoch_step
            state['lr'] = args.lr
            # start training
            engine = Engine(state)    
            acc_avg = engine.learning(model, criterion, au_criterion,train_dataset, val_dataset, optimizer)
            acc_avg_all = acc_avg + acc_avg_all
            f = open("lovo_acc.txt",'a')
            f.write(str(i)+' ')
            f.write(str(acc_avg))
            f.write('\n')
        acc_avg_all = float(acc_avg_all/129)
        print("final acc_avg:", acc_avg_all)
        #print(len(_train))
        #print(len(_val))

    end = time.time()
    print("total runing time is {}s:".format(end-start) )


    


if __name__ == '__main__':
    main_casme()
