# Newest 2018.11.23 9:53:00

from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#from keras.utils import to_categorical
import pickle
import config as cf
import numpy as np

import os
import sys
import time
import argparse
import matplotlib.pyplot as plt

from dataset import get_all_dataLoders, get_dataLoder
from models import EDIL, leaf, Clustering
from losses import pred_loss, consistency_loss
from utils import get_optim, get_all_assign, stack_or_create


parser = argparse.ArgumentParser(description='HD_CNN in PyTorch')
parser.add_argument('--dataset', default='cifar-100', type=str, help='Determine which dataset to be used')
parser.add_argument('--num_superclasses', default=2, type=int, help='The number of cluster centers')
#parser.add_argument('--num_epochs_pretrain', default=1, type=int, help='The number of pre-train epoches')
parser.add_argument('--num_epochs_train', default=200, type=int, help='The number of train epoches')
parser.add_argument('--pretrain_batch_size', default=128, type=int, help='The batch size of pretrain')
parser.add_argument('--train_batch_size', default=256, type=int, help='The batch size of train')
parser.add_argument('--min_classes', default=1, type=int, help='The minimum of classes in one superclass')
parser.add_argument('--num_test', default= 500, type=int, help='The number of test batch steps in a epoch ================= when final test, make it bigger')

parser.add_argument('--opt_type', default='sgd', type=str, help='Determine the type of the optimizer')
parser.add_argument('--pretrain_lr', default=0.1, type=float, help='The learning rate of pre-training')
parser.add_argument('--finetune_lr', default=0.001, type=float, help='The learning rate of inner model')
parser.add_argument('--drop_rate', default=0.5, type=float, help='The probability of to keep')
parser.add_argument('--weight_consistency', default=1e1, type=float, help='The weight of coarse category consistency')
parser.add_argument('--gamma', default=5, type=float, help='The weight for u_k')#1.25

parser.add_argument('--resume_model', default= True, type=bool, help='resume the whole model from checkpoint')
args = parser.parse_args()

# Hyper Parameter settings
cf.use_cuda = torch.cuda.is_available()

trainloader, testloader, pretrainloader, validloader = get_all_dataLoders(args, valid=True, one_hot=True)
args.num_classes = 10 if args.dataset == 'cifar-10' else 100

# ============== Data class Preparation =============
# this generate the dict to feed the data
Class_generator = {}
for i in range(10):
    Class_generator[i]=np.arange(i*10,(i+1)*10)


# Model
print('\nModel setup')
net = EDIL(args)
function = Clustering(args)

# ============== some important dictionary ============
Leaf = {}
Branch = {}
Branch_clusting_result = {}
Superclass = {}
New_superclass = {}

# ============== save all important dictionaries ============
def save_and_load(Superclass, Branch, Branch_clusting_result, Leaf,  num = 0 ,load = False):
    save_point = cf.var_dir + args.dataset
    if not load:
        if num%10 == 0 and num != 0:
            torch.save(Superclass ,save_point+'/classes'+str(num)+'_Superclass.pkl')
            torch.save(Branch ,save_point+'/classes'+str(num)+'_Branch.pkl')
            torch.save(Branch_clusting_result ,save_point+'/classes'+str(num)+'_Branch_clusting_result.pkl')
            torch.save(Leaf ,save_point+'/classes'+str(num)+'_Leaf.pkl')
            print('\n================= Model saving finish ================')
        torch.save(Superclass ,save_point+'/Superclass.pkl')
        torch.save(Branch ,save_point+'/Branch.pkl')
        torch.save(Branch_clusting_result ,save_point+'/Branch_clusting_result.pkl')
        torch.save(Leaf ,save_point+'/Leaf.pkl')
        print('\n================= Model saving finish ================')
    else:
        if os.path.exists(save_point+'/Superclass.pkl'):
            Superclass = torch.load(save_point+'/Superclass.pkl')
        if os.path.exists(save_point+'/Branch.pkl'):
            Branch = torch.load(save_point+'/Branch.pkl')
        if os.path.exists(save_point+'/Branch_clusting_result.pkl'):
            Branch_clusting_result = torch.load(save_point+'/Branch_clusting_result.pkl')
        if os.path.exists(save_point+'/Leaf.pkl'):
            Leaf = torch.load(save_point+'/Leaf.pkl')
        return Superclass, Branch, Branch_clusting_result, Leaf


# ============== prepare the model and load parameters if possible ============
def prepared_model(num_output, dict):
    save_point = cf.var_dir + args.dataset
    model = leaf(args, num_output)
    if len(dict)>1:
        print('\nLoad the model: Leaf'+ dict[:-1]+ ' for: Leaf'+ dict)
        variables = torch.load(save_point + '/Leaf'+ dict[:-1]+'.pkl')
    elif len(dict)==1:
        if os.path.exists(save_point + '/Leaf0.pkl'):
            print('\nLoad the model: Leaf0 for: Leaf'+dict)
            variables = torch.load(save_point + '/Leaf0.pkl')
    if len(dict)>=1 and os.path.exists(save_point + '/Leaf0.pkl'):
        needs = {}
        for need in model.state_dict():
            if need in variables:
                needs[need] = variables[need]
            else:
                needs[need] = model.state_dict()[need]
        model.load_state_dict(needs)
    return model


# ============== save specific model ============
def save_model(model, dict, name = 'Leaf'):
    print('\nSaving the leaf model: Leaf'+dict)
    save_point = cf.var_dir + args.dataset
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    share_variables = model.state_dict()
    if name == 'Leaf':
        share_variables.pop('fc2.bias','dict: fc2.bias dot found in dictionary share_variables')
        share_variables.pop('fc2.weight','dict: fc2.weight dot found in dictionary share_variables')
    torch.save(share_variables, save_point + '/'+ name + dict+'.pkl')


# ============== train the branch to increase the rooting acc ============
def train_branch(branch, clusting_result, classes):
    for epoch in range(args.num_epochs_train):
        required_train_loader = get_dataLoder(args, classes = classes, mode='Train', one_hot=True)
        param = list(branch.parameters())
        optimizer, lr = get_optim(param, args, mode='preTrain', epoch=epoch)
        for batch_idx, (inputs, targets) in enumerate(required_train_loader):
            if cf.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets).float()
            outputs = branch(inputs)
            matrix = np.vstack(((np.ones(np.shape(clusting_result))-clusting_result), clusting_result))
            matrix = torch.from_numpy(matrix.transpose().astype(np.float32))
            if cf.use_cuda:
                matrix = matrix.cuda()
            outputs = outputs.mm(matrix)
            targets = targets.mm(matrix)
            loss = pred_loss(outputs,targets)
            loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update
            sys.stdout.write('\r')
            sys.stdout.write('Train Branch with Epoch [%3d/%3d] Iter [%3d/%3d]\t\t Loss: %.4f'
                         %(epoch+1, args.num_epochs_train, batch_idx+1, (pretrainloader.dataset.train_data.shape[0]//args.pretrain_batch_size)+1,
                           loss.item()))
            sys.stdout.flush()
    return branch

# ============== calculate the rooting and final classification acc ============
def branch_accuracy(branch, model0, model1, clusting_result, classes):
    required_valid_loader = get_dataLoder(args, classes = classes, mode='Valid', one_hot=False)
    num_ins = 0
    correct = 0
    root_count = 0
    root_right = 0
    for batch_idx, (inputs, targets) in enumerate(required_valid_loader):
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        inputs, targets = Variable(inputs), Variable(targets).long()
        # targets = torch.argmax(targets,1)
        outputs = branch(inputs)
        matrix = np.vstack(((np.ones(np.shape(clusting_result))-clusting_result), clusting_result))
        branch_result = np.dot(outputs.data.cpu().numpy(), matrix.transpose())
        outputs = torch.argmax(outputs,1)
        outputs0 = model0(inputs)
        outputs0 = np.argmax(outputs0.data.cpu().numpy(),1)
        outputs1 = model1(inputs)
        outputs1 = np.argmax(outputs1.data.cpu().numpy(),1)
        count0=[];
        count1=[];
        for i in range (np.shape(classes)[0]):
            if clusting_result[i]==0:
                count0 += [classes[i]]
            else:
                count1 += [classes[i]]
        if cf.use_cuda:
            count0, count1 = torch.Tensor(count0).cuda(), torch.Tensor(count1).cuda()
        else:
            count0, count1 = torch.Tensor(count0), torch.Tensor(count1)
        for i in range(np.shape(branch_result)[0]):
            if branch_result[i,0]>=branch_result[i,1]:
                outputs[i] = count0[outputs0[i]]
                if targets[i].to(torch.float32) in count0:
                    root_right += 1
            else:
                outputs[i] = count1[outputs1[i]]
                if targets[i].to(torch.float32) in count1:
                    root_right += 1
            root_count += 1
        num_ins += targets.size(0)
        correct += outputs.eq(targets.data).cpu().sum().item()
    print('\nrooting acc is:'+str(100*root_right/root_count)+'%')
    acc = 100.*correct/num_ins
    return acc

# ============== learn the leaf model and do the clustering ============
def learn_and_clustering(args, L, epoch, classes, test = True, cluster = True, save = False):
    required_train_loader = get_dataLoder(args, classes= classes, mode='Train', one_hot=True)
    # L = leaf(args,np.shape( classes)[0])
    param = list(L.parameters())
    optimizer, lr = get_optim(param, args, mode='preTrain', epoch=epoch)
    print('\n==> Epoch %d, LR=%.4f' % (epoch+1, lr))
    best_acc=0
    required_data = []
    required_targets = []
    for batch_idx, (inputs, targets) in enumerate(required_train_loader):
        if batch_idx>=args.num_test:
            break
        # targets = targets[:, sorted(list({}.fromkeys((torch.max(targets.data, 1)[1]).numpy()).keys()))]
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU setting
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets).long()
        outputs = L(inputs) # Forward Propagation
        loss = pred_loss(outputs,targets)
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        _, predicted = torch.max(outputs.data, 1)
        num_ins = targets.size(0)
        correct = predicted.eq((torch.max(targets.data, 1)[1])).cpu().sum()
        acc=100.*correct.item()/num_ins
        sys.stdout.write('\r')
        sys.stdout.write('Train Epoch [%3d/%3d] Iter [%3d/%3d]\t\t Loss: %.4f Accuracy: %.3f%%'
                         %(epoch+1, args.num_epochs_train, batch_idx+1, (pretrainloader.dataset.train_data.shape[0]//args.pretrain_batch_size)+1,
                           loss.item(), acc))
        sys.stdout.flush()
        #========================= saving the model ============================
        if epoch+1 == args.num_epochs_train and acc>best_acc and save:
            print('\nSaving the best leaf model...\t\t\tTop1 = %.2f%%' % (acc))
            save_point = cf.var_dir + args.dataset
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(L.state_dict(), save_point + '/L0.pkl')
            best_acc=acc
    #============================ valid training result ==================================
    if epoch+1 == args.num_epochs_train and test:
        required_valid_loader = get_dataLoder(args, classes= classes, mode='Valid', one_hot = True)
        num_ins = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(required_valid_loader):
            if cf.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            inputs, targets = Variable(inputs), Variable(targets).long()
            outputs = L(inputs)

            #======================= prepare data for clustering ============================
            if cluster:
                batch_required_data = outputs
                batch_required_targets = targets
                batch_required_data = batch_required_data.data.cpu().numpy() if cf.use_cuda else batch_required_data.data.numpy()
                batch_required_targets = batch_required_targets.data.cpu().numpy() if cf.use_cuda else batch_required_targets.data.numpy()
                required_data = stack_or_create(required_data, batch_required_data, axis=0)
                required_targets = stack_or_create(required_targets, batch_required_targets, axis=0)
            targets = torch.argmax(targets,1)
            _, predicted = torch.max(outputs.data, 1)
            num_ins += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        #============================ clustering ==================================
        if cluster:
            print('\n==> Doing the spectural clusturing')
            required_data = np.argmax(required_data, 1)
            required_targets = np.argmax(required_targets,1)
            F = function.confusion_matrix(required_data, required_targets)
            D = (1/2)*((np.identity(np.shape(classes)[0])-F)+np.transpose(np.identity(np.shape(classes)[0])-F))
            cluster_result = function.spectral_clustering(D, K=args.num_superclasses, gamma=10)
        acc = 100.*correct/num_ins
        print("\nValidation Epoch %d\t\tAccuracy: %.2f%%" % (epoch+1, acc))
        if cluster:
            return L, cluster_result, acc
        else:
            return L, _, acc
    else:
        return L, _ , _

# ============== based on all the Newsuperclass, generate the new model archetecture ============
def ExtendLeafModel(Leaf, Branch, Superclass, New_superclass):
    print('Leaf is not empty')
    for dict in New_superclass:
        if dict not in Superclass:
            Superclass[dict]=[]
        if New_superclass[dict] != Superclass[dict]:
            for epoch in range(args.num_epochs_train):
                if epoch == 0:
                    Model = prepared_model(np.shape(New_superclass[dict])[0], dict)
                    # Model = leaf(args,np.shape(New_superclass[dict])[0])
                    if cf.use_cuda:
                        Model.cuda()
                        cudnn.benchmark = True
                Leaf[dict], clusting_result, acc = learn_and_clustering(args, Model, epoch, New_superclass[dict], test = True, cluster = True)
            save_model(Leaf[dict], dict)
            Branch[dict]=Leaf[dict]
            Branch[dict] = train_branch(Branch[dict], clusting_result, New_superclass[dict])
            Branch_clusting_result[dict] = clusting_result
            for i in range (np.shape(clusting_result)[0]):
                if i == 0:
                    Superclass[dict+'0'] = []
                    Superclass[dict+'1'] = []
                if clusting_result[i] == 0:
                    Superclass[dict+'0'] += [New_superclass[dict][i]]
                if clusting_result[i] == 1:
                    Superclass[dict+'1'] += [New_superclass[dict][i]]
            for epoch in range(args.num_epochs_train):
                if not (Superclass[dict+'0'] and Superclass[dict+'0']): # dealing with if one superclass is empty
                    break
                if epoch == 0:
                    Model0 = prepared_model(np.shape( Superclass[dict+'0'])[0], dict)
                    Model1 = prepared_model(np.shape( Superclass[dict+'1'])[0], dict)
                    if cf.use_cuda:
                        Model0.cuda()
                        Model1.cuda()
                        cudnn.benchmark = True
                Leaf[dict+'0'] ,_ , _ = learn_and_clustering(args, Model0, epoch, Superclass[dict+'0'], test = True, cluster = False)
                Leaf[dict+'1'] ,_ , _ = learn_and_clustering(args, Model1, epoch, Superclass[dict+'1'], test = True, cluster = False)
            if (Superclass[dict+'0'] and Superclass[dict+'0']):
                save_model(Leaf[dict+'0'], dict+'0')
                save_model(Leaf[dict+'1'], dict+'1')
            if not (Superclass[dict+'0'] and Superclass[dict+'0']): # dealing with if one superclass is empty
                branch_acc = acc
            else:
                branch_acc = branch_accuracy(Branch[dict], Leaf[dict+'0'], Leaf[dict+'1'], Branch_clusting_result[dict], New_superclass[dict])
            print('\nTwo small leaf with branch acc: '+str(branch_acc)+'% and Huge Leaf acc: '+str(acc) +'%')
            if branch_acc > acc:
                save_model(Branch[dict], dict, name = 'Branch')
                Leaf.pop(dict,'dict dot found in dictionary Leaf')
                Superclass.pop(dict,'dict dot found in dictionary Superclass')
                #New_superclass.pop(dict,'dict dot found in dictionary New_superclass')
                print('\n---- New Branch:', dict, 'is created, small Leaf 0 and 1 is created ----')
            else:
                Branch_clusting_result.pop(dict,'dict dot found in dictionary Branch_clusting_result')
                Branch.pop(dict ,'dict dot found in dictionary Branch')
                Leaf.pop(dict+'0','dict dot found in dictionary Leaf')
                Leaf.pop(dict+'1','dict dot found in dictionary Leaf')
                Superclass.pop(dict+'0','dict dot found in dictionary Superclass')
                Superclass.pop(dict+'1','dict dot found in dictionary Superclass')
                Superclass.pop(dict,'dict dot found in dictionary Superclass')
                Superclass[dict]= New_superclass[dict]
                print('\n------------------ Huge Leaf:', dict, 'is created ------------------')
        else:
            print('\nNo new classes is classified to this branch QAQ')
        print('\n============== One time incremental finished ==============')
    return Leaf, Branch, Superclass, Branch_clusting_result


# ============== calculate the newsuperclass ============
def embranchment(Branch, Branch_clusting_result, dict, Superclass, New_superclass, Class_added):
    print('Sort the new classes by using exist branches')
    required_train_loader = get_dataLoder(args, classes= Class_added, mode='Train', one_hot=False)
    required_data = []
    required_targets = []
    for batch_idx, (inputs, targets) in enumerate(required_train_loader):
        if batch_idx>=args.num_test:
            break
        #targets = targets[:, sorted(list({}.fromkeys((torch.max(targets.data, 1)[1]).numpy()).keys()))]
        if cf.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU setting
            Branch[dict].cuda
            cudnn.benchmark = True
        inputs, targets = Variable(inputs), Variable(targets).long()
        outputs = Branch[dict](inputs) # Forward Propagation
        _, predicted = torch.max(outputs.data, 1)

        #======================= prepare data for confusion matrix Φ ============================
        batch_required_data = torch.argmax(outputs,1)
        batch_required_targets = targets
        batch_required_data = batch_required_data.data.cpu().numpy() if cf.use_cuda else batch_required_data.data.numpy()
        batch_required_targets = batch_required_targets.data.cpu().numpy() if cf.use_cuda else batch_required_targets.data.numpy()
        required_data = stack_or_create(required_data, batch_required_data, axis=1)
        required_targets = stack_or_create(required_targets, batch_required_targets, axis=1)
    #======================= calculate the confusion matrix Φ ============================
    class0 = {}
    class1 = {}

    for Class in Class_added:
        class0[Class] = 0
        class1[Class] = 0
        for j in range(len(required_targets)):
            if Branch_clusting_result[dict][required_data[j]]==0 and required_targets[j]==Class:
                class0[Class] += 1
            elif Branch_clusting_result[dict][required_data[j]]==1 and required_targets[j]==Class:
                class1[Class] += 1
    zero=[]
    one=[]
    for i in class0:
        if class0[i] >= class1[i]:
            zero += [i]
        else:
            one += [i]
    New_superclass.pop(dict,'dict dot found in dictionary New_superclass')
    if dict+'0' in Superclass:
        New_superclass[dict+'0'] = Superclass[dict+'0'] + zero
    else:
        New_superclass[dict+'0'] = zero
    if dict+'1' in Superclass:
        New_superclass[dict+'1'] = Superclass[dict+'1'] + one
    else:
        New_superclass[dict+'1'] = one
    return New_superclass


if args.resume_model:
    print('\n=========== Begin loading the previous model ===========')
    Superclass, Branch, Branch_clusting_result, Leaf = save_and_load(Superclass, Branch, Branch_clusting_result, Leaf, load = True)
    print('\n=========== Finish loading the previous model ===========')
    print('\n============== Construct the incremental learning model ==============')
    for newclasses in Class_generator:
        # =========== load to the previous training step ===========
        continues = 'false'
        for Class in Class_generator[newclasses]:
            for dict in Superclass:
                if Class_generator[newclasses][Class_generator[newclasses].shape[0]-1] in Superclass[dict]:
                    continues = 'true'
        if continues == 'true':
            continue
        # =========== let the new class added pass all the branch(rooting) to the final superclass ===========
        if Branch:
            print('Branch is not empty')
            New_superclass = {}
            for branch_dict in Branch:
                if len(branch_dict) == 1:
                    New_superclass = embranchment(Branch, Branch_clusting_result, branch_dict, Superclass, New_superclass, sorted(Class_generator[newclasses]))
                    # dict: dict for that branch;  sorted(Class_generator[newclasses]: new class added and need to be sorted
                else:
                    New_superclass.update(embranchment(Branch, Branch_clusting_result, branch_dict, Superclass, New_superclass, sorted(New_superclass[branch_dict])))
            Leaf, Branch, Superclass, Branch_clusting_result = ExtendLeafModel(Leaf, Branch, Superclass, New_superclass)
        else:
            print('Branch is empty')
            if Leaf:
                New_superclass['0'] = np.append(Superclass['0'], Class_generator[newclasses])
            else:
                New_superclass['0'] = Class_generator[newclasses]

            # =================== based on the newsuperclass, iterate the model =================
            Leaf, Branch, Superclass, Branch_clusting_result = ExtendLeafModel(Leaf, Branch, Superclass, New_superclass)
        print('\n============== Ten new classes added to model ==============\n')
        print(Superclass)
        print('\n================= end of superclass for now ================')
        save_and_load(Superclass, Branch, Branch_clusting_result, Leaf, num=(newclasses+1)*10)
    print('\n================= Model construction finish ================')

#================ test the model ================( I have not test it yet QAQ)
if args.resume_model:
    print('\n=========== Begin loading the previous model ===========')
    Superclass, Branch, Branch_clusting_result, Leaf = save_and_load(Superclass, Branch, Branch_clusting_result, Leaf, load = True)
    print('\n=========== Finish loading the previous model ===========')
    num_ins_total = []
    correct_total = []
    for c in range(100):
        num_ins = []
        correct = []
        for dict in Superclass:
            if c in Superclass[dict]:
                dict_id = dict
        required_train_loader = get_dataLoder(args, classes=[c], mode='Test', one_hot=False)
        for batch_idx, (inputs, targets) in enumerate(required_train_loader):
            # need to get the output in this level
            for count in range(len(dict_id)-1):
                dict=dict_id[:count+1]
                outs_hot = Branch[dict](inputs)
                outputs = torch.argmax(outs_hot, 1)
                if count==0:
                    num_ins_total += outputs.size(0)
                    num_ins += outputs.size(0)
                num = outputs.size(0)
                if count+1 < len(dict_id)-1:
                    select=[]
                    for i in range (num):
                        if dict+str(Branch_clusting_result[dict][outputs.data.cpu().numpy()[i]]) in dict_id:
                            select += [i]
                    inputs = inputs[select,:,:,:]
                elif count+1 == len(dict_id)-1:
                    final_outs = Leaf[dict_id](inputs)
                    final_outs = np.argmax(final_outs.data.cpu().numpy(),1)
                    for j in range(final_outs.size(0)):
                        if Superclass[dict_id][final_outs[j]] == targets.data.cpu().numpy()[j]:
                            correct_total += 1
                            correct += 1
                else:
                    print('error !!!!!!!!!!!!!!!!!!!')
        acc = 100.*correct_total/num_ins_total
        print('\nThe accuracy for the test class:'+str(c)+' is:'+acc+'%')
    acc_total = 100.*correct_total/num_ins_total
    print('\n===== The total accuracy for 100 test classes is:'+acc+'% =====')
    print('\n===================== Test process finish =====================')
