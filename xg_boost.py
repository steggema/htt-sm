## Ad-hoc tau ID training with sklearn using ROOT trees as input
# Requires root_numpy https://github.com/rootpy/root_numpy
# Jan Steggemann 27 Aug 2015
import numpy as np
import time

import xgboost as xgb
from root_numpy import root2array, root2rec


def trainVars():
    return [
    'mt',
    'n_jets',
    'met_pt',
    'pthiggs',
    'vbf_mjj',
    'vbf_deta',
    'vbf_n_central',
    'l2_pt',
    'l1_pt',
    'svfit_transverse_mass',
    'delta_phi_l1_l2',
    'delta_eta_l1_l2',
    'svfit_mass'
    ]

files_sig = [
    'data/inclusive_HiggsVBF125_weight.root',
    'data/inclusive_HiggsGGH125_weight.root'
]

files_ZTT = [
    # 'data/inclusive_HiggsGGH125_weight.root'
    'data/inclusive_ZTT_weight.root',
    'data/inclusive_ZTTM10_weight.root',
]

#files_signal = files_sig + files_ZTT

files_bg = [
'data/inclusive_TBarToLeptons_tch_powheg_weight.root',
'data/inclusive_TBar_tWch_weight.root',
'data/inclusive_TT_weight.root',
'data/inclusive_TToLeptons_tch_powheg_weight.root',
'data/inclusive_T_tWch_weight.root',
'data/inclusive_VVTo2L2Nu_weight.root',
'data/inclusive_W1Jets_weight.root',
'data/inclusive_W2Jets_weight.root',
'data/inclusive_W3Jets_weight.root',
'data/inclusive_W4Jets_weight.root',
'data/inclusive_WWTo1L1Nu2Q_weight.root',
'data/inclusive_WZTo1L1Nu2Q_weight.root',
'data/inclusive_WZTo1L3Nu_weight.root',
'data/inclusive_WZTo2L2Q_weight.root',
'data/inclusive_WZTo3L_weight.root',
'data/inclusive_W_weight.root',
'data/inclusive_ZJM10_weight.root',
'data/inclusive_ZJ_weight.root',
'data/inclusive_ZLM10_weight.root',
'data/inclusive_ZL_weight.root',
# 'data/inclusive_ZTTM10_weight.root',
# 'data/inclusive_ZTT_weight.root',
'data/inclusive_ZZTo2L2Q_weight.root',
'data/inclusive_ZZTo4L_weight.root',
# 'data/inclusive_data_obs_weight.root',
]

vals = np.zeros(4)

#print 'How many classes would you like?'
#num = input("Enter a value between 2 and 4 (inclusive): ")
num = 2

if num == 2:
    vals = [1, 0, 0, 0]

elif num == 3:
    vals = [2, 1, 0, 0]

elif num == 4:
    vals = [3, 2, 1, 0]

else:
    print 'Invalid entry. Please try again.'
    exit()

with open("XG_BoostTest1.1", "w") as text_file:
    text_file.write("XGBoost Optimization w/ 1000 Trees Unweighted\n\n")
    text_file.write("Learn Rate\tMax Depth\tN_Est.\tBest Cut\tBest S\tBest B\tMax AMS\n\n")

#selection = 'vbf_mjj > 400. && vbf_deta > 3.0'
#selection = '!(vbf_mjj > 400. && vbf_deta > 3.0)'
selection = '1'

def startClock():
    global startTime
    startTime=time.time()

def timer():
    
    RunTime=time.time()-startTime
        
    if RunTime >= 5:
        raise StopTraining()

    return



def sigmoid(x):
  return 1./(1. + np.exp(-x))

def AMS(s, b, b_r=10.):
    return np.sqrt(2.0*((s+b+b_r)*np.log(1.0+s/(b+b_r))-s))

def logregobj(preds, dtrain):
    y = dtrain.get_label()
    weight = dtrain.get_weight()
    # yhat = sigmoid(preds)

    max_ams = 0.
    best_cutoff = 0.
    for cutoff in np.arange(0., 0.1, 0.005):
        yhat = preds > cutoff    # approx ams using soft probability instead of hard predicted class
        # import pdb; pdb.set_trace()
        s = np.sum( weight * y * yhat )
        b = np.sum( weight * (1.-y) * yhat )
        # negative it since in xgboost, we minimize the loss
        ams = AMS(s, b)
        if ams > max_ams:
            max_ams = ams
            best_cutoff = cutoff
    # bReg = 10.0
    # tmp = 1.0 + s/(b + bReg)
    # # ds / dyhat
    # ds = np.log(tmp)/ams
    # # db / dyhat
    # db = (np.log(tmp) + 1 - tmp)/ams
    # # d(-ams) / dyhat
    # #grad = -(ds*(yhat-y)+db*(1-(yhat-y)))# @phunter's version
    # grad = (ds * y + db * (1.-y)) * weight * yhat * (1.-yhat) # @George's version
    # hess = np.ones(yhat.shape)/(10**3) #constant
    # import pdb; pdb.set_trace()
    return 'max_ams@{c}'.format(c=best_cutoff), max_ams #, 1.#grad, hess[0]

def trainXGB(data, target, weight, UNweight, ntrees=1000, extraW=1.):

    nCrossVal = 2
    # kf = KFold(len(training_data), nCrossVal, shuffle=True, random_state=1)

    train_indices = np.where((data[:, 10]*1000.).astype(int)%2==1)
    test_indices = np.where((data[:, 10]*1000.).astype(int)%2==0)
    
    kf = [(train_indices, test_indices), (test_indices, train_indices)]
    
    # trainIndices = training_data[abs(int(training_data[10]*1000.))%2 == 0]
    # testIndices = training_data[abs(int(training_data[10]*1000.))%2 == 1]
    
    print 'Cross-validation:', nCrossVal, 'folds'
    
    for i_fold, (trainIndices, testIndices) in enumerate(kf):
        
        startClock()
        
        print 'Starting fold'
        
        staged_maxAMS = []
        staged_bestS = []
        staged_bestB = []
        staged_bestCutoffs = []
        staged_nTrees = []

        d_train = data[trainIndices]
        d_test = data[testIndices]
        
        t_train = target[trainIndices]
        global t_test
        t_test = target[testIndices]
        
        w_train = weight[trainIndices]
        w_test = weight[testIndices]
        
        w_trainUN = UNweights[trainIndices]
        global w_testUN
        w_testUN = UNweights[testIndices]

        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        xgmat = xgb.DMatrix(d_train, label=t_train, missing=-999.0, weight=w_trainUN)
        xgmat_test = xgb.DMatrix(d_test, label=t_test, missing=-999.0, weight=w_testUN)

        # setup parameters for xgboost
        param = {}
        # use logistic regression loss, use raw prediction before logistic transformation
        # since we only need the rank
        param['objective'] = 'binary:logistic'
        # scale weight of positive examples
        param['scale_pos_weight'] = extraW
        param['eta'] = eta
        param['max_depth'] = max_depth
        param['eval_metric'] = 'auc' # You may be able to use ams here
        param['silent'] = 1
        param['nthread'] = 1
        param['subsample'] = 0.5
        # param['colsample_bytree'] = 0.5
        param['min_child_weight'] = 0.1

        # you can directly throw param in, though we want to watch multiple metrics here
        plst = list(param.items())#+[('eval_metric', 'ams@0.145')]

        test_mat = xgb.DMatrix(d_test, missing=-999.0)
        watchlist = [(xgmat, 'train'), (xgmat_test, 'test')]
        # boost 120 tres
        # num_round = 10

        bst = xgb.train(plst, xgmat, ntrees, watchlist, feval=logregobj)
        
        scores = [bst.predict(test_mat)]
        
        for i in range(0,ntrees):
            if i%100 == 0:
                bst.update(xgmat, i)
                scores = [bst.predict(test_mat)]
                best_cutoff, best_S, best_B, max_AMS = calculateAMS(scores)
                
                print max_AMS
                staged_bestCutoffs.append(best_cutoff)
                staged_bestB.append(best_B)
                staged_bestS.append(best_S)
                staged_maxAMS.append(max_AMS)

                staged_nTrees.append(i+1)
        
        # save out model
        bst.save_model('higgs.model')
        
        max_AMS = max(staged_maxAMS)
        argmax_AMS = staged_maxAMS.index(max_AMS)
        best_cutoff = staged_bestCutoffs[argmax_AMS]
        best_n_for_AMS = staged_nTrees[argmax_AMS]
        best_B = staged_bestB[argmax_AMS]
        best_S = staged_bestS[argmax_AMS]
        
        print 'BEST N FOR AMS', best_n_for_AMS
        print 'BEST CUTOFF ', best_cutoff
        print 'MAX AMS ', max_AMS, '\n\n'

        with open("XG_BoostTest1.1", "a") as text_file:
            text_file.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (eta, max_depth, best_n_for_AMS, best_cutoff, best_S,best_B, max_AMS))

    return bst



def calculateAMS(scores):

    n_cutoffs=100
    cutoffs=np.arange(n_cutoffs)
    AMS=np.zeros(n_cutoffs)
    S=np.zeros(n_cutoffs)
    B=np.zeros(n_cutoffs)
    cutoffs=np.multiply(cutoffs,1/float(n_cutoffs))
    j=0
        
    for cutoff in cutoffs:
        is_htt = np.zeros(len(t_test))
        is_htt[np.where(scores[0]>cutoff)]=1
        
        temp1=np.multiply(w_testUN, t_test)
        s=np.multiply(is_htt,temp1)
            
        temp2=np.multiply(w_testUN, (1-t_test))
        b=np.multiply(is_htt,temp2)
        
        S[j]=np.sum(s)
        B[j]=np.sum(b)
        AMS[j]=np.sqrt(2.0*((S[j]+B[j]+10.0)*np.log(1.0+S[j]/(B[j]+10.0))-S[j]))
        
        j=j+1
        
    best_cutoff=cutoffs[AMS.argmax()]
    best_S=S[AMS.argmax()]
    best_B=B[AMS.argmax()]
    max_AMS=np.amax(AMS)

    return best_cutoff, best_S, best_B, max_AMS

def readFiles():
    print 'Reading files...'

    # weightsS = root2rec(files_signal, treename='tree', branches=['weight'], selection=selection)
    weights_sig = root2rec(files_sig, treename='tree', branches=['full_weight'], selection=selection)['full_weight']
    weights_ZTT = root2rec(files_ZTT, treename='tree', branches=['full_weight'], selection=selection)['full_weight']
    weightsB = root2rec(files_bg, treename='tree', branches=['full_weight'], selection=selection)['full_weight']

    sum_weights_sig = np.sum(weights_sig)
    sum_weights_ZTT = np.sum(weights_ZTT)
    sum_weightsB = np.sum(weightsB)

    normWeights_ZTT = weights_ZTT * sum_weights_sig/sum_weights_ZTT
    normWeightsB = weightsB * sum_weights_sig/sum_weightsB

    # nS = len(weightsS)
    n_sig = len(weights_sig)
    n_ZTT = len(weights_ZTT)
    nB = len(weightsB)
    
    # fullWeight = np.concatenate((weightsS, weightsB))
    fullWeight = np.concatenate((weights_sig, normWeights_ZTT, normWeightsB))
    unNormFullWeight = np.concatenate((weights_sig, weights_ZTT, weightsB))
    # fullWeight = fullWeight['weight']
    
    # fullWeight = np.ones(len(fullWeight))
    
    # del weightsS, weightsB
    
    # arrSB = root2array(files_signal + files_bg, treename='tree', branches=trainVars(), selection=selection)
    arrSB = root2array(files_sig + files_ZTT + files_bg, treename='tree', branches=trainVars(), selection=selection)
    
    # Need a matrix-like array instead of a 1-D array of lists for sklearn
    arrSB = (np.asarray([arrSB[var] for var in trainVars()])).transpose()
    
    # targets = np.concatenate((np.ones(nS),np.zeros(nB)))
    # targets = np.concatenate((np.ones(n_sig)*2, np.ones(n_ZTT),np.zeros(nB)))
    targets = np.concatenate((np.ones(n_sig)*vals[0], np.ones(n_ZTT)*vals[1], np.ones(nB)*vals[2]))
    
    print 'Done reading files.'
    
    #import pdb; pdb.set_trace()
    
    return arrSB, fullWeight, unNormFullWeight, targets

etas = [0.025, 0.05, 0.1]
max_depths = [4,6,8]

global eta
global max_depth

for i in etas:
    for j in max_depths:
        
        eta=i
        max_depth=j
        
        if __name__ == '__main__':

            classifier = 'GBRT' # 'Ada' #'GBRT'
            doTrain = True

            print 'Read training and test files...'
            training, weights, UNweights, targets = readFiles()

            print 'Sizes'
            print training.nbytes, weights.nbytes, targets.nbytes

            if doTrain:
                print 'Start training'

                if classifier == 'GBRT':
                    trainXGB(training, targets, weights, UNweights)

