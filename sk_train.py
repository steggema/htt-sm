## Ad-hoc tau ID training with sklearn using ROOT trees as input
# Requires root_numpy https://github.com/rootpy/root_numpy
# Jan Steggemann 27 Aug 2015

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# from sklearn.cross_validation import train_test_split #cross_val_score
from sklearn.cross_validation import KFold

from sklearn.metrics import roc_curve

# For model I/O
from sklearn.externals import joblib

from root_numpy import root2array, root2rec
def trainVars():
    return [
        'mt', 'l2_mt', 'n_jets', 'met_pt', 'pthiggs', 'vbf_mjj', 'vbf_deta', 'vbf_n_central', 'l2_pt', 'l1_pt','mvis', 'l1_eta', 'l2_eta', 'delta_phi_l1_l2', 'delta_eta_l1_l2', 'pt_l1l2', 'delta_phi_j1_met', 'pzeta_disc', 'jet1_pt', 'jet1_eta'
    ]

files_vbf = [
    'data/inclusive_HiggsVBF125.root',
    'data/inclusive_HiggsGGH125.root'
]

files_ztt = [
    'data/inclusive_ZTT.root', 
]

files_signal = files_vbf + files_ztt

files_bg = [
'data/inclusive_TBarToLeptons_tch_powheg.root',
'data/inclusive_TToLeptons_tch_powheg.root',
'data/inclusive_TBar_tWch.root',
'data/inclusive_T_tWch.root',
'data/inclusive_TT.root',
'data/inclusive_VVTo2L2Nu.root',
'data/inclusive_WWTo1L1Nu2Q.root',
'data/inclusive_WZTo1L1Nu2Q.root',
'data/inclusive_WZTo1L3Nu.root',
'data/inclusive_WZTo2L2Q.root',
# 'data/inclusive_WZTo3L.root',
'data/inclusive_W.root',
# 'data/inclusive_ZJM10.root',
'data/inclusive_ZJ.root',
# 'data/inclusive_ZLM10.root',
'data/inclusive_ZL.root',
# 'data/inclusive_ZTTM10.root',
# 'data/inclusive_ZTT.root',
'data/inclusive_ZZTo2L2Q.root',
'data/inclusive_ZZTo4L.root',
'data/inclusive_QCD.root',
# 'data/inclusive_data_obs.root',
]

selection = 'vbf_mjj>500. && abs(vbf_deta)>3.5 '
selection = 'n_jets>0.5 && !(vbf_mjj>500. && abs(vbf_deta)>3.5)'
selection = 'n_jets < 0.5'
selection = '1.'

def createGBRT(learning_rate=0.01, max_depth=4, n_estimators=1000, subSample=0.5):
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=1, loss='deviance', verbose=1, subsample=subSample, max_features=0.5) #loss='exponential'/'deviance'
     # loss='deviance', verbose=1, subsample=subSample)
    return clf


def train(clf, training_data, target, weights, set_neg_to_zero=True):
    print clf

    sumWeightsSignal = np.sum(weights[np.where(target == 1)])
    sumWeightsBackground = sum(weights[np.where(target == 0)])

    print 'Sum weights signal', sumWeightsSignal
    print 'Sum weights background', sumWeightsBackground

    aveWeightSignal = sumWeightsSignal/np.sum(target)
    print 'Average weight signal', aveWeightSignal
    aveWeightBG = sumWeightsSignal/np.sum(1-target)
    print 'Average weight background', aveWeightBG

    
    nCrossVal = 2
    # kf = KFold(len(training_data), nCrossVal, shuffle=True, random_state=1)

    train_indices = np.where((training_data[:, 10]*1000.).astype(int)%2==1)
    test_indices = np.where((training_data[:, 10]*1000.).astype(int)%2==0)

    kf = [(train_indices, test_indices), (test_indices, train_indices)]

    # trainIndices = training_data[abs(int(training_data[10]*1000.))%2 == 0]
    # testIndices = training_data[abs(int(training_data[10]*1000.))%2 == 1]


    print 'Cross-validation:', nCrossVal, 'folds'

    for i_fold, (trainIndices, testIndices) in enumerate(kf):
        print 'Starting fold'

        d_train = training_data[trainIndices]
        d_test = training_data[testIndices]

        t_train = target[trainIndices]
        t_test = target[testIndices]

        w_train = weights[trainIndices]
        w_test = weights[testIndices]

        if set_neg_to_zero:
            # import pdb; pdb.set_trace()
            # w_train = np.apply_along_axis(lambda x: x if x > 0. else 0., 1, w_train)
            w_train = (w_train>0) * w_train

        # del training_data, target, weights, trainIndices, testIndices, kf

        clf.fit(d_train, t_train, w_train)

        print 'Produce scores'
        # scores = clf.decision_function(d_test)
        scores = clf.predict_proba(d_test)

        # import pdb; pdb.set_trace()

        effs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        print 'ZTT vs background'

        fpr, tpr, tresholds = roc_curve(t_test[np.where(t_test > 0)], scores[np.where(t_test > 0)][:,1:2], sample_weight=w_test[np.where(t_test > 0)], pos_label=1)

        for eff in effs:
            print 'Fake rate at signal eff', eff, fpr[np.argmax(tpr>eff)]

        print 'VBF+ggH vs background'

        fpr, tpr, tresholds = roc_curve(t_test[np.where(t_test != 1)], scores[np.where(t_test != 1)][:,0:1], sample_weight=w_test[np.where(t_test != 1)], pos_label=0)

        for eff in effs:
            print 'Fake rate at signal eff', eff, fpr[np.argmax(tpr>eff)]

        print 'VBF+ggh vs ZTT'

        fpr, tpr, tresholds = roc_curve(t_test[np.where(t_test != 2)], scores[np.where(t_test != 2)][:,0:1], 
            sample_weight=w_test[np.where(t_test != 2)], pos_label=0)

        for eff in effs:
            print 'Fake rate at signal eff', eff, fpr[np.argmax(tpr>eff)]

        # Classic ROC curve

        # fpr, tpr, tresholds = roc_curve(t_test, scores, sample_weight=w_test)
        # joblib.dump((fpr, tpr, tresholds), 'roc_vals.pkl')
       

        # for eff in effs:
        #     print 'Fake rate at signal eff', eff, fpr[np.argmax(tpr>eff)]
    
        # Can save with different features if necessary
        # joblib.dump(clf, 'train/{name}_clf_{i_fold}_vbf.pkl'.format(name=clf.__class__.__name__, i_fold=i_fold), compress=9)
        # joblib.dump(clf, 'train/{name}_clf_{i_fold}_1jet.pkl'.format(name=clf.__class__.__name__, i_fold=i_fold), compress=9)
        joblib.dump(clf, 'train/{name}_clf_{i_fold}_0jet.pkl'.format(name=clf.__class__.__name__, i_fold=i_fold), compress=9)

    # if doCrossVal:
    print 'Feature importances:'
    print clf.feature_importances_

    varList = trainVars()
    for i, imp in enumerate(clf.feature_importances_):
        print imp, varList[i] if i<len(varList) else 'N/A'
    
    return clf


def readFiles():
    print 'Reading files...'

    # weightsS = root2rec(files_signal, treename='tree', branches=['full_weight'], selection=selection)
    weights_vbf = root2rec(files_vbf, treename='tree', branches=['full_weight'], selection=selection)['full_weight']
    weights_ztt = root2rec(files_ztt, treename='tree', branches=['full_weight'], selection=selection)['full_weight']
    weightsB = root2rec(files_bg, treename='tree', branches=['full_weight'], selection=selection)['full_weight']

    sum_weights_vbf = np.sum(weights_vbf)
    sum_weights_ztt = np.sum(weights_ztt)
    sum_weightsB = np.sum(weightsB)

    weights_ztt = weights_ztt * sum_weights_vbf/sum_weights_ztt
    weightsB = weightsB * sum_weights_vbf/sum_weightsB

    # nS = len(weightsS)
    n_vbf = len(weights_vbf)
    n_ztt = len(weights_ztt)
    nB = len(weightsB)

    # fullWeight = np.concatenate((weightsS, weightsB))
    fullWeight = np.concatenate((weights_vbf, weights_ztt, weightsB))
    # fullWeight = fullWeight['full_weight']

    # fullWeight = np.ones(len(fullWeight))

    # del weightsS, weightsB

    # arrSB = root2array(files_signal + files_bg, treename='tree', branches=trainVars(), selection=selection)
    arrSB = root2array(files_vbf + files_ztt + files_bg, treename='tree', branches=trainVars(), selection=selection)

    # Need a matrix-like array instead of a 1-D array of lists for sklearn
    arrSB = (np.asarray([arrSB[var] for var in trainVars()])).transpose()

    # targets = np.concatenate((np.ones(nS),np.zeros(nB)))
    targets = np.concatenate((np.ones(n_vbf)*2, np.ones(n_ztt),np.zeros(nB)))

    print 'Done reading files.'

    return arrSB, fullWeight, targets


if __name__ == '__main__':

    classifier = 'GBRT' # 'Ada' #'GBRT'
    doTrain = True

    print 'Read training and test files...'
    training, weights, targets = readFiles()        

    print 'Sizes'
    print training.nbytes, weights.nbytes, targets.nbytes

    if doTrain:
        print 'Start training'

        if classifier == 'GBRT':
            clf = createGBRT()
            train(clf, training, targets, weights)
