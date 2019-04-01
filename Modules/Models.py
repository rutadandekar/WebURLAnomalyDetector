import math
import pickle
import numpy as np
from Modules import Utils

class AttributeLengthModel:
    def __init__(self):
        self.name = "Attribute Length Model"
        # unset initial parameters
        self.sigma_by_threshold, self.threshold_sqrt, self.mean = 0.0,0.0,0.0
        self.variance, self.atr_lengths = 0.0,0.0
        self.prob_thresh = 0.5

    def learning(self, dataset, limit=None):
        # training requires only positive data attributes
        data = dataset.get_attribute_dictionary(positive_only=True)
        # Sort by attribute occourance
        data_sorted = sorted(data, key=lambda atr: len(data[atr]), reverse=True)
        # find mean, variance, and thresholds
        self.mean = {}
        self.variance = {}
        self.atr_lengths = []
        for i,k in enumerate(data_sorted[:limit]):
            if len(data[k])==0:
                continue
            self.mean[k] = sum([len(v) for v in data[k]])/float(len(data[k]))
            self.variance[k] =sum([((len(v)-self.mean[k])**2) for v in data[k]])/float((len(data[k])-1)+1e-9)
            self.atr_lengths.append({k:[float(len(v)) for v in data[k]]})
        # calcuate thesholds
        thresholds, sigma_by_threshold, threshold_sqrt = {}, {}, {}
        for dictionary in self.atr_lengths:
            for key in dictionary:
                if key in self.mean:
                    for v in dictionary[key]:
                        thresholds[key] = round(((v-self.mean[key])**2),5)
        for t,v in zip(thresholds,self.variance):
            threshold_sqrt[t] = math.sqrt(thresholds[t])
            if thresholds[t] == 0: sigma_by_threshold[t] = 0.0
            else: sigma_by_threshold[t] = round((self.variance[v]**2)/thresholds[t])
        self.sigma_by_threshold, self.threshold_sqrt = sigma_by_threshold, threshold_sqrt

    def save_parameters(self,filename):
        model_parameters = [self.sigma_by_threshold, self.threshold_sqrt, self.mean]
        filehandler = open(filename,"wb")
        pickle.dump(model_parameters,filehandler)

    def load_parameters(self,filename):
        filehandler = open(filename,"rb")
        model_parameters = pickle.load(filehandler)
        self.sigma_by_threshold,self.threshold_sqrt, self.mean = model_parameters[0],model_parameters[1], model_parameters[2]

    def detection(self, dataset):
        predictions = []
        accuracy = -1
        for data in dataset.data:
            data_atrs = data[0]
            _p = []
            for atr in data_atrs:
                # Make sure key is in all lists
                if not all(atr in l for l in (self.mean,self.threshold_sqrt)):
                    continue
                # Current atrribute length
                atr_l = float(len(data_atrs[atr]))
                # Chebyshev inequality from paper
                p = 1.0 if abs(atr_l-self.mean[atr])>self.threshold_sqrt[atr] else 0.0
                _p.append(p)
            predictions.append(0 if len(_p)==0 else sum(_p)/len(_p))
        return predictions

class AttributeCharachterDistributionModel:
    def __init__(self,bins=[1,1,2,6,7,238]):
        self.name = "Attribute Charachter Distribution Model"
        self.bins = bins

    def learning(self, dataset):
        # training requires only positive data attributes
        data = dataset.get_attribute_dictionary(positive_only=True)
        attribute_icd_bins = {}
        attribute_ascii_distribution = []
        for atr in data:
            for value in data[atr]:
                if value=='':
                    continue
                list_of_asciis = [0.0]*255
                for o in value:
                    list_of_asciis[ord(o)] += 1
                # Relative frequencies of each character
                try:
                    np.array(list_of_asciis)/sum(list_of_asciis)
                except FloatingPointError:
                    print list_of_asciis,sum(list_of_asciis)
                list_of_asciis = np.sort(np.array(list_of_asciis)/sum(list_of_asciis))[::-1]
                attribute_ascii_distribution.append(list_of_asciis)
            if not attribute_ascii_distribution:
                continue
            mrfd = np.mean(attribute_ascii_distribution,axis = 0)
            # ICD of attribute
            mrfd_idx = 0
            icd = []
            for bin in self.bins:
                icd.append(sum(mrfd[mrfd_idx:mrfd_idx+bin]))
                mrfd_idx += bin
            attribute_icd_bins[atr] = icd
        self.expected_values_icd_bins = attribute_icd_bins

    def save_parameters(self,filename):
        model_parameters = self.expected_values_icd_bins
        filehandler = open(filename,"w")
        pickle.dump(model_parameters,filehandler)

    def load_parameters(self,filename):
        filehandler = open(filename,"r")
        self.expected_values_icd_bins = pickle.load(filehandler)

    def detection(self, dataset):
        predictions = []
        number_of_urls = 1
        for data in dataset.data:
            data_atrs = data[0]
            x_square,x_val  = [], []
            for atr in data_atrs:
                if data_atrs[atr]=='':
                    continue
                list_of_ascii_values = [0.0]*255
                for o in data_atrs[atr]:
                    list_of_ascii_values[ord(o)] += 1
                list_of_ascii_values = np.sort(np.array(list_of_ascii_values)/sum(list_of_ascii_values))[::-1]
                mrfd_idx = 0
                observed_values_icd_bins = []
                for bin in self.bins:
                    observed_values_icd_bins.append(sum(list_of_ascii_values[mrfd_idx:mrfd_idx+bin]))
                    mrfd_idx += bin
                obs = np.array(observed_values_icd_bins)
                if atr in self.expected_values_icd_bins:
                    exp = (np.array(self.expected_values_icd_bins[atr])*len(data_atrs[atr])) + 1e-9
                    diff = (obs - exp)
                    sq = diff ** 2
                    div = sq / exp
                    x_square.append(np.nansum(div))
                    if np.nansum(div) <= 9.2363:
                        x_val.append(0.0)
                    else:
                        x_val.append(1.0)
            predictions.append(0 if len(x_val)==0 else sum(x_val)/len(x_val))
        return predictions

class ArrtributeStructuralInference:
    def __init__(self):
        self.name = "Attribute Structural Inference Model"
        self.attribute_markov_models = {}

    def learning(self, dataset):
        # training requires only positive data attributes
        data = dataset.get_attribute_dictionary(positive_only=True)
        for atr in data:
            self.attribute_markov_models[atr] = self.compute_markov_model(data[atr])

    def compute_markov_model(self,values):
        #Example usage values = ['20','202','203','204']
        transition_probabilities = {}
        for value in values:
            value_w_start_end = ['Start']+list(value)+['End']
            # value = ['Start','2','0','End']
            for idx,char in enumerate(value_w_start_end):
                # idx=0, char='Start'
                if not char in transition_probabilities:
                    # transition_probabilities['Start'] = {}
                    transition_probabilities[char] = {}
                if char==value_w_start_end[-1]: continue
                if not value_w_start_end[idx+1] in transition_probabilities[char]:
                    # transition_probabilities['Start']['2'] = 1
                    transition_probabilities[char][value_w_start_end[idx+1]] = 1
                else:
                    # transition_probabilities['Start']['2'] += 1
                    transition_probabilities[char][value_w_start_end[idx+1]] += 1
        for char in transition_probabilities:
            next_char_sum = float(sum(transition_probabilities[char].values()))
            for next_char in transition_probabilities[char]:
                transition_probabilities[char][next_char] = transition_probabilities[char][next_char]/next_char_sum
        emmision_probabilities = {}
        for char in transition_probabilities:
            emmision_probabilities[char] = {}
            n_transitions = len(transition_probabilities[char].keys())
            for next_char in transition_probabilities[char]:
                if next_char == 'End' and transition_probabilities[char]['End']  != 1.0:emmision_probabilities[char][next_char] = 1.0
                else:emmision_probabilities[char][next_char] = 1.0/n_transitions
        return {'transition_probabilities':transition_probabilities, 'emmision_probabilities':emmision_probabilities}

    def mul(self, mulList):
        mul = 1.0
        for m in mulList:
            mul *= m
        return mul

    def save_parameters(self,filename):
        model_parameters = self.attribute_markov_models
        filehandler = open(filename,"w")
        pickle.dump(model_parameters,filehandler)

    def load_parameters(self,filename):
        filehandler = open(filename,"r")
        self.attribute_markov_models = pickle.load(filehandler)

    def detection(self, dataset):
        predictions = []
        for data in dataset.data:
            data_atrs = data[0]
            structinfer = []
            for attr in data_atrs:
                value = ['Start']+list(data_atrs[attr])+['End']
                if attr in self.attribute_markov_models:
                    lst_transprob,lst_emitprob = [], []
                    n = 0
                    for index,char in enumerate(value):
                        n += 1
                        lst_trans,lst_emit = [], []
                        if index < len(value)-1:
                            if char in self.attribute_markov_models[attr]['transition_probabilities']:
                                for ch in self.attribute_markov_models[attr]['transition_probabilities'][char]:
                                    if value[index+1] == ch:lst_trans.append(self.attribute_markov_models[attr]['transition_probabilities'][char][value[index+1]])
                                    else:lst_trans.append(0)
                                lst_transprob.append(sum(lst_trans))
                            if char in self.attribute_markov_models[attr]['emmision_probabilities']:
                                for ch in self.attribute_markov_models[attr]['emmision_probabilities'][char]:
                                    if value[index+1] == ch:lst_emit.append(self.attribute_markov_models[attr]['emmision_probabilities'][char][value[index+1]])
                                    else:lst_emit.append(0)
                                lst_emitprob.append(sum(lst_emit))
                    prob_of_word = self.mul(lst_transprob) * self.mul(lst_emitprob)
                    structinfer.append(prob_of_word)
            predictions.append(sum(structinfer))
        return predictions

class AttributePresenceModel:
    def __init__(self):
        self.name = "Attribute Presence Model"
        self.attributes_present = []

    def learning(self, dataset):
        # training requires only positive data attributes
        data = dataset.get_attribute_dictionary(positive_only=True)
        self.attributes_present = data.keys()

    def save_parameters(self,filename):
        model_parameters = self.attributes_present
        filehandler = open(filename,"w")
        pickle.dump(model_parameters,filehandler)

    def load_parameters(self,filename):
        filehandler = open(filename,"r")
        self.attributes_present = pickle.load(filehandler)

    def detection(self, dataset):
        predictions = []
        for data in dataset.data:
            data_atrs = data[0]
            _p = []
            for atr in data_atrs:
                p = 0.0
                if not atr in self.attributes_present:
                    p = 1.0
                _p.append(p)
            predictions.append(0 if len(_p)==0 else sum(_p)/len(_p))
        return predictions
