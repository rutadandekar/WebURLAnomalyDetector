import time
import datetime
import os
import argparse

class Timeout:
    def __init__(self):
        self._t = datetime.datetime.now()
        # Filtering
        self.M = 0.9
        self.N = 1.0-self.M
        self.time_remaining = 0.0

    def time_difference(self,set=False,debug=None):
        t = datetime.datetime.now()
        print 'Rate:',1.0/(t-self._t).total_seconds(),("" if debug is None else debug)
        if set==True:
            self._t = t
        return 1.0/(t-self._t).total_seconds()

    def prediction(self,current,total,debug=False):
        t = datetime.datetime.now()
        rate = 1.0/(t-self._t).total_seconds()
        self._t = t
        self.time_remaining = self.M*self.time_remaining + self.N*((total-current)/(rate*3600))
        print 'Time remaining:',self.time_remaining,'Hours',("" if debug is False else str(current)+'/'+str(total))

def file_exists(f_name):
    return os.path.isfile(f_name)

def path_exists(p_name, create_if_not=False):
    p = os.path.exists(p_name)
    if p==False and create_if_not==True:
        os.makedirs(p_name)
    return p

def compute_accuracy(dataset,predictions,clip=0.0):
    accuracy = 1.0 - sum([abs((1 if a>clip else 0)-b[1]) for a,b in zip(predictions,dataset.data)])/float(len(dataset.data))
    return accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Web URL Anomaly Detector")
    parser.add_argument("--mode", help="Train/Detect",default='Train')
    parser.add_argument("--data", help="path to file to process from",default=None)
    return parser.parse_args()
