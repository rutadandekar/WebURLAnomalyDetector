from Modules import Utils
import os
import numpy as np

class DataPreprocessing:
    def __init__(self, file):
        self.f = None
        self.fname = file
        self.data = []
        self.urls = []
        self.positive_data = None
        self.negative_data = None
        self.atr_dict = None
        self.ops = None
        self.where = 0
        self.read_data_from_file(file)

    def __del__(self):
        if (not self.f is None) and (not self.f.closed):
            self.f.close()

    def read_data_from_file(self,f):
        self.open_file_if_exists(f)
        self.read_and_label_data()

    def open_file_if_exists(self,f):
        if(not Utils.file_exists(f)):
            self.f = None
        else:
            self.f = open(f,'r')

    def read_and_label_data(self, pos=None, nolabel=False):
        if self.f is None:
            return
        self.data = []
        self.urls = []
        if pos is None:st_size = 0
        else: st_size = pos
        self.f.seek(st_size)
        for line in self.f:
            # only proceed if querry in url
            if not '?' in line:
                continue
            # remove any excess data, keep the get request and response code
            l = line.split('"')
            if len(l)<3:
                continue
            # make sure this is a get request
            if not l[1][:3]=='GET':
                continue
            # Find attributes from querries
            atrs = self.find_attributes(l[1])
            atrs_ordered = self.find_attributes_ordered(l[1])
            if nolabel==True:
                label = 0
            else:
                # Make sure response code is provided
                response = l[2].split(' ')
                if (len(response)<2) or (not len(response[1])==3):
                    continue
                # Determine label (0->positive, 1->negative)
                label = 0 if (int(response[1])>=200 and int(response[1]<300)) else 1
            # add to a ordered list
            self.data.append((atrs,label,atrs_ordered))
            self.urls.append(line)
        # record the last read file position
        self.where = self.f.tell()
        self.f.close()

    def find_attributes(self,url_formatted):
        atr_list = url_formatted.split('?')[-1].split(' ')[0].split('&')
        queries = {}
        for atr in atr_list:
            # make sure it has a key value pair
            if not '=' in atr:
                continue
            kv = atr.split('=')
            queries[kv[0]]= kv[1]
        return queries

    def find_attributes_ordered(self,url_formatted):
        atr_list = url_formatted.split('?')[-1].split(' ')[0].split('&')
        queries = []
        for atr in atr_list:
            # make sure it has a key value pair
            if not '=' in atr:
                continue
            kv = atr.split('=')
            queries.append((kv[0],kv[1]))
        return queries

    def get_positive_data(self):
        if not len(self.data)>0:
            raise ValueError("Please load data first")
        if not self.positive_data is None:
            return self.positive_data
        self.positive_data = [x for x in self.data if x[1]==0]
        return self.positive_data

    def get_negative_data(self):
        if not len(self.data)>0:
            raise ValueError("Please load data first")
        if not self.negative_data is None:
            return self.negative_data
        self.negative_data = [x for x in self.data if x[1]==1]
        return self.negative_data

    def get_attribute_dictionary(self, positive_only=True, negative_only=False):
        if not len(self.data)>0:
            raise ValueError("Please load data first")
        if not self.atr_dict is None:
            return self.atr_dict
        data = self.get_positive_data() if positive_only is True else (self.get_negative_data() if negative_only is True else self.data)
        self.atr_dict = {}
        for d in data:
            for atr in d[0]:
                if atr in self.atr_dict:
                    self.atr_dict[atr].append(d[0][atr])
                else:
                    self.atr_dict[atr] = [d[0][atr]]
        return self.atr_dict

    def outputs(self):
        if not len(self.data)>0:
            raise ValueError("Please load data first")
        if not self.ops is None:
            return self.ops
        self.ops = []
        for d in self.data:
            self.ops.append(d[1])
        return self.ops

    def save_anomalous_urls(self,preds,fname):
        with open(fname,'a+') as afile:
            for url,pred in zip(self.urls,preds):
                if pred[1]>=0.5:
                    afile.write(url)
