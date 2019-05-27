import os
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import random
import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import sys
import pickle
import gensim
import numpy as np
import xml.dom.minidom
import json
class oldReader:
    def __init__(self):
        self.path='../CDR_Data/CDR.Corpus.v010516'


    def read_file(self,name='train'):
        filepath=self.path
        if(name=='train'):
            filepath+='/CDR_TrainingSet.PubTator.txt'
        elif(name=='dev'):
            filepath+='./CDR_DevelopmentSet.PubTator.txt'
        elif(name=='test'):
            filepath += './CDR_TestSet.PubTator.txt'

        myfile=open(filepath,'r',encoding='utf-8')
        document=[]
        temp=[]
        for sentence in myfile:

            if(sentence=='\n'):
                document.append(temp)
                temp=[]
            else:
                temp.append(sentence)

        train_instances=[]
        train_y=[]
        max_instance=0
        pos_count=0
        max_token=0
        origal_pos=0
        for doc in document:
            temp_c=0
            title=doc[0]
            title_index=title.find('|')
            newtitle=title[title_index+3:len(title)]
            abstract=doc[1]
            abstract_index=abstract.find('|')
            newabstract=abstract[abstract_index+3:len(abstract)]
            entity_dict={}
            entity_list=[]
            relation_list=[]
            all_text=newtitle+newabstract
            for index in range(2,len(doc)):
                sentence=doc[index]
                if(sentence.find('CID')!=-1):
                    relation=doc[index]
                    relation=relation.replace('\n','')
                    token_r=relation.split('\t')
                    relation_list.append([token_r[2],token_r[3]])
                    origal_pos+=1
                    temp_c+=1
                else:
                    token=sentence.split('\t')
                    token[5]=token[5].replace('\n','')
                    if(token[5].find('|')==-1):
                        entity_dict[token[5]]=token[1:5]
                        entity_list.append(token)
                    else:
                        pairs=token[5].split('|')
                        for i in pairs:
                            entity_dict[i]=token[1:5]
                            entity_list.append(token)
            entity_list.reverse()
            for i in entity_list:
                i[5]=i[5].replace('\n','')
                start = (int)(i[1])
                end = (int)(i[2])
                e = all_text[start:end]
                all_text=all_text[0:start]+i[5]+all_text[end:]
            key_list=list(entity_dict.keys())
            instances_count=0
            temp_pos=0
            for i in range(len(key_list)):
                for j in range(i+1,len(key_list)):
                    all_text=all_text.replace('\n',' ')
                    e1 = entity_dict[key_list[i]]
                    e2 = entity_dict[key_list[j]]
                    if((e1[3]=='Chemical'and e2[3]=='Disease')or(e2[3]=='Chemical'and e1[3]=='Disease')):
                        instances_count += 1
                        train_instances.append(all_text)
                        temp_entity_pairs=[key_list[i],key_list[j]]
                        res_temp_entity_pairs=[key_list[j],key_list[i]]
                        if(temp_entity_pairs in relation_list or res_temp_entity_pairs in relation_list):
                            train_y.append(1)
                            pos_count+=1
                            temp_pos+=1
                        else:
                            train_y.append(0)
            if(temp_pos!=temp_c):
                print('a')
            max_instance=max(instances_count,max_instance)
        print(instances_count)
        print('pos is ',pos_count)
        print('orginal is',origal_pos)

        return train_instances,train_y

    def divide_sentence(self,train_instances):
        max_sentence=0
        all_doc=[]
        for doc in train_instances:
            count=0
            temp=[]
            start = 0
            index=doc.find('.',start)
            while(index!=-1):
                count+=1
                if(index<len(doc)-2 and doc[index+1]==' ' and doc[index+2].isupper()):
                    temp.append(doc[start:index])
                    start=index+1
                    index=doc.find('.',start)
                else:
                    index = doc.find('.',index+1)
            max_sentence=max(max_sentence,count)
            temp.append(doc[start:])
            all_doc.append(temp)
        return all_doc,max_sentence

    def tokenizer(self,train_all_doc,train_instances,dev_all_doc,dev_instances,test_all_doc,test_instances,maxsentence,mylen):

        length_up_num=0
        max_token=0
        allfile=[train_all_doc,dev_all_doc,test_all_doc]
        rawfile=[train_instances,dev_instances,test_instances]
        all_document=[]
        for file in allfile:
            documents = []
            for doc in file:
                temp=[]
                for sencent in doc:
                    token = text_to_word_sequence(sencent, filters='!"#$%&()*+,-.:;=?@[\]^_`{|}/~')
                    max_token=max(max_token,len(token))
                    if(len(token)>mylen):
                        length_up_num+=1
                    temp.append(token)
                documents.append(temp)
            all_document.append(documents)

        all_raw=[]
        for file in rawfile:
            train_raw = []
            for sencent in train_instances:
                sencent =text_to_word_sequence(sencent, filters='!"#$%&()*+,-.:;=?@[\]^_`{|}/~')
                train_raw.append(sencent)
            all_raw.append(train_raw)

        all_dict=[]
        all_dict.extend(all_raw[0])
        all_dict.extend(all_raw[1])
        all_dict.extend(all_raw[2])
        print('max token is ',max_token)
        print('up than ',mylen,' num is',length_up_num)
        tokenizer=Tokenizer(filters='!"#$%&()*+,-.:;=/?@[\]^_`{|}~\'>',lower=True,split=" ")
        tokenizer.fit_on_texts(all_dict)
        vocab_size = len(tokenizer.word_index) + 1
        print(tokenizer.word_docs)
        print('vocab', vocab_size)

        pad_sentence=[]
        for i in range(mylen):
            pad_sentence.append(0)
        pad_sentence=np.array(pad_sentence)
        train_x=[]
        for i in all_document[0]:
            train_index_x = tokenizer.texts_to_sequences(i)
            train_index_x =(list)(sequence.pad_sequences(train_index_x,mylen,padding='post',truncating='post'))
            for j in range(len(train_index_x),maxsentence):
                train_index_x.append(pad_sentence)
            train_x.append(train_index_x)
        train_x=np.reshape(train_x,(len(train_x),maxsentence,mylen))

        dev_x = []
        for i in all_document[1]:
            train_index_x = tokenizer.texts_to_sequences(i)
            train_index_x = (list)(sequence.pad_sequences(train_index_x, mylen, padding='post', truncating='post'))
            for j in range(len(train_index_x), maxsentence):
                train_index_x.append(pad_sentence)
            dev_x.append(train_index_x)
        dev_x = np.reshape(dev_x, (len(dev_x), maxsentence, mylen))

        test_x = []
        for i in all_document[2]:
            train_index_x = tokenizer.texts_to_sequences(i)
            train_index_x = (list)(sequence.pad_sequences(train_index_x, mylen, padding='post', truncating='post'))
            for j in range(len(train_index_x), maxsentence):
                train_index_x.append(pad_sentence)
            test_x.append(train_index_x)
        test_x = np.reshape(test_x, (len(test_x), maxsentence, mylen))
        return train_x,dev_x,test_x

    def save_file(self,name, data):
        f = open('../data/' + name, 'w')
        pickle.dump(data, f)
        # for line in data:
        #     f.write(str(line))
        #     f.write("\n")
        f.close()


    def resign_dev(train_x, dev_x,train_y,dev_y):

        for i in range(3155):
            train_x.append(dev_x[i])
            train_y.append(dev_y[i])
        dev_x = dev_x[3155:]
        dev_y = dev_y[3155:]
        return train_x,train_y,dev_x,dev_y





if __name__ == "__main__":
     myreader=oldReader()
     train_instances, train_y = myreader.read_file(name='train')
     dev_instances, dev_y = myreader.read_file(name='dev')
     test_instances, tests_y = myreader.read_file(name='test')
     train_instances, train_y, dev_instances, dev_y = resign_dev(train_instances, dev_instances, train_y, dev_y)
     train_y = np.reshape(train_y, (len(train_y), 1))
     dev_y = np.reshape(dev_y, (len(dev_y), 1))
     tests_y = np.reshape(tests_y, (len(tests_y), 1))
     train_all_doc, train_max_sentence = myreader.divide_sentence(train_instances)
     dev_all_doc, dev_max_sentence = myreader.divide_sentence(dev_instances)
     test_all_doc, test_max_sentence = myreader.divide_sentence(test_instances)

     maxsentence =max(train_max_sentence,dev_max_sentence,test_max_sentence)
     mylen=120
     train_x, dev_x, test_x=myreader.tokenizer(train_all_doc,train_instances,dev_all_doc,dev_instances,test_all_doc,test_instances,maxsentence,mylen)

     myreader.save_file('train_x',train_x)
     myreader.save_file('dev_x', dev_x)
     myreader.save_file('test_x', test_x)
     myreader.save_file('train_y', train_y)
     myreader.save_file('dev_y', dev_y)
     myreader.save_file('tests_y', tests_y)
     # myreader.save_file('train_instances.txt', train_instances)
     # myreader.save_file('dev_instances.txt', dev_instances)
     # myreader.save_file('test_instances.txt', test_instances)
     # myreader.save_file('train_all_doc.txt', train_all_doc)
     # myreader.save_file('dev_all_doc.txt', dev_all_doc)
     # myreader.save_file('test_all_doc.txt', test_all_doc)

     print('finish')




