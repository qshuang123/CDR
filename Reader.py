from __future__ import division
import os
import sys
from keras.layers import Flatten
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import random
import numpy as np
from keras import backend as K
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import sys
import pickle
import gensim
import numpy as np
from utils.threshold import load_pkl
import xml.dom.minidom


class Reader:
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
        entity_ins=[]
        train_y=[]
        max_instance=0
        pos_count=0
        max_token=0
        origal_pos=0
        all_entity_list={}


        for doc in document:
            temp_c=0
            title=doc[0]
            title_index=title.find('|')
            docID=title[:title_index]
            newtitle=title[title_index+3:len(title)]
            abstract=doc[1]
            abstract_index=abstract.find('|')
            newabstract=abstract[abstract_index+3:len(abstract)]
            entity_dict={}
            entity_list=[]
            relation_list=[]
            entity_help = []
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
                        entity_help.append(token)
                    else:
                        pairs=token[5].split('|')
                        for i in pairs:
                            entity_dict[i]=token[1:5]
                            entity_list.append(token)
                            entity_help.append(token)
            all_entity_list[docID] = entity_help
            entity_list.reverse()



            for i in entity_list:
                i[5]=i[5].replace('\n','')
                replace_id=i[5]
                if(i[5])=='-1':
                    replace_id='D000000'
                start = (int)(i[1])
                end = (int)(i[2])
                e = all_text[start:end]
                type=''
                if i[4]=='Chemical':
                    type='ch_'
                else:
                    type='ds_'
                all_text=all_text[0:start]+type+replace_id+all_text[end:]
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
                        entity1_text = e1[2]
                        entity2_text = e2[2]
                        e1_tagged=''
                        if(e1[3]=='Chemical'):
                            e1_tagged_start='start'
                            e1_tagged_end='ch_end'
                            e2_tagged_start='start'
                            e2_tagged_end='ds_end '
                        else:
                            e2_tagged_start = 'start'
                            e2_tagged_end = 'ch_end'
                            e1_tagged_start = 'start'
                            e1_tagged_end = 'ds_end'
                        e1_id=key_list[i]
                        e2_id=key_list[j]
                        if(key_list[i]=='-1'):
                            e1_id='D000000'
                        if(key_list[j]=='-1'):
                            e2_id='D000000'
                        replace_e1 = e1_tagged_start + ' ' + e1_id + ' ' + e1_tagged_end
                        replace_e2 = e2_tagged_start + ' ' + e2_id + ' ' + e2_tagged_end
                        entity1_id = e1_id
                        entity2_id = e2_id
                        entity=[docID, entity1_id,entity1_text,entity2_id,entity2_text]
                        entity_ins.append(entity)
                        replace_ins=all_text.replace(e1_id,replace_e1)
                        replace_ins=replace_ins.replace(e2_id,replace_e2)
                        train_instances.append(replace_ins)
                        temp_entity_pairs=[key_list[i],key_list[j]]
                        entity_list.append([key_list[i],key_list[j]])
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

        return train_instances,train_y,entity_ins,all_entity_list

    def divide_sentence(self,train_instances):
        max_sentence=0
        all_doc=[]
        for doc in train_instances:
            count=0
            temp=[]
            start = 0
            if(doc.find('?')!=-1):
                doc = doc.replace('?', '.')
            index=doc.find('.',start)
            while(index!=-1):
                count+=1
                if(index < len(doc)-2 and doc[index+1] == ' ' and doc[index+2].isupper()):
                    temp.append(doc[start:index])
                    start=index+1
                    index=doc.find('.',start)
                else:
                    index = doc.find('.',index+1)
            max_sentence=max(max_sentence,count)
            temp.append(doc[start:])
            all_doc.append(temp)
        return all_doc,max_sentence


    def tokenizer(self,train_all_doc,dev_all_doc,test_all_doc,maxsentence,mylen):

        length_up_num=0
        max_token=0
        allfile=[train_all_doc,dev_all_doc,test_all_doc]

        wordvec=[]
        all_document=[]
        for file in allfile:
            documents = []
            for doc in file:
                temp=[]
                word=[]
                for sencent in doc:
                    sen=""
                    token = text_to_word_sequence(sencent, filters='!"#$%&()*+,-.:;=?@[\]^`{|}/~', lower=True, split=" ")
                    max_token=max(max_token,len(token))
                    if(len(token)>mylen):
                        length_up_num+=1
                    temp.append(token)
                    for c in token:
                        sen=sen+c+" "
                    sen=sen.strip(" ")
                    word.append(sen)
                wordvec.append(word)
                documents.append(temp)
            all_document.append(documents)


        all_raw=[]
        for file in all_document:
            for inst in file:
                instance = []
                for sencent in inst:
                    for word in sencent:
                        instance.append(word)
                all_raw.append(instance)


        all_dict=[]
        all_dict.extend(all_raw)

        print('max token is ',max_token)
        print('up than ',mylen,' num is',length_up_num)
        tokenizer=Tokenizer(filters='!"#$%&()*+,-.:;=/?@[\]^`{|}~\'>',lower=True,split=" ")
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
        return train_x,dev_x,test_x,tokenizer,wordvec

    def save_file(self,name, data):
        f = open('../data/' + name, 'wb')
        pickle.dump(data, f)
        f.close()
    def save_txt(self,name, data):
        f = open('../data/' + name, 'w')
        # pickle.dump(data, f)
        for line in data:
            f.write(str(line))
            f.write("\n")
        f.close()

    def resign_dev(self, train_x, dev_x,train_y,dev_y,train_entity,dev_entity):

        # for i in range(3155):
        #     train_x.append(dev_x[i])
        #     train_y.append(dev_y[i])
        # dev_x = dev_x[3155:]
        # dev_y = dev_y[3155:]

        for i in range(4238):
            train_x.append(dev_x[i])
            train_y.append(dev_y[i])
            train_entity.append(dev_entity[i])
        dev_x = dev_x[4238:]
        dev_y = dev_y[4238:]
        dev_entity = dev_entity[4238:]
        return train_x,train_y,dev_x,dev_y,train_entity,dev_entity

    def reshape_y(self,train_y):
        train_y = np.reshape(train_y, (len(train_y), 1))
        return train_y

    def produce_matrix(self,tokenizer):
        all_num = 0
        oov = 0
        embeddings_index = {}
        myword2vec = gensim.models.KeyedVectors.load_word2vec_format(r'../data/Drug_wordvec_300.txt', binary=False) #GloVe Model
        print('loaded')
        for i in range(len(myword2vec.index2word)):
            embeddings_index[myword2vec.index2word[i]]=myword2vec.syn0[i]
        embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
        for word, i in tokenizer.word_index.items():
            all_num += 1
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                print(word)
                oov += 1
        print('all is  ', all_num)
        print('the number of not found ', oov)
        return embedding_matrix


    def get_entity1_pos(self, wordvec, maxlen, maxsentence):
        all_pos=[]
        for doc in wordvec:
            doc_pos = []
            for sen in doc:
                words = sen.split(" ")
                pos = []
                for n in range(len(words)):
                    pos.append(0)
                i=0
                for word in words:
                    if word =='ch_start':
                        pos[i+1]=1
                    # if word == 'ds_start':
                    #     pos[i+1]=1
                    i += 1
                doc_pos.append(pos)
            all_pos.append(doc_pos)

        position = []
        for doc in all_pos:
            doc_ins=[]
            doc_new=[]
            doclen = len(doc)
            if doclen < maxsentence:
                j=[]
                for i in range(maxsentence-doclen):
                    doc.append(j)
                doc_ins = doc
            else:
                for n in range(maxsentence):
                    doc_ins.append(doc[n])
            for sen in doc_ins:
                sen_ins=[]
                senlen = len(sen)
                if senlen < maxlen:
                    for i in range(maxlen - senlen):
                        sen.append(0)
                    sen_ins=sen
                else:
                    for n in range(maxlen):
                        sen_ins.append(sen[n])
                doc_new.append(sen_ins)
            position.append(doc_new)
        print(np.array(position).shape)
        position = np.array(position)
        return position
    def get_entity2_pos(self, wordvec, maxlen, maxsentence):
        all_pos=[]
        for doc in wordvec:
            doc_pos = []
            for sen in doc:
                words = sen.split(" ")
                pos = []
                for n in range(len(words)):
                    pos.append(0)
                i=0
                for word in words:
                    # if word =='ch_start':
                    #     pos[i+1]=1
                    if word == 'ds_start':
                        pos[i+1]=1
                    i += 1
                doc_pos.append(pos)
            all_pos.append(doc_pos)

        position = []
        for doc in all_pos:
            doc_ins=[]
            doc_new=[]
            doclen = len(doc)
            if doclen < maxsentence:
                j=[]
                for i in range(maxsentence-doclen):
                    doc.append(j)
                doc_ins = doc
            else:
                for n in range(maxsentence):
                    doc_ins.append(doc[n])
            for sen in doc_ins:
                sen_ins=[]
                senlen = len(sen)
                if senlen < maxlen:
                    for i in range(maxlen - senlen):
                        sen.append(0)
                    sen_ins=sen
                else:
                    for n in range(maxlen):
                        sen_ins.append(sen[n])
                doc_new.append(sen_ins)
            position.append(doc_new)
        print(np.array(position).shape)
        position = np.array(position)
        return position

    def get_position(self, entity_pos, maxlen, maxsentence):
        position = []
        print(entity_pos.shape)
        all_pos = np.reshape(entity_pos, [-1, maxlen*maxsentence])
        print(all_pos.shape)
        print(all_pos[0])
        all_pos = all_pos.tolist()
        tag=0
        for doc in all_pos:
            redoc=list(reversed(doc))
            sen_pos = []
            i = 0
            for sen in doc:
                if sen != 1:
                    if 1 not in doc[i:]:
                        index1 = -1
                    else:
                        index1 = doc[i:].index(1)
                    if 1 not in redoc[maxlen * maxsentence - i - 1:]:
                        index2 = -1
                    else:
                        index2 = redoc[maxlen * maxsentence - i - 1:].index(1)
                    if(index1 == -1 and index2 == -1):
                        sen_pos.append(0)
                    else:
                        index1 = index1 if index1!=-1 else sys.maxsize
                        index2 = index2 if index2!=-1 else sys.maxsize
                        pos = maxlen*maxsentence-index1 if index1<index2 else maxlen*maxsentence-index2
                        sen_pos.append(pos)
                else:
                    sen_pos.append(maxlen*maxsentence)
                i+=1
            position.append(sen_pos)
            print("***************"+str(tag)+"*********")
            tag+=1
        position = np.array(position)
        position = np.reshape(position,[-1,maxsentence,maxlen])
        print(position.shape)
        return position

    def get_position_wewighted(self, position):
        position = position.astype(np.float32)
        for i in range(len(position)):
            print(i)
            for j in range(len(position[0])):
                sum=0
                for k in range(len(position[0][0])):
                    sum+=position[i][j][k]
                if sum!=0:
                    for k in range(len(position[0][0])):
                        temp=position[i][j][k]
                        temp1 = (float(position[i][j][k])/sum)
                        position[i][j][k] = temp1
        return position

    def delete_sentence(self,train):
        for i in range(len(train)):
            j=len(train[i])
            while j >= 0:
                j -= 1
                if "ch_" not in train[i][j] and "ds_" not in train[i][j]:
                    del train[i][j]


        count=0
        num=0
        for i in range(len(train)):
            for j in range(len(train[i])):
                count=max(count,len(train[i]))
                num = max(num, len(train[i][j]))

        return train,count,num

    def delete_train_neg_ins(self,train_x,train_y):

        deletelist = load_pkl('train_delete_list')

        positive = 0
        negetive = 0
        nopair=0

        # 倒序进行删除
        redelete = list(reversed(deletelist))
        for p in redelete:
            print('del')
            del train_x[p]
            del train_y[p]


        print("finish")
        print("pos"+str(positive))
        print("neg"+str(negetive))
        print("nopair"+str(nopair))
        return train_x, train_y


    def delete_dev_neg_ins(self,dev_x, dev_y):
        deletelist = load_pkl('dev_delete_list')
        positive = 0
        negetive = 0
        nopair=0

        # 倒序进行删除
        redelete = list(reversed(deletelist))
        for p in redelete:
            print('del')
            del dev_x[p]
            del dev_y[p]

        print("finish")
        print("pos"+str(positive))
        print("neg"+str(negetive))
        print("nopair"+str(nopair))
        return dev_x, dev_y










if __name__ == "__main__":
     myreader = Reader()
     # train_instances, train_y, train_entity ,train_doc= myreader.read_file(name='train')
     # dev_instances, dev_y, dev_entity, dev_doc = myreader.read_file(name='dev')
     # test_instances, tests_y, test_entity,test_doc = myreader.read_file(name='test')
     #
     # train_instances,train_y, dev_instances, dev_y,train_entity,dev_entity = myreader.resign_dev(train_instances, dev_instances, train_y, dev_y, train_entity,dev_entity)  # 20%
     #
     # train_all_doc, train_max_sentence = myreader.divide_sentence(train_instances)
     # dev_all_doc, dev_max_sentence = myreader.divide_sentence(dev_instances)
     # test_all_doc, test_max_sentence = myreader.divide_sentence(test_instances)
     #
     # train_y = myreader.reshape_y(train_y)
     # dev_y = myreader.reshape_y(dev_y)
     # tests_y = myreader.reshape_y(tests_y)
     #
     # train_all_doc, train_max_sentence,train_num = myreader.delete_sentence(train_all_doc)
     # dev_all_doc, dev_max_sentence,dev_num = myreader.delete_sentence(dev_all_doc)
     # test_all_doc, test_max_sentence,test_num = myreader.delete_sentence(test_all_doc)
     #
     # maxsentence = max(train_max_sentence, dev_max_sentence, test_max_sentence)
     # print(train_max_sentence)
     # print(dev_max_sentence)
     # print(test_max_sentence)
     #
     # maxsentence = 23
     # maxlen = 200
     #
     # train_x, dev_x, test_x, tokenizer, wordvec = myreader.tokenizer(train_all_doc, dev_all_doc, test_all_doc, maxsentence,maxlen)
     # embedding_matrix = myreader.produce_matrix(tokenizer)


     entity=load_pkl('test_doc_entity_list')
     print(0)


     # entity1_pos = myreader.get_entity1_pos(wordvec,maxlen,maxsentence)
     # entity2_pos = myreader.get_entity2_pos(wordvec, maxlen, maxsentence)
     #
     #
     # position1 = myreader.get_position(entity1_pos, maxlen, maxsentence)
     # position2 = myreader.get_position(entity2_pos, maxlen, maxsentence)





     # myreader.save_file("train_x",train_x)
     # myreader.save_file("dev_x",dev_x)
     # myreader.save_file("test_x",test_x)
     # myreader.save_file("train_y", train_y)
     # myreader.save_file("dev_y", dev_y)
     # myreader.save_file("tests_y", tests_y)
     # myreader.save_file("entity1_pos", entity1_pos)
     # myreader.save_file("position1", position1)
     # myreader.save_file("entity2_pos",entity2_pos)
     # myreader.save_file("position2",position2)
     # myreader.save_file("train_entity",train_entity)
     # myreader.save_file("dev_entity",dev_entity)
     # myreader.save_file("test_entity",test_entity)
     # myreader.save_file("embedding_matrix", embedding_matrix)













