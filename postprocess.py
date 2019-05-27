import pickle
import numpy as np
from utils.threshold import load_pkl, f1


def ansisy_predict(predict,ins):
    y=[]
    dict={}
    for i in ins:
        dict[i[0]]=0
    for i in range(len(predict)):
        if predict[i]>0.5:
            y.append(1)
            docid=ins[i][0]
            dict[docid]=1
        else:
            y.append(0)
    # dict没有抽取出正例的文章为0
    return y, dict

def ansis_doc(docs):
    dict={}
    for doc_id,entity_list in docs.items():
        entity_pair=[]
        dic_chemical={}
        dic_disease={}

        for entity in entity_list:
            entity_type=entity[4]
            entity_id=entity[5]
            if entity_type=='Chemical':
                if entity_id in dic_chemical.keys():
                    v=dic_chemical.get(entity_id)
                    v+=1
                    dic_chemical[entity_id]=v
                else:
                    dic_chemical[entity_id] = 1
            if entity_type=='Disease':
                if entity_id in dic_disease.keys():
                    v=dic_disease.get(entity_id)
                    v+=1
                    dic_disease[entity_id]=v
                else:
                    dic_disease[entity_id] = 1
        entity1_id = ""
        entity1_nums = 0
        entity2_id = ""
        entity2_nums = 0
        for k,v in dic_chemical.items():
            if(v>entity1_nums):
                entity1_id=k
                entity1_nums=v

        for k,v in dic_disease.items():
            if(v>entity2_nums):
                entity2_id=k
                entity2_nums=v
        entity_pair.append(entity1_id)
        entity_pair.append(entity2_id)
        dict[doc_id]=entity_pair

    return dict

def match(y,y_dict,y_ins,doc_dict):
    update_y_list=[]
    new_y=[]
    count=0
    doccount=0
    for doc, tag in y_dict.items():
        if tag==0:
            doccount+=1
            print("shuoud")
            print(doc_dict.get(doc))
            target_ins=doc_dict.get(doc)

            entity1_id=target_ins[0]
            entity2_id = target_ins[1]
            if entity1_id=="-1":
                entity1_id="D000000"
            if entity2_id=="-1":
                entity2_id="D000000"
            # entity1_id=""
            # entity2_id=""
            # entity1s_id=[]
            # entity2s_id=[]
            #
            # index1 = target_ins[0].find("|")
            # if index1==-1:
            #     entity1_id=target_ins[0]
            # else:
            #     entity1s_id = target_ins[0].split("|")
            #
            # index2 = target_ins[1].find("|")
            # if index2 == -1:
            #     entity2_id = target_ins[1]
            # else:
            #     entity2s_id = target_ins[1].split("|")

            vis = False
            for i in range(len(y_ins)):
                if y_ins[i][0]==doc :
                        # if entity1_id==y_ins[i][1] or y_ins[i][1] in entity1s_id:
                        #     if entity2_id == y_ins[i][3] or y_ins[i][3] in entity2s_id:
                        if entity1_id==y_ins[i][1] and entity2_id == y_ins[i][3]:
                                update_y_list.append(i)
                                count+=1
                                vis = True
        if tag==0 and vis==False:
            print("no")
            print(doc_dict.get(doc))
    for i in range(len(y)):
        if i in update_y_list:
            if y[i]==0:
                y[i]=1
    print(count)
    print(doccount)
    return y


def post_process(predict_file):
    test_entity = load_pkl('test_doc_entity_list')
    true_y=load_pkl('tests_y')
    file = open('../result/' + predict_file, 'rb')
    predict = pickle.load(file)
    file.close()
    test = load_pkl('test_doc_entity')

    y,y_dict=ansisy_predict(predict,test_entity)
    doc_dict=ansis_doc(test)
    new_y = match(y,y_dict,test_entity,doc_dict)
    # p, r, f = f1(true_y, (predict > 0.5).astype(int))
    p1, r1, f_1 = f1(true_y, new_y)
    # print(f)
    print(f_1)
    print("///////")
    return new_y,true_y

# postprocess2


def mesh_id_dict():
    mesh_dict = {}
    f = open('../CDR_Data/' + 'mtrees2019.bin', 'rb')
    for line in f:
        line = str(line)[2:-3]
        mesh_one = line.split(";")
        mesh_dict[mesh_one[0].lower()] = mesh_one[1]

    return mesh_dict


def get_entity_mesh(mesh_dict,instance_entity):
    instance_entity_mesh = []
    sum = 0
    for i in range(len(instance_entity)):
        temp = instance_entity[i]
        disease = instance_entity[i][4]
        if mesh_dict.__contains__(disease):
            temp.append(mesh_dict.get(disease))
            sum += 1
        else:
            temp.append("undefine")
        instance_entity_mesh.append(temp)
    return instance_entity_mesh


def delete_special_instance(instance_entity_mesh):
    res=[]
    doc={}
    doc_entity=[]
    doc_tag = ""
    for i in range(len(instance_entity_mesh)):

        if doc_tag=="":
            doc_entity.append(instance_entity_mesh[i][-1])
            doc_tag=instance_entity_mesh[i][0]
            if i == len(instance_entity_mesh):
                doc[doc_tag] = doc_entity

        else:
            if instance_entity_mesh[i][0] == doc_tag:
                doc_entity.append(instance_entity_mesh[i][-1])
            else:
                temp=[]
                temp.extend(doc_entity)
                doc[doc_tag]=temp
                doc_tag=instance_entity_mesh[i][0]
                doc_entity.clear()
                doc_entity.append(instance_entity_mesh[i][-1])
    temp = []
    temp.extend(doc_entity)
    doc[doc_tag] = temp

    for j in range(len(instance_entity_mesh)):
        doc_index=instance_entity_mesh[j][0]
        entity = instance_entity_mesh[j][-1]
        meshid=[]
        if doc.__contains__(doc_index):
            meshid.extend(doc.get(doc_index))
        else:
            print(doc_index)
        tag=False
        for m in meshid:
            if m != entity and m.find(entity) != -1:
                tag=True
                break
        if tag==True:
            res.append(j)

    return res


def post2(predict_file, delete_list,true_y,predict):
    # true_y = load_pkl('tests_y')
    # file = open('../result/' + predict_file, 'rb')
    # predict = pickle.load(file)
    # file.close()

    new_pre=[]
    new_y=[]
    index=0
    for i in range(len(predict)):
        if i not in delete_list:
            new_pre.append(predict[i])
            new_y.append(true_y[i])


    new_pre=np.array(new_pre)
    new_y = np.array(new_y)

    # p, r, f = f1(true_y, (predict > 0.5).astype(int))
    p1, r1, f_1 = f1(new_y, (new_pre > 0.5).astype(int))
    # print(f)
    print(f_1)
if __name__ == "__main__":
    predict='0.6009433962264151_predict'
    new_y,predict_y=post_process(predict)
    test_entity = load_pkl('test_doc_entity_list')
    train_entity = load_pkl('train_doc_entity_list')
    dev_entity = load_pkl('dev_doc_entity_list')
    mesh_dict = mesh_id_dict()
    instance_entity_mesh = get_entity_mesh(mesh_dict,test_entity)
    delete_list = delete_special_instance(instance_entity_mesh)
    train_entity_mesh = get_entity_mesh(mesh_dict, train_entity)
    train_delete_list = delete_special_instance(train_entity_mesh)
    dev_entity_mesh = get_entity_mesh(mesh_dict, dev_entity)
    dev_delete_list = delete_special_instance(dev_entity_mesh)
    #
    post2(predict,delete_list,new_y,predict_y)