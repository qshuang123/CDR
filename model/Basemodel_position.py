from keras import Input, Model, optimizers
from keras.layers import Dense, Dropout, Bidirectional, GRU, Embedding, TimeDistributed, GlobalMaxPooling1D, concatenate
from keras.optimizers import RMSprop
from sklearn.metrics import f1_score
import pickle
from keras import backend as K
import logging
logging.basicConfig(level=logging.INFO, filename='mylog.log')
from utils.CyclicLR import CyclicLR
from utils.threshold import threshold_search, f1, count, load_pkl


batch_size = 32



def basemodel_position(drop,embedding_matrix,max_sentence_length,max_sentence_number):
    inputs = Input(shape=(max_sentence_number,max_sentence_length,), name='input')
    position1 = Input(shape=(max_sentence_number, max_sentence_length,), name='position1')
    position2 = Input(shape=(max_sentence_number, max_sentence_length,), name='position2')
    context_vector = TimeDistributed(Embedding(len(embedding_matrix), 300, input_length=(max_sentence_length,), mask_zero=False,
                               weights=[embedding_matrix], name='sentence_emd', trainable=True))(inputs)
    pos_embedding=Embedding(5801, 30, input_length=(max_sentence_length,), mask_zero=False,name='pos_emd',trainable=True)
    # entity_vector = TimeDistributed(Embedding(2, 30, input_length=(max_sentence_length,), mask_zero=False,name='pos_emd2'))(inputs)
    pos1=TimeDistributed(pos_embedding)(position1)
    pos2 = TimeDistributed(pos_embedding)(position2)
    x = concatenate([context_vector, pos1, pos2])
    lstm = TimeDistributed(Bidirectional(GRU(256, return_sequences=False, dropout=drop)))(x)
    out = Dropout(drop)(lstm)
    out = GlobalMaxPooling1D()(out)
    out = Dense(64)(out)
    out = Dense(1, activation='sigmoid', name='output')(out)
    model = Model([inputs,position1,position2], out)
    model.summary()
    adam = optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def run(model,x_train,y_train,x_develop,y_develop,x_test,y_test,train_pos1,dev_pos1,test_pos1,train_pos2,dev_pos2,test_pos2):
    clr = CyclicLR(base_lr=0.001, max_lr=0.002,
                   step_size=300., mode='exp_range',
                   gamma=0.99994)
    for i in range(20):
        logging.info('epoch'+str(i))
        # model.fit(x_train, y_train, epochs=1, batch_size=batch_size, callbacks=[clr, ])
        model.fit([x_train,train_pos1,train_pos2], y_train, epochs=1, batch_size=batch_size)
        pred_dev_y = model.predict([x_develop,dev_pos1,dev_pos2], batch_size=batch_size, verbose=0)
        # best_threshold, best_scores = threshold_search(y_develop,pred_dev_y)
        best_score = f1(y_develop, (pred_dev_y > 0.5).astype(int))
        print("Epoch: ", i, "-    dev F1 Score: {:.4f}".format(best_score))
        logging.info("Epoch: "+str(i)+"-    dev F1 Score: "+str(best_score))
        pred_test_y = model.predict([x_test,test_pos1,test_pos2], batch_size=batch_size, verbose=0)
        best_score = f1(y_test, (pred_test_y > 0.5).astype(int))
        print("Epoch: ", i, "-    Val F1 Score: {:.4f}".format(best_score))
        logging.info("Epoch: "+str(i)+"-    Val F1 Score:"+str(best_score))







if __name__ == "__main__":

    position1 = load_pkl('position1')
    position2 = load_pkl('position2')
    train_pos1 = position1[:9753]
    dev_pos1 = position1[9753:10837]
    test_pos1 = position1[10837:]
    train_pos2 = position2[:9753]
    dev_pos2 = position2[9753:10837]
    test_pos2 = position2[10837:]

    train_x = load_pkl('train_x')
    train_y = load_pkl('train_y')
    count(train_y)
    dev_x = load_pkl('dev_x')
    dev_y = load_pkl('dev_y')
    count(dev_y)
    test_x = load_pkl('test_x')
    test_y = load_pkl('tests_y')
    embedding_matrix = load_pkl('embedding_matrix')
    logging.info('model+position+dense')

    model = basemodel_position(0.3,embedding_matrix,200,23)
    run(model, train_x, train_y, dev_x, dev_y, test_x, test_y, train_pos1, dev_pos1, test_pos1, train_pos2, dev_pos2, test_pos2)







