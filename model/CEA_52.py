import keras
from keras import Input, Model, optimizers
from keras.activations import softmax
from keras.backend import batch_dot
from keras.layers import Dense, Dropout, Bidirectional, GRU, Embedding, TimeDistributed, GlobalMaxPooling1D, \
    concatenate, Lambda, Dot, Permute, Concatenate, Multiply, Add
from keras.optimizers import RMSprop
from sklearn.metrics import f1_score
import pickle
from keras import backend as K
import logging
logging.basicConfig(level=logging.INFO, filename='mylog.log')
from utils.CyclicLR import CyclicLR
from utils.threshold import threshold_search, f1, count, load_pkl


batch_size = 16

def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


def soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""

    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Lambda(lambda x: softmax(x, axis=2), output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(w_att_2)
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def entity_attention(input):
    """Align text representation with neural soft attention"""

    pos = K.reshape(input[0], (-1, 23, 200, 1))
    pos = K.repeat_elements(pos, 512, -1)
    # 对200求最大
    result = pos*input[1]
    result1 = K.reshape(result, (-1, 200, 512))
    result1 = GlobalMaxPooling1D()(result1)
    result2 = K.reshape(result1,(-1,23,512))
    # result2 = GlobalMaxPooling1D()(result2)
    return result2

def entity_attention_output_shape(input_shape):
    shape = list(input_shape[0])
    shape[2] = 512
    return tuple(shape)


def get_context(input):
    """Align text representation with neural soft attention"""

    result1 = K.reshape(input, (-1, 200, 512))
    result1 = GlobalMaxPooling1D()(result1)
    result2 = K.reshape(result1, (-1, 23, 512))
    # result2 = GlobalMaxPooling1D()(result2)
    return result2

def get_context_output_shape(input_shape):
    shape = list(input_shape)
    shape.pop(2)
    assert len(shape) == 3
    return tuple(shape)


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


def skip(context_rep, entity1_rep, entity2_rep,k):
    for i in range(k):
        context_rep = keras.layers.Permute((2, 1))(context_rep)
        atten1_weighted = Dot(axes=1)([context_rep, entity1_rep])
        context_rep = keras.layers.Permute((2, 1))(context_rep)
        atten1_weighted = Lambda(lambda x: softmax(x, axis=1), output_shape=unchanged_shape)(atten1_weighted)
        context1_rep = Dot(axes=1)([atten1_weighted, context_rep])

        context_atten1 = keras.layers.RepeatVector(512)(atten1_weighted)
        context_atten1 = keras.layers.Permute((2, 1))(context_atten1)
        context1 = keras.layers.Multiply()([context_atten1, context_rep])

        context_rep = keras.layers.Permute((2, 1))(context_rep)
        atten2_weighted = Dot(axes=1)([context_rep, entity2_rep])
        context_rep = keras.layers.Permute((2, 1))(context_rep)
        atten2_weighted = Lambda(lambda x: softmax(x, axis=1), output_shape=unchanged_shape)(atten2_weighted)
        context2_rep = Dot(axes=1)([atten2_weighted, context_rep])

        context_atten2 = keras.layers.RepeatVector(512)(atten2_weighted)
        context_atten2 = keras.layers.Permute((2, 1))(context_atten2)
        context2 = keras.layers.Multiply()([context_atten2, context_rep])

        entity1_rep = concatenate([entity1_rep, context1_rep])
        entity1_rep = Dense(512)(entity1_rep)
        entity2_rep = concatenate([entity2_rep, context2_rep])
        entity2_rep = Dense(512)(entity2_rep)

        context_rep = concatenate([context_rep, context1, context2])
        context_rep = Dense(512)(context_rep)
    print(context_rep.shape)
    print(entity1_rep.shape)
    print(entity2_rep.shape)
    return context_rep,entity1_rep,entity2_rep





def basemodel_position(drop,embedding_matrix,max_sentence_length,max_sentence_number,k):
    inputs = Input(shape=(max_sentence_number,max_sentence_length,), name='input')
    position1 = Input(shape=(max_sentence_number, max_sentence_length,), name='position1')
    position2 = Input(shape=(max_sentence_number, max_sentence_length,), name='position2')
    entity1_mask = Input(shape=(max_sentence_number, max_sentence_length,), name='entity1_mask')
    entity2_mask = Input(shape=(max_sentence_number, max_sentence_length,), name='entity2_mask')
    context_vector = TimeDistributed(Embedding(len(embedding_matrix), 300, input_length=(max_sentence_length,), mask_zero=False,
                               weights=[embedding_matrix], name='sentence_emd', trainable=True))(inputs)
    pos_embedding=Embedding(5801, 30, input_length=(max_sentence_length,), mask_zero=False,name='pos_emd',trainable=True)
    # entity_vector = TimeDistributed(Embedding(2, 30, input_length=(max_sentence_length,), mask_zero=False,name='pos_emd2'))(inputs)
    pos1=TimeDistributed(pos_embedding)(position1)
    pos2 = TimeDistributed(pos_embedding)(position2)
    x = concatenate([context_vector, pos1, pos2])

    lstm = TimeDistributed(Bidirectional(GRU(256, return_sequences=True, dropout=drop)))(x)
    print(lstm.shape)
    context_rep = Lambda(get_context, output_shape=get_context_output_shape)(lstm)
    # context_rep = lstm
    entity1_lstm = Lambda(entity_attention, output_shape=entity_attention_output_shape)([entity1_mask, lstm])
    entity2_lstm = Lambda(entity_attention, output_shape=entity_attention_output_shape)([entity2_mask, lstm])

    entity1_rep, entity2_rep = soft_attention_alignment(entity1_lstm, entity2_lstm)

    entity1_rep = GlobalMaxPooling1D()(entity1_rep)
    entity2_rep = GlobalMaxPooling1D()(entity2_rep)

    context_rep, entity1_rep, entity2_rep = skip(context_rep, entity1_rep, entity2_rep, k)
    context_rep = GlobalMaxPooling1D()(context_rep)

    res = concatenate([context_rep, entity1_rep, entity2_rep])
    res = Dense(64)(res)
    res = Dropout(drop)(res)

    out = Dense(1, activation='sigmoid', name='output')(res)
    model = Model([inputs, position1, position2, entity1_mask, entity2_mask], out)
    model.summary()
    adam = optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def run(model,x_train,y_train,x_develop,y_develop,x_test,y_test,train_pos1,dev_pos1,test_pos1,train_pos2,dev_pos2,test_pos2,
        train_entity1_pos, dev_entity1_pos, test_entity1_pos, train_entity2_pos, dev_entity2_pos, test_entity2_pos):
    clr = CyclicLR(base_lr=0.001, max_lr=0.002,
                   step_size=300., mode='exp_range',
                   gamma=0.99994)
    for i in range(10):
        logging.info('epoch'+str(i))
        # model.fit(x_train, y_train, epochs=1, batch_size=batch_size, callbacks=[clr, ])
        model.fit([x_train,train_pos1,train_pos2,train_entity1_pos,train_entity2_pos], y_train, epochs=1, batch_size=batch_size)
        pred_dev_y = model.predict([x_develop,dev_pos1,dev_pos2,dev_entity1_pos,dev_entity2_pos], batch_size=batch_size, verbose=0)
        # best_threshold, best_scores = threshold_search(y_develop,pred_dev_y)
        precision, recall, best_score = f1(y_develop, (pred_dev_y > 0.5).astype(int))
        print("Epoch: ", i, "-    dev F1 Score: {:.4f}".format(best_score))
        logging.info("Epoch: " + str(i) + "-    dev F1 Score: " + str(best_score))
        pred_test_y = model.predict([x_test, test_pos1, test_pos2, test_entity1_pos, test_entity2_pos],
                                    batch_size=batch_size, verbose=0)
        p, r, f = f1(y_test, (pred_test_y > 0.5).astype(int))
        print("Epoch: ", i, "-    Val F1 Score: {:.4f}".format(f))
        logging.info("Epoch: " + str(i) + "-    Val p Score:" + str(p))
        logging.info("Epoch: " + str(i) + "-    Val r Score:" + str(r))
        logging.info("Epoch: " + str(i) + "-    Val F1 Score:" + str(f))


if __name__ == "__main__":

    position1 = load_pkl('position1')
    position2 = load_pkl('position2')
    train_pos1 = position1[:9753]
    dev_pos1 = position1[9753:10837]
    test_pos1 = position1[10837:]
    train_pos2 = position2[:9753]
    dev_pos2 = position2[9753:10837]
    test_pos2 = position2[10837:]

    entity1_pos = load_pkl('entity1_pos')
    entity2_pos = load_pkl('entity2_pos')
    train_entity1_pos = entity1_pos[:9753]
    dev_entity1_pos = entity1_pos[9753:10837]
    test_entity1_pos = entity1_pos[10837:]
    train_entity2_pos = entity2_pos[:9753]
    dev_entity2_pos = entity2_pos[9753:10837]
    test_entity2_pos = entity2_pos[10837:]

    train_x = load_pkl('train_x')
    train_y = load_pkl('train_y')
    count(train_y)
    dev_x = load_pkl('dev_x')
    dev_y = load_pkl('dev_y')
    count(dev_y)
    test_x = load_pkl('test_x')
    test_y = load_pkl('tests_y')
    embedding_matrix = load_pkl('embedding_matrix')


    kk=[0]
    for k in kk:
        logging.info('basemodel+position+entity_mask_ESIM52_'+str(k))
        model = basemodel_position(0.3,embedding_matrix,200,23,k)
        run(model, train_x, train_y, dev_x, dev_y, test_x, test_y, train_pos1, dev_pos1, test_pos1, train_pos2, dev_pos2, test_pos2,
            train_entity1_pos, dev_entity1_pos, test_entity1_pos, train_entity2_pos, dev_entity2_pos, test_entity2_pos)







