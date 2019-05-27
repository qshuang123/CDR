
from model.basemodel import base_model, run
from model.double_lstm import double_lstm
from process.Reader import Reader
from utils.threshold import load_pkl
import logging
logging.basicConfig(level=logging.INFO, filename='mylog.log')

if __name__ == "__main__":

    myreader = Reader()
    train_instances, train_y = myreader.read_file(name='train')
    dev_instances, dev_y = myreader.read_file(name='dev')
    test_instances, tests_y = myreader.read_file(name='test')

    train_instances, train_y, dev_instances, dev_y = myreader.resign_dev(train_instances, dev_instances, train_y,
                                                                         dev_y)  # 20%

    train_y = myreader.reshape_y(train_y)
    dev_y = myreader.reshape_y(dev_y)
    tests_y = myreader.reshape_y(tests_y)

    train_all_doc, train_max_sentence = myreader.divide_sentence(train_instances)
    dev_all_doc, dev_max_sentence = myreader.divide_sentence(dev_instances)
    test_all_doc, test_max_sentence = myreader.divide_sentence(test_instances)

    maxsentence = max(train_max_sentence, dev_max_sentence, test_max_sentence)
    print(train_max_sentence)
    print(dev_max_sentence)
    print(test_max_sentence)
    maxsentence_list = [58]
    maxlen_list = [100]

    for maxsentence in maxsentence_list:
        for maxlen in maxlen_list:
            logging.info('***********************************************************************************************')
            logging.info('maxsentence'+str(maxsentence))
            logging.info('maxlen' + str(maxlen))
            train_x, dev_x, test_x, tokenizer, wordvec = myreader.tokenizer(train_all_doc, dev_all_doc, test_all_doc,
                                                                            maxsentence, maxlen)
            myreader.save_file('train_x', train_x)
            myreader.save_file('dev_x', dev_x)
            myreader.save_file('test_x', test_x)
            myreader.save_file('train_y', train_y)
            myreader.save_file('dev_y', dev_y)
            myreader.save_file('tests_y', tests_y)

            print('finish')

            train_x = load_pkl('train_x')
            train_y = load_pkl('train_y')

            dev_x = load_pkl('dev_x')
            dev_y = load_pkl('dev_y')

            test_x = load_pkl('test_x')
            test_y = load_pkl('tests_y')
            embedding_matrix = load_pkl('embedding_matrix')
            logging.info('basemodel')
            drop = [0.3]
            for i in drop:
                logging.info('drop' + str(i))
                model = base_model(i,embedding_matrix,max_sentence_length=maxlen,max_sentence_number=maxsentence)
                run(model, train_x, train_y, dev_x, dev_y, test_x, test_y)

            logging.info('double_lstm_model')
            drop = [0.3, 0.4, 0.5, 0.6, 0.7]
            for i in drop:
                logging.info('drop' + str(i))
                model = double_lstm(i, embedding_matrix, max_sentence_length=maxlen, max_sentence_number=maxsentence)
                run(model, train_x, train_y, dev_x, dev_y, test_x, test_y)

