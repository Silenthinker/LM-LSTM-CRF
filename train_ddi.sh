python train_w.py --data_path ../data/drugddi2011 --train_file train.ddi --dev_file val.ddi --test_file test.ddi --checkpoint ./checkpoint/drugddi2011_0 --gpu -1 --caseless --fine_tune --small_crf --emb_file ../data/word_embedding/glove/glove.6B.100d.txt --embedding_dim 100 --load_check_point ./checkpoint/drugddi2011_0_lstm_crf.model


python train_lstm.py --data_path ../data/drugddi2011 --train_file train.ddi --dev_file val.ddi --test_file test.ddi --checkpoint ./checkpoint/drugddi2011_0 --gpu -1 --caseless --fine_tune --emb_file ../data/word_embedding/glove/glove.6B.100d.txt --embedding_dim 100
