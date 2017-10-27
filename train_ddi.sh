python train_wc.py --data_path ../data/drugddi2011 --train_file train.ddi --dev_file val.ddi --test_file test.ddi --checkpoint ./checkpoint/drugddi2011_we_2 --gpu -1 --caseless --fine_tune --high_way --co_train --emb_file ../data/word_embedding/glove/glove.6B.100d.txt --word_dim 100

python train_w.py --data_path ../data/drugddi2011 --train_file train.ddi --dev_file val.ddi --test_file test.ddi --checkpoint ./checkpoint/drugddi2011_0 --gpu -1 --caseless --fine_tune --small_crf --emb_file ../data/word_embedding/glove/glove.6B.100d.txt --embedding_dim 100
