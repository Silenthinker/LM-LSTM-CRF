python train_wc.py --data_path ../data/drugddi2013 --train_file train.ddi --dev_file test.ddi --test_file test.ddi --checkpoint ./checkpoint/drugddi2013_we --gpu -1 --caseless --fine_tune --high_way --co_train --emb_file ../data/word_embedding/glove.6B/glove.6B.100d.txt --word_dim 100

python train_w.py --data_path ../data/drugddi2013 --train_file train.ddi --dev_file test.ddi --test_file test.ddi --checkpoint ./checkpoint/drugddi2013 --gpu -1 --caseless --fine_tune --rand_embedding
