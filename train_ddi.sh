python train_wc.py --data_path ../data/DrugDDI --train_file train.ddi --dev_file test.ddi --test_file test.ddi --checkpoint ./checkpoint/drugddi --gpu -1 --caseless --fine_tune --high_way --co_train --rand_embedding

python train_w.py --data_path ../data/DrugDDI --train_file train.ddi --dev_file test.ddi --test_file test.ddi --checkpoint ./checkpoint/drugddi --gpu -1 --caseless --fine_tune --rand_embedding
