python eval_w.py --load_arg ./checkpoint/drugddi_lstm_crf.json --load_check_point ./checkpoint/drugddi_lstm_crf.model --gpu -1 --dev_file ../data/DrugDDI/test.ddi --test_file ../data/DrugDDI/test.ddi

python eval_wc.py --load_arg ./checkpoint/drugddi2011_we_cwlm_lstm_crf.json --load_check_point ./checkpoint/drugddi2011_we_cwlm_lstm_crf.model --gpu -1 --dev_file ../data/drugddi2011/test.ddi --test_file ../data/drugddi2011/test.ddi

python eval_wc.py --load_arg ./checkpoint/drugddi2013_we_cwlm_lstm_crf.json --load_check_point ./checkpoint/drugddi2013_we_cwlm_lstm_crf.model --gpu -1 --dev_file ../data/drugddi2013/test.ddi --test_file ../data/drugddi2013/test.ddi

python eval_wc.py --load_arg ./checkpoint/drugddi2011_we_cwlm_lstm_crf.json --load_check_point ./checkpoint/drugddi2011_we_cwlm_lstm_crf.model --gpu -1 --dev_file ../data/ddi_tiny/test.ddi --test_file ../data/ddi_tiny/test.ddi

