python eval_w.py --load_arg ./checkpoint/drugddi2011_1_lstm_crf.json --load_check_point ./checkpoint/drugddi2011_1_lstm_crf.model --gpu -1 --dev_file ../data/drugddi2011/test.ddi --test_file ../data/drugddi2011/test.ddi

python eval_wc.py --load_arg ./checkpoint/drugddi2011_we_cwlm_lstm_crf.json --load_check_point ./checkpoint/drugddi2011_we_cwlm_lstm_crf.model --gpu -1 --dev_file ../data/drugddi2011/test.ddi --test_file ../data/drugddi2011/test.ddi

python eval_wc.py --load_arg ./checkpoint/drugddi2013_we_cwlm_lstm_crf.json --load_check_point ./checkpoint/drugddi2013_we_cwlm_lstm_crf.model --gpu -1 --dev_file ../data/drugddi2013/test.ddi --test_file ../data/drugddi2013/test.ddi

python eval_wc.py --load_arg ./checkpoint/drugddi2011_we_cwlm_lstm_crf.json --load_check_point ./checkpoint/drugddi2011_we_cwlm_lstm_crf.model --gpu -1 --dev_file ../data/ddi_tiny/train.ddi --test_file ../data/ddi_tiny/train.ddi

python seq_wc.py --load_arg ./checkpoint/drugddi2011_we_cwlm_lstm_crf.json --load_check_point ./checkpoint/drugddi2011_we_cwlm_lstm_crf.model --gpu -1 --input_file ../data/drugddi2011/test.ddi --output_file ../data/drugddi2011/pred.ddi

