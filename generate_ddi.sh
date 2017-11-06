python seq_wc.py --load_arg ./checkpoint/drugddi2011_we_2_cwlm_lstm_crf.json --load_check_point ./checkpoint/drugddi2011_we_2_cwlm_lstm_crf.model --gpu -1 --input_file ../data/drugddi2011/test.ddi --output_file ../data/drugddi2011/pred.ddi -keep_iobes

python seq_lstm.py --load_arg ./checkpoint/drugddi2011_0_lstm.json --load_check_point ./checkpoint/drugddi2011_0_lstm.model --gpu -1 --input_file ../data/drugddi2011/test.ddi --output_file ../data/drugddi2011/pred.ddi

python seq_w.py --load_arg ./checkpoint/drugddi2011_0_lstm_crf.json --load_check_point ./checkpoint/drugddi2011_0_lstm_crf.model --gpu -1 --input_file ../data/ddi_tiny/test.ddi --output_file ../data/ddi_tiny/output.txt
