python3 ice.py                                      \
	--train out/12_month_train                      \
	--test out/12_month_test_monthly                \
	-k 10                                           \
	-n -1                                           \
	--pval-consider cal-only                        \
	-t constrained-search                           \
	-c cred                                         \
	--cs-max f1_k:0.90                              \
	--cs-con kept_pos_perc:0.85,kept_neg_perc:0.85  \
	--rs-samples 100000


python3 ice.py                                      \
	--dataset extended-features                     \
	-k 10                                           \
	-n -1                                           \
	--pval-consider cal-only                        \
	-t constrained-search                           \
	-c cred                                         \
	--cs-max f1_k:0.90                              \
	--cs-con kept_pos_perc:0.85,kept_neg_perc:0.85  \
	--rs-samples 100000