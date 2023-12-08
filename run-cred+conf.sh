# run all cred experiments

# these experiments produce plots for CCE, TCE, ICE
# p-vals using cred+conf

bash experiments/tce-ice-cce-cred+conf/tce.sh &&
bash experiments/tce-ice-cce-cred+conf/ice.sh &&
bash experiments/tce-ice-cce-cred+conf/cce.sh
