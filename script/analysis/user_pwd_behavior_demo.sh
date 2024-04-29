
INPUT=/disk/yjt/PersonalTarGuess/data/analysis/4iQ_1w.csv
OUTPUT=/disk/yjt/PersonalTarGuess/result/analysis/4iQ_demo_user_pwd_behavior.json
CNT=10000

python /disk/yjt/PersonalTarGuess/src/analysis/user_pwd_behavior.py \
-i ${INPUT} \
-o ${OUTPUT} \
-n ${CNT}
