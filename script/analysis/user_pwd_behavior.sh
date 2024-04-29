
INPUT=/disk/yjt/PersonalTarGuess/data/analysis/Collections1_100w.csv
OUTPUT=/disk/yjt/PersonalTarGuess/result/analysis/Collections1_user_pwd_behavior.json
CNT=1000000

python /disk/yjt/PersonalTarGuess/src/analysis/user_pwd_behavior.py \
-i ${INPUT} \
-o ${OUTPUT} \
-n ${CNT}
