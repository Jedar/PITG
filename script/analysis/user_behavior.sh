
INPUT=/disk/yjt/PersonalTarGuess/data/analysis/4iQ_100w.csv
OUTPUT=/disk/yjt/PersonalTarGuess/result/analysis/4iQ_user_edit_behavior.json
CNT=1000000

python /disk/yjt/PersonalTarGuess/src/analysis/user_edit_behavior.py \
-i ${INPUT} \
-o ${OUTPUT} \
-n ${CNT}
