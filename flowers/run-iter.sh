#run command for training and testing
# Usage: ./run-iter.sh [iter]

i=0
while [ $i -lt $1 ]
do
    echo "!!!!!!!!!!!!!!!!!!Iteration `expr $i + 1`!!!!!!!!!!!!!!!!!!"
    python3 ./flowers.py
    i=`expr $i + 1`
done