python -u main.py -d A/A > log1.txt 2>&1 &
python -u main.py -d B/B > log2.txt 2>&1 &
python -u main.py -d C/C > log3.txt 2>&1 &
echo "Running three parallel instances in the background"