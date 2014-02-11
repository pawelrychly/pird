import os
import yaml
from subprocess import Popen, PIPE
import sys
import time

out = open("tmp.arff", "w+")

out.write("@relation tmp\n")

for i in range(13):
    out.write("@attribute min"  + str(i) + " numeric\n")
for i in range(13):
    out.write("@attribute max"  + str(i) + " numeric\n")
for i in range(13):
    out.write("@attribute mean"  + str(i) + " numeric\n")
for i in range(13):
    out.write("@attribute var"  + str(i) + " numeric\n")
for i in range(13):
    out.write("@attribute max_f_auto"  + str(i) + " numeric\n")
for i in range(13):
    out.write("@attribute max_f"  + str(i) + " numeric\n")
for i in range(13):
    out.write("@attribute argmax_f" + str(i) + " numeric\n")
for i in range(13):
    out.write("@attribute sfm"  + str(i) + " numeric\n")

out.write("@attribute dec {classical,hiphop,reggae,country,jazz,metal,disco,pop,rock,blues}\n")

out.write("@data\n")
#counter = 0
#for root, dirs, files in os.walk("/hdd/gtzan/genres/"):
#   for f in files:
#        print counter
#        counter += 1
#        cat =  root[root.rfind("/")+1:]
#        if cat == "metal" or cat == "jazz" or cat == "country" or cat == "classical" or cat == "reggae" or cat == "hiphop" or cat == "disco" or cat == "rock" or cat == "pop":
#            continue
#        fpath = os.path.join(root, f)
#        process = Popen(['python','/home/pawel/PycharmProjects/mir/standard_mfcc-extended.py', fpath, 'y.yaml'], stdout=PIPE)
#        stdout, stderr = process.communicate()
#        print stdout, stderr

process = Popen(['python','/home/dawid/Pulpit/standard_mfcc-extended_tempo.py', sys.argv[1], 'y.yaml'], stdout=PIPE)
stdout, stderr = process.communicate()
#print stdout
#print stderr

cat =  sys.argv[1][sys.argv[1].rfind("/")+1:]
cat = cat[:cat.find(".")]
if cat not in ["classical","hiphop","reggae","country","jazz","metal","disco","pop","rock","blues"]:
    cat = "classical"

with open('y.yaml', 'r') as f:
    doc = yaml.load(f)
    max_f_auto = doc["lowlevel"]["mfcc"]["max_f_auto"][0]
    max_f = doc["lowlevel"]["mfcc"]["max_f"][0]
    sfm = doc["lowlevel"]["mfcc"]["sfm"][0]
    argmax = doc["lowlevel"]["mfcc"]["argmaxf_auto"]
    #print max_f_auto
    #print max_f
    #print sfm

    minimum = doc["lowlevel"]["mfcc"]["min"]
    maximum = doc["lowlevel"]["mfcc"]["max"]
    mean = doc["lowlevel"]["mfcc"]["mean"]
    var = doc["lowlevel"]["mfcc"]["var"]

    for i in minimum:
        out.write(str(i) + ",")
    for i in maximum:
        out.write(str(i) + ",")
    for i in mean:
        out.write(str(i) + ",")
    for i in var:
        out.write(str(i) + ",")
    for i in max_f_auto:
        out.write(str(i) + ",")
    for i in max_f:
        out.write(str(i) + ",")
    for i in argmax:
        out.write(str(i) + ",")
    for i in sfm:
        out.write(str(i) + ",")


    out.write(cat + "\n")
    os.system('rm y.yaml')

out.close()
print "Datas extracted"
time.sleep(1)

process = Popen(['java','-cp', '/home/dawid/Pobrane/weka-3-7-10/weka.jar', 'weka.classifiers.meta.Bagging', '-T', '/home/dawid/Pulpit/tmp.arff', "-l", sys.argv[2]], stdout=PIPE)
stdout, stderr = process.communicate()
#print stdout
#print stderr

result = stdout.split("\n")[-12].strip().split()
print "CLASSIFIER DECISSION IS: "
print "*******************************************************"
print "\t\t\t", 
for i in range(10):
    if result[i] == "1":
        print ["classical","hiphop","reggae","country","jazz","metal","disco","pop","rock","blues"][i]
print "*******************************************************"