import numpy as np

# filename = "../data_raw/cora/"
# filename = "../../../data_raw/reddit-small/"
filename="../../../data/pubmed/"
edgesFilename = filename + "edges.txt"
featsFilename = filename + "featsClass.txt"


edgesFile = open(edgesFilename, 'r')
featsClassFile = open(featsFilename, 'r')

line=edgesFile.readline()
count_edge=0
while line:
    count_edge=count_edge+1
    line=edgesFile.readline()

print("edges number:{0}".format(count_edge))


line_feat=featsClassFile.readline()
line_list=line_feat.split('\t')
print("feature number:{0}".format(len(line_list)-2))
classList=[]
classList.append(line_list[-1])
count_feat=0
while line_feat:
    count_feat=count_feat+1
    if not classList.count(line_list[-1]):
        classList.append(line_list[-1])
    line_feat=featsClassFile.readline()
    line_list=line_feat.split('\t')


print("vertex number:{0}".format(count_feat))

print("degree:{0}".format(count_edge/count_feat))
print("class number:{0}".format(len(classList)))


edgesFile.close()
featsClassFile.close()