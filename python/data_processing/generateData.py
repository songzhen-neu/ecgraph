import random
if __name__ == '__main__':
    filename = '/mnt/data/test/'
    edgesFileName = filename + 'edges_raw.txt'
    featsClassFileName = filename + 'featsClass_raw.txt'

    edgesFile = open(edgesFileName, 'w+')
    featsClassFile = open(featsClassFileName, 'w+')

    # edgesFile.write('0\t1\n')
    # edgesFile.write('1\t2\n')
    # edgesFile.write('2\t3\n')
    # edgesFile.write('3\t4\n')
    # edgesFile.write('1\t0\n')
    random.seed(1)
    # insert nodes
    for i in range(36):
        str_node=str(i)
        for j in range(34):
            str_node+=('\t'+str(random.randint(-10,10)))
        label=random.randint(0,1)
        str_node+='\t'+str(label)+'\n'
        featsClassFile.write(str_node)
    edgeSet=set()

    # for i in range(20):
    #     for j in range(1):
    #         rand_i=random.randint(0,19)
    #         while rand_i==i:
    #             rand_i=random.randint(0,19)
    #         if edgeSet.__contains__(str(i)+','+str(rand_i)):
    #             continue
    #         else:
    #             edgesFile.write(str(i)+'\t'+str(rand_i)+'\n')
    #             edgesFile.write(str(rand_i)+'\t'+str(i)+'\n')
    #             edgeSet.add(str(rand_i)+','+str(i))


    for i in range(36):
        for j in range(2):
            nid=random.randint(0,35)
            edgesFile.write(str(nid)+'\t'+str(i)+'\n')
            edgesFile.write(str(i)+'\t'+str(nid)+'\n')

    edgesFile.close()

    # featsClassFile.write('0\t1\t0\ta\n')
    # featsClassFile.write('1\t0\t1\ta\n')


    featsClassFile.close()
