import random
if __name__ == '__main__':
    filename = '../../data/test/'
    edgesFileName = filename + 'edges.txt'
    featsClassFileName = filename + 'featsClass.txt'

    edgesFile = open(edgesFileName, 'w+')
    featsClassFile = open(featsClassFileName, 'w+')

    # edgesFile.write('0\t1\n')
    # edgesFile.write('1\t2\n')
    # edgesFile.write('2\t3\n')
    # edgesFile.write('3\t4\n')
    # edgesFile.write('1\t0\n')

    # insert nodes
    for i in range(36):
        str_node=str(i)
        for j in range(34):
            str_node+=('\t'+str(i))
        str_node+='\ta\n'
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
        edgesFile.write('0\t'+str(i)+'\n')
        edgesFile.write(str(i)+'\t'+'0\n')

    edgesFile.close()

    # featsClassFile.write('0\t1\t0\ta\n')
    # featsClassFile.write('1\t0\t1\ta\n')


    featsClassFile.close()
