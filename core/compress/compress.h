//
// Created by songzhen on 2021/1/18.
//

#ifndef DGNN_TEST_COMPRESS_H
#define DGNN_TEST_COMPRESS_H
#include <vector>
#include <iostream>
using namespace std;

class Compress {
public:

    static uint oneByteCompress(vector<uint> fourItemsVec);
    static uint twoBitCompress(vector<uint> &fourItemsVec);
    static uint fourBitCompress(vector<uint> fourItemsVec);
    static uint eightBitCompress(vector<uint> fourItemsVec);
    static uint sixteenBitCompress(vector<uint> fourItemsVec);
    static int oneByteCompress_int(vector<int> fourItemsVec);
};


#endif //DGNN_TEST_COMPRESS_H
