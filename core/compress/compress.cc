//
// Created by songzhen on 2021/1/18.
//

#include "compress.h"
uint Compress::oneByteCompress(vector<uint> ItemsVec) {

    ItemsVec[0] = ItemsVec[0] << 24;
    ItemsVec[1] = ItemsVec[1] << 16;
    ItemsVec[2] = ItemsVec[2] << 8;
//                        fourItemsVec[3]=fourItemsVec[3];
    uint compress_value = ItemsVec[0] |ItemsVec[1] | ItemsVec[2] | ItemsVec[3];
    return compress_value;
}


int Compress::oneByteCompress_int(vector<int> ItemsVec) {

    ItemsVec[0] = ItemsVec[0] << 24;
    ItemsVec[1] = ItemsVec[1] << 16;
    ItemsVec[2] = ItemsVec[2] << 8;
//                        fourItemsVec[3]=fourItemsVec[3];
    int compress_value = ItemsVec[0] | ItemsVec[1] | ItemsVec[2] | ItemsVec[3];
    return compress_value;
}

uint Compress::twoBitCompress(vector<uint> &ItemsVec) {
    uint compress_value=0;
    for(int i=0;i<16;i++){
//        ItemsVec[i]=ItemsVec[i]<<(32-(i+1)*2);
        compress_value=compress_value|(ItemsVec[i]<<(32-(i+1)*2));
//        compress_value=compress_value|ItemsVec[i];
    }
    return compress_value;

}



uint Compress::fourBitCompress(vector<uint> ItemsVec) {
    uint compress_value=0;
    for(int i=0;i<8;i++){
//        ItemsVec[i]=ItemsVec[i]<<(32-(i+1)*4);
        compress_value=compress_value|(ItemsVec[i]<<(32-(i+1)*4));
    }
    return compress_value;

}

uint Compress::eightBitCompress(vector<uint> ItemsVec) {
    uint compress_value=0;
    for(int i=0;i<4;i++){
//        ItemsVec[i]=ItemsVec[i]<<(32-(i+1)*8);
        compress_value=compress_value|(ItemsVec[i]<<(32-(i+1)*8));
    }
    return compress_value;

}

uint Compress::sixteenBitCompress(vector<uint> ItemsVec) {
    uint compress_value=0;
    for(int i=0;i<2;i++){
//        ItemsVec[i]=ItemsVec[i]<<(32-(i+1)*16);
        compress_value=compress_value|(ItemsVec[i]<<(32-(i+1)*16));
    }
    return compress_value;

}