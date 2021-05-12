//
// Created by songzhen on 2020/10/8.
//
#include <string>
using namespace std;
class Animal{
public:
    virtual ~Animal(){}
    virtual std::string go(int n_times)=0;
};

class Dog: public Animal{
public:
    std::string go(int n_times) override{
        std::string result;
        for(int i=0;i<n_times;i++){
            result.append("woof!");
        }
        return result;
    }
};

std::string call_go(Animal *animal){
    return animal->go(3);
}