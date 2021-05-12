from collections import defaultdict

dict1 = defaultdict(int)
dict2 = defaultdict(set)
dict3 = defaultdict(str)
dict4 = defaultdict(list)

dict1[1] = 'aaa'

dict5 = dict()
dict5['name'] = 'songzhen'
dict5['age'] = 12

print(dict1[1])
print(dict2[1])
print(dict3[1])
print(dict4[1])

print(dict5['name'])
