class ObjectDict(dict):
    def __init__(self, *args, **kwargs):
        print('aaa')
        super(ObjectDict, self).__init__(*args, **kwargs)

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ObjectDict(value)
        return value


# 很简单但经典，可以像访问属性一样访问dict中的键值对。
if __name__ == '__main__':
    od = ObjectDict(asf={'a': 1}, d=True)
    print(od.asf, od.asf.a)  # {'a': 1} 1
    print(od.d)  # True
