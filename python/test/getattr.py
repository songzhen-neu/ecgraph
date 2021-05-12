class Animal(object):
    def __getattr__(self, item):
        return "tm"

    def __call__(self):
        print("aaaa")


class Cat(Animal):
    def __init__(self):
        self.name = "jn"


cat = Cat()
cat()
print(cat.name)
print(getattr(cat, 'name'))
print("*" * 20)
print(cat.age)
print(getattr(cat, 'age'))
