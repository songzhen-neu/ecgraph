class Human(object):
    def __init__(self, sex, name):
        self.sex = sex
        self.name = name

    def __getattr__(self, item):
        print("this is Human getattr")

    def __call__(self,*input):
        print('aaa')


class Child(Human):
    def __init__(self, name, sex, school):
        super().__init__(sex,name)
        self.school = school

    # def __call__(self,):
    #     print("this is __call__")


a = Child('Tom', 'male', 'shiyan')
print(a.weight)
a('aa',1)
# print(a.school)
