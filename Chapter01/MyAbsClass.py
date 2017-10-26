class MyAbsClass:
    number = 20
    name = "John Rambo"

    def __init__(self, number=10):
        self.real = number

    def absolute_value(self, x):
        if x >= 0:
            return x
        else:
            return -x

obj = MyAbsClass()
value = obj.absolute_value(-10)
print("The absolute value of -10 is: "+ str(value))
print(obj.number)
print(obj.name)