class A:
    def __init__(self):
        self.a = 1
    def printa(self):
        print("a=", self.a)

class B:
    def __init__(self):
        self.b = 2
        self.A_in_B = A()
        self.A_in_B.a = 3
    def printa(self):
        print("a=", self.A_in_B.a)
    def set_a(self,num):
        self.A_in_B.a=num

if __name__ == '__main__':
    ins_A = A()
    ins_A.printa()
    ins_B = B()
    ins_B.printa()
    ins_B.set_a(6)
    ins_B.printa()