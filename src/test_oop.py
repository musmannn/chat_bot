class usman:
    def __init__(self):
        self.name = "Usman"
        self.age = 25
    def change_age(self,age):
        self.age = age
    def change_age_again(self,):
        print(f"age inside {self.age}")
        
        
        
person = usman()
print(person.age)
person.change_age(19)
person.change_age_again()
print(person.age)