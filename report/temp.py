# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#trying out classes

#make new class "Dog()"

class Dog():
    def __init__(self, name, age):
        self.name = name #these are attributes, they are not called with brackets
        self.age = age
    
    def get_name(self):
        return self.name # this is a method of the class Dog. By calling it, one gets the name of the dog
    
    def get_age(self):
        return self.age
    
    
d1 = Dog("Tim", 20) #make object d1 from class Dog, and assign attributes name and age

d1.get_name() #call the get_name method of the object from class Dog(), which returns the previously assigned name
