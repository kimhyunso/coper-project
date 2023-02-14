import re

class RegularExpression:
    def __init__(self, pattern):
        self.pattern = re.compile(pattern) 
    def replace(self):
        self.pattern.sub()
    def replace_count(self):
        self.pattern.subn()
    def find_all(self):
        self.pattern.findall()

