class PriorityStack:

    def insert(self, value, priority):
        length = self.__len__()
        index = -1
        while True:
            index += 1
            if index > length - 1 or priority >= self.list[index][1]:
                break
        
        self.list = self.list[:index] + [(value, priority)] + self.list[index:]

    def remove(self):
        if self.is_empty():
            raise IndexError('PriorityStack is empty')
        return self.list.pop(0)

    def peek(self):
        if self.is_empty():
            raise IndexError('PriorityStack is empty')
        return self.list[0]

priority_stack = PriorityStack()
priority_stack.insert(1, 3)
priority_stack.insert(2, 2)
priority_stack.insert(4, 3)
