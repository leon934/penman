class ExpressionParser:
    def __init__(self, expression_list: list) -> None:
        self.expression = expression_list
        self.operator_list = [['+', '-'], ['*', '/'], ['(', ')']]
        # Used ONLY for comparisons and searching.
        self.comparison_operator_list = ['+', '-', '*', '/', '(', ')']
        self.number_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.output_stack, self.operator_stack = [], []

    # These helper functions assume all inputted numbers are integers.
    def add(self, num1: str, num2: str):
        return int(num1) + int(num2)
    
    def subtract(self, num1: str, num2: str):
        return int(num1) - int(num2)
    
    def multiply(self, num1: str, num2: str):
        return int(num1) * int(num2)
    
    def divide(self, num1: str, num2: str):
        return int(num1) / int(num2)
    
    def combineIntegers(self):
        i = 0
        
        while i < len(self.expression) - 1:
            if self.expression[i + 1] in self.number_list:
                self.expression[i] += self.expression[i + 1]
                self.expression.pop(i + 1)
            else:
                i += 2

        print(self.expression)

    # TODO: Implement parenthesis and replace while loops that grab the number.
    def convertToPostfix(self):
        # First number SHOULD be an integer.
        for operator in self.comparison_operator_list:
            if self.expression[0] == operator:
                raise("First value should not be an operator.")
        
        self.output_stack.append(self.expression.pop(0))

        for i in range(len(self.expression)):
            # First 
            if i == 0:
                self.operator_stack.append(self.expression.pop(0))
            else:
                for i, operator in enumerate(self.comparison_operator_list):
                    if self.operator_stack[len(self.operator_stack) - 1] == operator:
                        curr_operator_order = i

                    if self.expression[0] == operator:
                        next_operator_order = i
                
                # If the following operator has a higher precedence...
                if next_operator_order > curr_operator_order:
                    self.operator_stack.append(self.expression.pop(0))
                # Otherwise remove the last element from the operator stack and add it to the output stack.
                else:
                    self.output_stack.append(self.operator_stack.pop(len(self.operator_stack) - 1))
                    self.operator_stack.append(self.expression.pop(0))

            
            self.output_stack.append(self.expression.pop(0))

            if self.expression == []:
                break

        for i in range(len(self.operator_stack)):
            self.output_stack.append(self.operator_stack.pop())

        print(self.output_stack)

    def solvePostfix(self):
        i = 0

        if len(self.output_stack) == 2:
            raise("Error in parsing expression. Output stack should not have only two elements.")

        while len(self.output_stack) != 1:
            try:
                int(self.output_stack[i])
                int(self.output_stack[i + 1])

                if self.output_stack[i + 2] not in self.comparison_operator_list:
                    raise ValueError()
                
                operator = self.output_stack[i + 2]

                match operator:
                    case '+':
                        self.output_stack[i] = self.add(self.output_stack[i], self.output_stack[i + 1])
                    case '-':
                        self.output_stack[i] = self.subtract(self.output_stack[i], self.output_stack[i + 1])
                    case '*':
                        self.output_stack[i] = self.multiply(self.output_stack[i], self.output_stack[i + 1])
                    case '/':
                        self.output_stack[i] = self.divide(self.output_stack[i], self.output_stack[i + 1])
                    
                self.output_stack.pop(i + 1)
                self.output_stack.pop(i + 1)
                
                i -= 1

            except ValueError:
                if self.output_stack[i + 2] not in self.comparison_operator_list:
                    i += 1

                if self.output_stack[i + 1] in self.comparison_operator_list:
                    i -= 1

        return self.output_stack[0]

eq1 = ExpressionParser(['1', '*', '3'])
eq1.combineIntegers()
eq1.convertToPostfix()
print(eq1.solvePostfix())