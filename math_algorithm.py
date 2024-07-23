


def stringToEquation(stringArray: list) -> int:
    value = ""
    numArray = []
    operator = ""

    for i in range(len(stringArray)):
        string = stringArray.pop(0)

        try:
            int(string)
            value += string

        except(ValueError):
            numArray.append(value)
            value = ""

            if len(numArray) == 2: # Does the operation on two numbers once they exist.
                match operator:
                    case '+':
                        print("+")
                        numArray[0] = str(float(numArray[0]) + float(numArray[1]))
                    case '-':
                        print("-")
                        numArray[0] = str(float(numArray[0]) - float(numArray[1]))
                    case '*': # Figure this out later. (), *, \cdot, etc.
                        numArray[0] = str(float(numArray[0]) * float(numArray[1]))
                    case '/':
                        numArray[0] = str(float(numArray[0]) / float(numArray[1]))

                del numArray[1]

            operator = string
    

    print(numArray)
    
# def equationSolve(numberArray: list, operatorArray : list):
#     print
    
    
    

stringToEquation(['2', '3', '+', '3', '5', '-', '6'])