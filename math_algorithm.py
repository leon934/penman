numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def stringToEquation(stringArray: list) -> int:
    numArray = []

    if len(stringArray) == 0:
        raise Exception("No equation found.")

    # Grabs the operator, and the first and second digits.
    numArray.append(getNumber(stringArray))
    operator = stringArray.pop(0)
    numArray.append(getNumber(stringArray))

    if numArray[1] == '':
        raise Exception("Incomplete equation.")

    # Calculates the values given the numbers and operators.
    while len(numArray) == 2:
        match operator:
            case '+':
                numArray[0] = str(float(numArray[0]) + float(numArray[1]))
            case '-':
                numArray[0] = str(float(numArray[0]) - float(numArray[1]))
            case '*': # FIXME: Figure this out later. (), *, \cdot, etc.
                numArray[0] = str(float(numArray[0]) * float(numArray[1]))
            case '/':
                numArray[0] = str(float(numArray[0]) / float(numArray[1]))

        del numArray[1]

        if len(stringArray) > 0: # If there's still values in the stringArray, continue calculating.
            operator = stringArray.pop(0)
            numArray.append(getNumber(stringArray))
    
    return numArray[0]

# Finds the numbers up to the first operato or end.
def getNumber(stringArray: list) -> int:
    value = ""

    while len(stringArray) != 0 and stringArray[0] in numbers:
        value += stringArray.pop(0)

    return value

# These are the test cases I used; feel free to try more.

# print(stringToEquation([]))
# print(stringToEquation(['2', '-']))
# print(stringToEquation(['2', '3', '+', '3', '5']))
# print(stringToEquation(['2', '3', '+', '3', '5', '-', '6']))