
#__name__ is only __main__ if run as script, not if run from a library.
if __name__ == '__main__':
    import sys

    print(sys.argv)

    _, num1, num2 = sys.argv

    print(int(num1) * int(num2))

def greet():
    print('Hi there!')
