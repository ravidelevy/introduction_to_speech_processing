import sys
import numpy as np

def print_p(p: float):
    print("%.3f" % p)

def compute_ctc(matrix, string, tokens):
    string = tokens[0] +\
             tokens[0].join(string[i:i+1] for i in range(len(string))) +\
             tokens[0]
    number_of_frames = matrix.shape[0]
    sequence_length = len(string)
    probabilites = np.zeros((number_of_frames, sequence_length))
    
    probabilites[0][0] = matrix[0][tokens.index(string[0])]
    probabilites[0][1] = matrix[0][tokens.index(string[1])]
    for i in range(1, number_of_frames):
        for j in range(sequence_length):
            if j == 0:
                probabilites[i][j] = probabilites[i - 1][j] * matrix[i][tokens.index(string[j])]
            elif j == 1 or string[j] == tokens[0] or string[j] == string[j - 2]:
                probabilites[i][j] = (probabilites[i - 1][j - 1] +
                                      probabilites[i - 1][j]) * matrix[i][tokens.index(string[j])]
            else:
                probabilites[i][j] = (probabilites[i - 1][j - 2] +
                                      probabilites[i - 1][j - 1] +
                                      probabilites[i - 1][j]) * matrix[i][tokens.index(string[j])]
    
    return probabilites[-1][-1] + probabilites[-1][-2]


def main():
    args = sys.argv[1:]
    matrix_path = args[0]
    matrix = np.load(matrix_path)
    string = args[1]
    tokens = args[2]
    print_p(compute_ctc(matrix, string, '@' + tokens))


if __name__ == '__main__':
    main()
