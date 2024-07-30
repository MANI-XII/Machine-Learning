#!/usr/bin/env python
# coding: utf-8

# In[6]:


def count_vowels_and_consonants(input_string):
   
    vowels = "aeiouAEIOU"
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"

   
    vowel_count = 0
    consonant_count = 0

   
    for char in input_string:
        if char in vowels:
            vowel_count += 1
        elif char in consonants:
            consonant_count += 1

    return vowel_count, consonant_count

def main():
    
    user_input = input("Enter a string: ")

    vowels, consonants = count_vowels_and_consonants(user_input)

    print(f"Number of vowels: {vowels}")
    print(f"Number of consonants: {consonants}")

if __name__ == "__main__":
    main()


# In[2]:


def matrix_multiply(A, B):
    """Multiplies two matrices A and B."""
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

def main():
   
    rows_A = int(input("Enter the number of rows for matrix A: "))
    cols_A = int(input("Enter the number of columns for matrix A: "))
    
    print("Enter the elements of matrix A row by row (space-separated):")
    matrix_A = []
    for i in range(rows_A):
        row = list(map(float, input().split()))
        if len(row) != cols_A:
            print("Error: Each row must have exactly", cols_A, "elements.")
            return
        matrix_A.append(row)

    
    rows_B = int(input("Enter the number of rows for matrix B: "))
    cols_B = int(input("Enter the number of columns for matrix B: "))
    
    print("Enter the elements of matrix B row by row (space-separated):")
    matrix_B = []
    for i in range(rows_B):
        row = list(map(float, input().split()))
        if len(row) != cols_B:
            print("Error: Each row must have exactly", cols_B, "elements.")
            return
        matrix_B.append(row)

   
    if cols_A != rows_B:
        print("Error: Matrices cannot be multiplied. Number of columns in A must equal number of rows in B.")
        return

    result_matrix = matrix_multiply(matrix_A, matrix_B)

    
    print("The product of matrix A and matrix B is:")
    for row in result_matrix:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    main()


# In[3]:


def count_common_elements(list1, list2):
    """Counts the number of common elements between two lists."""
    common_count = 0
    common_elements = []  

   
    for element in list1:
        
        if element in list2 and element not in common_elements:
            common_count += 1
            common_elements.append(element)

    return common_count

def main():

    user_input1 = input("Enter the first list of integers separated by spaces: ")
    list1 = [int(num) for num in user_input1.split()]

    user_input2 = input("Enter the second list of integers separated by spaces: ")
    list2 = [int(num) for num in user_input2.split()]

    
    result = count_common_elements(list1, list2)

    
    print(f"Number of common elements: {result}")

if __name__ == "__main__":
    main()


# In[5]:


def transpose_matrix(matrix):
    """Transposes a given matrix."""
    rows = len(matrix)
    cols = len(matrix[0])

    
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]

   
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed

def main():
   
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

   
    print("Enter the matrix elements row by row (space-separated):")
    matrix = []
    for i in range(rows):
        row = list(map(float, input().split()))
        if len(row) != cols:
            print("Error: Each row must have exactly", cols, "elements.")
            return
        matrix.append(row)

   
    transposed_matrix = transpose_matrix(matrix)

   
    print("The transposed matrix is:")
    for row in transposed_matrix:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    main()


# In[ ]:




