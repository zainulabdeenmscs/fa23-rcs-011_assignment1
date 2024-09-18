#!/usr/bin/env python
# coding: utf-8

# ## Roll No. FA23-RCS-011
# ## Zain Ul Abdeen
# ## Assignment 1 Solution

# ### Task 1: 
# Lists, Dictionaries, Tuples 
# ### 1.1.  Create a list: nums = [3, 5, 7, 8, 12], make another list named ‘cubes’ and append the cubes of the given list ‘nums’ in this list and print it.

# In[6]:


# Given list
nums = [3, 5, 7, 8, 12]

# Creating an empty list to store cubes
cubes = []

# Appending cubes of the elements in nums to cubes list
for num in nums:
    cubes.append(num ** 3)

# Printing the cubes list
print(cubes)


# ### 1.2.  Create an empty dictionary: dict = {}, add the following data to the dictionary: ‘parrot’: 2, ‘goat’: 4, ‘spider’: 8, ‘crab’: 10 as key value pairs.

# In[7]:


# Creating an empty dictionary
my_dict = {}

# Adding key-value pairs to the dictionary
my_dict['parrot'] = 2
my_dict['goat'] = 4
my_dict['spider'] = 8
my_dict['crab'] = 10

# Printing the dictionary
print(my_dict)


# ### 1.3.  Use the ‘items’ method to loop over the dictionary (dict) and print the animals and their corresponding legs. Sum the legs of each animal, and print the total at the end.

# In[8]:


# Initializing a variable to store the total number of legs
total_legs = 0

# Looping over the dictionary using the items method
for animal, legs in my_dict.items():
    print(f"{animal} has {legs} legs")
    total_legs += legs  # Add the legs to the total

# Printing the total number of legs
print(f"Total number of legs are: {total_legs}")


# ### 1.4.  Create a tuple: A = (3, 9, 4, [5, 6]), change the value in the list from ‘5’ to ‘8’. 

# In[22]:


# Creating the tuple
A = (3, 9, 4, [5, 6])

# Changing the value in the list from 5 to 8
A[3][0] = 8

# Printing the updated tuple
print(A)


# ### 1.5.  Delete the tuple A

# In[23]:


# Deleting the tuple A
del A

# If we insert code print (A) it will result in an error showing that tuple A has been deleted
# It also returns an error after deletion of tuple A if we re-run this code
# So 1.4 solution is run again to create tuple A and then this code to delete it and remove error


# ### 1.6.  Create another tuple: B = (‘a’, ‘p’, ‘p’, ‘l’, ‘e’), print the number of occurrences of ‘p’ in the tuple. 

# In[24]:


# Creating the tuple
B = ('a', 'p', 'p', 'l', 'e')

# Counting the occurrences of 'p'
count_p = B.count('p')

# Printing the count
print(f"The letter 'p' occurs {count_p} times in the tuple.")


# ### 1.7.  Print the index of ‘l’ in the tuple. 

# In[8]:


# Printing the index of 'l' in the tuple
index_l = B.index('l')

# Printing the index
print(f"The letter 'l' is at index {index_l} in the tuple.")


# ### Task 2: 
# Numpy Use built-in functions of numpy library to complete this task. List of functions available here (https://numpy.org/doc/1.19/genindex.html)  
# A = 1 2 3 4 
#     5 6 7 8 
#     9 10 11 12
#  
# z = np.array([1, 0, 1])   
# 
# ### 2.1  Convert matrix A into numpy array

# In[39]:


import numpy as np

# Defining matrix A
A = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]]



# Converting matrix A into a NumPy array
A_array = np.array(A)
# Printing shape for 2.3 solution
print(A_array.shape)

# Printing the NumPy array
print(A_array)


# ### 2.2  Use slicing to pull out the subarray consisting of the first 2 rows and columns 1 and 2. Store it in b which is a numpy array of shape (2, 2).

# In[26]:


# Slicing the first 2 rows and columns 1 and 2 from A_array
b = A_array[:2, :2]

# Printing the subarray b
print(b)
print(f'Shape of numpy Array is: {b.shape}')


# ### 2.3  Create an empty matrix ‘C’ with the same shape as ‘A’.

# In[29]:


# Creating an empty matrix with shape 3x4
c = np.empty((3, 4))

print(c.shape)


# ### 2.4 Add the vector z to each column of the matrix ‘A’ with an explicit loop and store it in ‘C’.
# #### Create the following: X = np.array([[1,2],[3,4]]) Y = np.array([[5,6],[7,8]]) v = np.array([9,10]) 

# In[91]:


z = np.array([1, 0, 1])

# Adding vector z to each column of matrix A using an explicit loop
for i in range(A_array.shape[1]):  # Loop over the columns
    c[:, i] = A_array[:, i] + z  # Adding z to each column

print("Matrix A:\n", A_array)
print("\nVector z:", z.shape)
print("\nResulting matrix C:\n", c)


# In[40]:


# Defining the matrices and vector
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])
v = np.array([9, 10])


# ### 2.5  Add and print the matrices X and Y.

# In[49]:


# Adding and printing the matrices X and Y
sum_X_Y = X + Y

print(f"Matrix X + Matrix Y: \n{sum_X_Y}")


# ### 2.6  Multiply and print the matrices X and Y. 

# In[50]:


# Multiplying and printing the matrices X and Y
multiply_X_Y = X * Y
print(f"Multiplication of Matrix X and Matrix Y: \n{multiply_X_Y}")


# ### 2.7  Compute and print the element wise square root of matrix Y. 

# In[44]:


# Computing and printing the element-wise square root of matrix Y
elementwise_sqrt = np.sqrt(Y)
print(f"Element-wise square root of Matrix Y:\n {elementwise_sqrt}")


# ### 2.8  Compute and print the dot product of the matrix X and vector v.

# In[46]:


# Computing and printing the dot product of the matrix X and vector v
dot_product = np.dot(X, v)
print(f"Dot product of Matrix X and Vector v: \n {dot_product}")


# ### 2.9  Compute and print the sum of each column of X.

# In[51]:


# Computing and printing the sum of each column of X
column_sum = np.sum(X, axis=0)
print(f"Sum of each column of Matrix X:\n{column_sum} ")


# ### Task 3: Functions and Loops 3.1  Create a function ‘Compute’ that takes two arguments, distance and time, and use it to calculate velocity. 

# In[52]:


def Compute(distance, time):
    # Calculating velocity
    velocity = distance / time
    return velocity

# Examplifying
distance = 100  # Example distance in meters
time = 20       # Example time in seconds
velocity = Compute(distance, time)

# Print the result
print(f"The velocity is {velocity} meters per second.")


# ### 3.2 Make a list named ‘even_num’ that contains all even numbers up till 12. Create a function ‘mult’ that takes the list ‘even_num’  as an argument and calculates the products of all entries using a for loop.

# In[53]:


# Creating the list of even numbers up till 12
even_num = [2, 4, 6, 8, 10, 12]

# Define the function to calculate the product of all entries in the list
def mult(numbers):
    product = 1
    for number in numbers:
        product *= number
    return product

# Calculating the product of the even_num list
product_of_evens = mult(even_num)

# Printing the result
print(f"The product of all even numbers up till 12 is {product_of_evens}.")


# ### Task 4: Pandas Create a Pandas dataframe named ‘pd’ that contains 5 rows and 4 columns, similar to the one given below:

# In[ ]:


# This is the table of question
import pandas as pd

data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}


df = pd.DataFrame(data)


print(df)


# ### Solution:

# In[59]:


# Defining data for the DataFrame
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

# Creating the DataFrame
df = pd.DataFrame(data)

# Printing the DataFrame
print(df)


# In[116]:


# Define the data for the DataFrame
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)


# ### 4.1  Print only the first two rows of the dataframe. 

# In[61]:


# Printing the first two rows of the DataFrame
print(df.head(2))


# ### 4.2 Print the second column

# In[62]:


# Printing the second column of the DataFrame
print(df['C2'])


# ### 4.3  Change the name of the third column from ‘C3’ to ‘B3’.

# In[63]:


# Changing the third column name from 'C3' to 'B3'
df = df.rename(columns={'C3': 'B3'})

# Printing the updated DataFrame to confirm the change
print(df)


# ### 4.4  Add a new column to the dataframe and name it ‘Sum’. 
# #### 4.5  Sum the entries of each row and add the result in the column ‘Sum’.

# In[120]:


# Adding a new column 'Sum' that contains the sum of values from existing columns
df['Sum'] = df.sum(axis=1)

# Printing the updated DataFrame to confirm the new column
print(df)


# ### 4.6 Read CSV file named ‘hello_sample.csv’ (the file is available in the class Google Drive shared folder) into a Pandas dataframe.
# #### 4.7  Print complete dataframe. 

# In[73]:


# Reading the CSV file into a DataFrame
df2 = pd.read_csv('hello_sample.csv')

# Printing complete DataFrame
print(df2)


# ### 4.8  Print only bottom 2 records of the dataframe. 

# In[74]:


# Printing the bottom 2 records of the DataFrame
print(df2.tail(2))


# ### 4.9  Print information about the dataframe. 

# In[75]:


# Printing information about the DataFrame
print(df2.info())


# ### 4.10  Print shape (rows x columns) of the dataframe. 

# In[76]:


# Printing the shape of the DataFrame
print(df2.shape)


# In[77]:


print(df2.columns)


# ### 4.11  Sort the data of the dataFrame using column ‘Weight’. 

# In[78]:


# Sort the DataFrame by the 'Weight' column
df2_sorted = df2.sort_values(by='Weight')

# Print the sorted DataFrame
print(df2_sorted)


# ### 4.12  Use isnull() and dropna() methods of the Pandas dataframe and see if they produce any changes. 

# In[80]:


# Checking for missing values
missing_values = df2.isnull()

# Printing the DataFrame2 showing missing values
print("Missing values in the DataFrame2:")
print(missing_values)

# Dropping rows with any missing values
df2_dropped = df2.dropna()

# Printing the DataFrame after dropping missing values
print("\nDataFrame after dropping missing values:")
print(df2_dropped)

# Printing if any changes were made
if df2.shape != df2_dropped.shape:
    print("\nChanges were made: Rows with missing values were removed.")
else:
    print("\nNo changes were made: No missing values were found.")


# In[ ]:




