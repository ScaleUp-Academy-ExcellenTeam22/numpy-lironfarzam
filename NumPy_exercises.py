import numpy as np
import matplotlib.pyplot as plt
import calendar


def change_sign_of_vector_in_range(vector: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Change the sign of a vector in a given range.

    Args:
        vector (np.ndarray): Vector to change the sign of.
        start (int): Start index of the range.
        end (int): End index of the range.

    Returns:
        np.ndarray: Vector with changed sign in the given range.
    """
    vector[start:end] = -vector[start:end]
    return vector


def numPy_matrix_of_zeros_and_frame_of_given_num(rows: int = 10, columns: int = 10, frame_num: int = 1) -> np.ndarray:
    """
    Create a matrix of zeros and a frame of given number.
    Args:
        rows (int, optional): Size of rows. Defaults to 10.
        columns (int, optional): Size of cols. Defaults to 10.
        frame_num (int, optional): The number with which to line the frame. Defaults to 1.

    Returns:
        np.ndarray: Matrix of zeros and a frame of given number.
    """
    matrix = np.zeros((rows, columns))
    matrix[0, :] = frame_num
    matrix[-1, :] = frame_num
    matrix[:, 0] = frame_num
    matrix[:, -1] = frame_num
    return matrix


def numPy_add_given_vector_to_all_rows_of_matrix(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a given vector to all rows of a matrix.
    Args:
        matrix (np.ndarray): Matrix to add the vector to.
        vector (np.ndarray): Vector to add to the matrix.

    Returns:
        np.ndarray: Matrix with the vector added to all rows.
    """
    # create a matrix all row vectors of the given vector
    row_vectors = np.array([vector for i in range(matrix.shape[0])])
    # add the row vectors to the matrix to the right of the matrix
    return np.concatenate((matrix, row_vectors), axis=1)


def show_the_sine_wave_of_a_given_frequency(frequency: int) -> None:
    """
    Show the sine wave of a given frequency between -pi and pi.
    Args:
        frequency (int): Frequency of the sine wave.
    """
    # create a vector of the given frequency
    x = np.linspace(np.pi * -1, np.pi, frequency)
    y = np.sin(x)
    # plot the sine wave
    plt.plot(x, y)
    plt.show()


def swap_rows_of_a_matrix(matrix: np.ndarray, row_1: int, row_2: int) -> np.ndarray:
    """
    Swap two rows of a matrix.
    Args:
        matrix (np.ndarray): Matrix to swap rows.
        row_1 (int): First row to swap.
        row_2 (int): Second row to swap.

    Returns:
        np.ndarray: Matrix with swapped rows.
    """
    # create a copy of the matrix
    matrix_copy = matrix.copy()
    # swap the rows
    matrix[row_1, :], matrix[row_2, :] = matrix_copy[row_2, :], matrix_copy[row_1, :]
    return matrix


def mapping_array_by_given_number(array: np.ndarray, number: int = 5,
                                  switch_smaller_to: int = -1,
                                  switch_larger_to: int = 1,
                                  switch_equal_to: int = 0) -> np.ndarray:
    """
    Map an array by a given number.
    Args:
        array (np.ndarray): Array to map.
        number (int, optional): Number to map the array by. Defaults to 5.
        switch_smaller_to (int, optional): Value to change the smaller numbers to. Defaults to 0.
        switch_larger_to (int, optional): Value to change the bigger numbers to. Defaults to 1.
        switch_equal_to (int, optional): Value to change the equal numbers to. Defaults to 0.

    Returns:
        np.ndarray: Mapped array.
    """
    # create a copy of the array
    array_copy = array.copy()
    # map the array
    array_copy[array_copy < number] = switch_smaller_to
    array_copy[array_copy > number] = switch_larger_to
    array_copy[array_copy == number] = switch_equal_to
    return array_copy


def combine_one_and_two_dimensional_array_together(array_1: np.ndarray, array_2: np.ndarray) -> None:
    """
        Combine one and two dimensional array together.
    Args:
        array_1 (np.ndarray): First array to combine.
        array_2 (np.ndarray): Second array to combine.
    """
    print("array_1:\n", array_1)
    print("array_2:\n", array_2)
    print("combined array:")
    # combine the arrays
    for first, second in np.nditer([array_1, array_2]):
        print(f" {first} : {second}")


def get_days_in_month(date: str) -> int:
    """
    Get the number of days in a given month.
    Args:
        date (str): Date to get the number of days in format: mm-yyyy.
        
    Returns:
        int: Number of days in the month.
    """
    # get the month and year from the date
    month, year = date.split("-")
    # get the number of days in the month
    return calendar.monthrange(int(year), int(month))[1]


if __name__ == "__main__":
    print("NumPy Start:")
    print("-" * 20)

    # 1.
    print("Description:\n   NumPy program to create a vector with values from 0 to 20 and change the sign of the "
          "numbers in the range from 9 to 15:")
    print("output:", change_sign_of_vector_in_range(np.arange(21), 9, 15))
    print("-" * 20)

    # 2.
    print("Description:\n   NumPy program to create a vector of length 10 with values evenly distributed between 5 "
          "and 50:")
    print("output:", np.linspace(5, 50, 10))
    print("-" * 20)

    # 3.
    print("Description:\n   NumPy program to find the number of rows and columns of a given matrix:")
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    print("matrix:\n", matrix)
    print("(row, col):", np.shape(matrix))
    print("-" * 20)

    # 4.
    print("Description:\n   NumPy program to create a 10x10 matrix, in which the elements on the borders will be "
          "equal to 1, and frame is equal 0:")
    print("matrix:\n", numPy_matrix_of_zeros_and_frame_of_given_num(10, 10, 1))
    
    # 5.
    print("Description:\n   NumPy program to add a vector to  end of each row of a given matrix:")
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    vector = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print("matrix:\n", matrix)
    print("vector:\n", vector)
    print("output:\n", numPy_add_given_vector_to_all_rows_of_matrix(matrix, vector))
    print("-" * 20)

    # 6.
    print("Description:\n   NumPy program to compute the x and y coordinates for points on a sine curve and plot the "
          "points using matplotlib:")
    show_the_sine_wave_of_a_given_frequency(100)
    print("-" * 20)

    # 7.
    print("Description:\n  NumPy program to create a 4x4 array with random values, now create a new array from the "
          "said array swapping first and last rows:")
    matrix = np.random.randint(0, 10, (4, 4))
    print("matrix:\n", matrix)
    print("output:\n", swap_rows_of_a_matrix(matrix, 0, matrix.shape[0] - 1))
    print("-" * 20)

    # 8.
    print("Description:\n  NumPy program to replace all numbers in a given array which is equal, less and greater to "
          "a given number:")
    array = np.array([1, 2, 3, 4, 8, 8, 9, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print("array:\n", array)
    print("output:\n", mapping_array_by_given_number(array, number=5,
                                                     switch_smaller_to=-1,
                                                     switch_larger_to=1,
                                                     switch_equal_to=0))

    # 9.
    print("Description:\n  NumPy program to multiply two given arrays of same size element-by-element:")
    array_1 = np.array([1, 2, 3, 4, 5])
    array_2 = np.array([1, 2, 3, 4, 5])
    print("array_1:\n", array_1)
    print("array_2:\n", array_2)
    print("output:\n", np.multiply(array_1, array_2))
    print("-" * 20)

    # 10.
    print("Description:\n   NumPy program to sort an along the first, last axis of an array:")
    array = np.array([[4, 6], [2, 1]])
    print("array:\n", array)
    print("sort by first axis output:\n", np.sort(array, axis=0))
    print("sort by last axis output:\n", np.sort(array, axis=1))
    print("-" * 20)

    11.
    print("Description:\n   NumPy program to create a 3-D array with ones on a diagonal and zeros elsewhere:")
    print("output:\n", np.eye(3))
    print("-" * 20)

    # 12.
    print("Description:\n   NumPy program to remove single-dimensional entries from a specified shape:")
    matrix = np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
    print("matrix:\n", matrix)
    print("output:\n", np.squeeze(matrix))
    print("-" * 20)

    # 13.
    print("Description:\n   NumPy program to convert two 1-D arrays into a 2-D array:")
    array_1 = np.array([1, 2, 3, 4, 5])
    array_2 = np.array([6, 7, 8, 9, 10])
    print("array_1:\n", array_1)
    print("array_2:\n", array_2)
    print("output:\n", np.stack((array_1, array_2), axis=1))
    print("-" * 20)

    # 14.
    print("Description:\n   NumPy program to combine a one and a two dimensional array together and display their "
          "elements:")
    array_1 = np.array([1, 2, 3, 4, 5])
    array_2 = np.array([[6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    combine_one_and_two_dimensional_array_together(array_1, array_2)
    print("-" * 20)

    # 15.
    print("Description:\n   NumPy program to create a three-dimension array with shape (300,400,5) and set to a "
          "variable. Fill the array elements with values using unsigned integer (0 to 255):")
    array = np.random.randint(0, 255, (300, 400, 5))
    print("array:\n", array)
    print("-" * 20)

    # 16.
    print("Description:\n   NumPy program to sort the student id with increasing height of the students from given "
          "students id and height. Print the integer indices that describes the sort order by multiple columns and "
          "the sorted data:")
    students_id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    students_height = np.array([1.6, 1.5, 1.3, 1.9, 1.7, 1.6, 1.9, 1.8, 1.9, 2.0])
    print("students_id:\n", students_id)
    print("students_height:\n", students_height)
    students_sort_by_height = np.lexsort((students_id,students_height))
    print("students_sort_by_height:")
    for index in students_sort_by_height:
        print(students_id[index], students_height[index])
    print("-" * 20)

    # 17.
    print("Description:\n   NumPy program to compute the median of flattened given array:")
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print("array:\n", array)
    print("median:\n", np.median(array))
    print("-" * 20)

    # 18.
    print("Description:\n   program to count the number of days of specific month:")
    date = input("Enter a date in the format (MM-YYYY): ")
    print(f"The number of days in {date} is: { get_days_in_month(date) }")
    print("-" * 20)

    print("NumPy End.")

