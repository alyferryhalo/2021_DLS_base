# Пусть у нас есть следующий список, в котором элементы -- tuple из строк.
# Мы хотим отсортировать этот список по последней букве второго элемента каждого tuple, т.е. получить такой список:
sorted_items = sorted(items, key=lambda x: x[1][-1])

# Дан list x:
# x = [1, 2, 3, 4, 5]
# x[<YOUR CODE>] = [-1, -3, -5]
# Заполните слайс вместо <YOUR CODE>, чтобы результатом стал следующий list: [-5, 2, -3, 4, -1]
x[::-2] = [-1, -3, -5]
