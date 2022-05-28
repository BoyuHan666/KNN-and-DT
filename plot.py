import matplotlib.pyplot as plt
x = [1,2,3,4,5,6,7,8,9,10]
y = [79.1,74.0,78.7,75.0,80.2,80.8,83.1,81.8,82.8,82.9]
plt.title("different distance for data1") # Manhattan Euclidean
plt.xlabel("k value")
plt.ylabel("avg accuracy")
plt.plot(x,y,label='Euclidean')

x = [1,2,3,4,5,6,7,8,9,10]
y = [82.3,78.6,82.2,79.4,80.5,79.9,83.2,80.8,83.5,82.8]
plt.plot(x,y,label='Manhattan')
plt.legend()
plt.show()

x = [1,2,3,4,5,6,7,8,9,10]
y = [72.3,73.3,75.6,76.9,77.5,78.2,78.9,78.3,78.4,78.4]
plt.title("different distance for data2") # Manhattan Euclidean
plt.xlabel("k value")
plt.ylabel("avg accuracy")
plt.plot(x,y,label='Euclidean')

x = [1,2,3,4,5,6,7,8,9,10]
y = [76.4,76.4,76.8,77.5,77.8,78.3,78.7,77.3,77.9,77.2]
plt.plot(x,y,label='Manhattan')
plt.legend()
plt.show()