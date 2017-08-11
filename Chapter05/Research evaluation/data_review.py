import tensorflow as tf
import numpy as np
%autoindent
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

# Load data
data = np.load('data_with_labels.npz')
train = data['arr_0']/255.
labels = data['arr_1']

# Look at some data
print(train[0])
print(labels[0])

# If you have matplotlib installed
import matplotlib.pyplot as plt
plt.ion()

# One look at a letter/digit from each font
# Best to reshape as one large array, then plot
all_letters = np.zeros([5*36,62*36])
for font in range(5):
    for letter in range(62):
        all_letters[font*36:(font+1)*36,
                letter*36:(letter+1)*36] = \
                train[9*(font*62 + letter)]
plt.pcolormesh(all_letters,
        cmap=plt.cm.gray)

# Let's look at the jitters
f, plts = plt.subplots(3,3, sharex=True, sharey=True)
for i in range(3):
    for j in range(3):
        plts[i,j].pcolor(train[i + 3*j]) 

