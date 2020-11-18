import numpy as np



torch = np.load(file="/home/ghpark/BASNet/inconv_torch.npy")
tf = np.load(file="inconv_tf.npy")


c = 64

temp = np.zeros((1, 224, 224, c))
for i in range(c):
  temp[:,:,:,i] = torch[:,i,:,:]

print("torch")
print(torch.shape)
print("tf")
print(tf.shape)
print("temp")
print(temp.shape)

equal = np.array_equal(temp, tf)
print("Equal?")
print(equal)


