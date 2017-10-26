from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
image = np.array(Image.open('data/tiger.jpg'))

image = image / 255
row, col, _ = image.shape
print("pixels: ", row, "*", col)

fig = plt.figure(figsize=(15, 10))
a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(image)
a.set_title('Royal Bengal Tiger, Sundarban, Bangladesh')
plt.show()

image_red = image[:, :, 0]
image_green = image[:, :, 1]
image_blue = image[:, :, 2]

original_bytes = image.nbytes
print("The space needed to store this image is: ", original_bytes/1024, " KB")

U_r, d_r, V_r = np.linalg.svd(image_red, full_matrices=True)
U_g, d_g, V_g = np.linalg.svd(image_green, full_matrices=True)
U_b, d_b, V_b = np.linalg.svd(image_blue, full_matrices=True)

bytes_to_be_stored = sum([matrix.nbytes for matrix in [U_r, d_r, V_r, U_g, d_g, V_g, U_b, d_b, V_b]])
print("The matrices that we store have total size: ", bytes_to_be_stored/1024, " KB")

k = 50

U_r_k = U_r[:, 0:k]
V_r_k = V_r[0:k, :]
U_g_k = U_g[:, 0:k]
V_g_k = V_g[0:k, :]
U_b_k = U_b[:, 0:k]
V_b_k = V_b[0:k, :]

d_r_k = d_r[0:k]
d_g_k = d_g[0:k]
d_b_k = d_b[0:k]

compressed_bytes = sum([matrix.nbytes for matrix in 
                        [U_r_k, d_r_k, V_r_k, U_g_k, d_g_k, V_g_k, U_b_k, d_b_k, V_b_k]])
print("Compressed matrices have total size: ", compressed_bytes/1024, "KB")

ratio = compressed_bytes / original_bytes
print("Compression ratio between the original image size and the total size of the compressed factors is: ", ratio)

image_red_approx = np.dot(U_r_k, np.dot(np.diag(d_r_k), V_r_k))
image_green_approx = np.dot(U_g_k, np.dot(np.diag(d_g_k), V_g_k))
image_blue_approx = np.dot(U_b_k, np.dot(np.diag(d_b_k), V_b_k))

image_reconstructed = np.zeros((row, col, 3))

image_reconstructed[:, :, 0] = image_red_approx
image_reconstructed[:, :, 1] = image_green_approx
image_reconstructed[:, :, 2] = image_blue_approx

image_reconstructed[image_reconstructed < 0] = 0
image_reconstructed[image_reconstructed > 1] = 1

fig = plt.figure(figsize=(15, 10))
a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(image_reconstructed)
a.set_title('Compressed image of the Royal Bengal Tiger, using best rank-{} approximation'.format(k))
plt.show()

k = 10

U_r_k = U_r[:, 0:k]
V_r_k = V_r[0:k, :]
U_g_k = U_g[:, 0:k]
V_g_k = V_g[0:k, :]
U_b_k = U_b[:, 0:k]
V_b_k = V_b[0:k, :]

d_r_k = d_r[0:k]
d_g_k = d_g[0:k]
d_b_k = d_b[0:k]

compressed_bytes = sum([matrix.nbytes for matrix in 
                        [U_r_k, d_r_k, V_r_k, U_g_k, d_g_k, V_g_k, U_b_k, d_b_k, V_b_k]])
print("Compressed matrices have total size: ", compressed_bytes/1024, "KB")

image_red_approx = np.dot(U_r_k, np.dot(np.diag(d_r_k), V_r_k))
image_green_approx = np.dot(U_g_k, np.dot(np.diag(d_g_k), V_g_k))
image_blue_approx = np.dot(U_b_k, np.dot(np.diag(d_b_k), V_b_k))

image_reconstructed = np.zeros((row, col, 3))
image_reconstructed[:, :, 0] = image_red_approx
image_reconstructed[:, :, 1] = image_green_approx
image_reconstructed[:, :, 2] = image_blue_approx
image_reconstructed[image_reconstructed < 0] = 0
image_reconstructed[image_reconstructed > 1] = 1

fig = plt.figure(figsize=(15, 10))
a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(image_reconstructed)
a.set_title('Compressed image of the Royal Bengal Tiger, using best rank-{} approximation'.format(k))
plt.show()

k = 200

U_r_k = U_r[:, 0:k]
V_r_k = V_r[0:k, :]
U_g_k = U_g[:, 0:k]
V_g_k = V_g[0:k, :]
U_b_k = U_b[:, 0:k]
V_b_k = V_b[0:k, :]

d_r_k = d_r[0:k]
d_g_k = d_g[0:k]
d_b_k = d_b[0:k]

image_red_approx = np.dot(U_r_k, np.dot(np.diag(d_r_k), V_r_k))
image_green_approx = np.dot(U_g_k, np.dot(np.diag(d_g_k), V_g_k))
image_blue_approx = np.dot(U_b_k, np.dot(np.diag(d_b_k), V_b_k))

image_reconstructed = np.zeros((row, col, 3))
image_reconstructed[:, :, 0] = image_red_approx
image_reconstructed[:, :, 1] = image_green_approx
image_reconstructed[:, :, 2] = image_blue_approx
image_reconstructed[image_reconstructed < 0] = 0
image_reconstructed[image_reconstructed > 1] = 1

fig = plt.figure(figsize=(15, 10))
a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(image_reconstructed)
a.set_title('Compressed image of the Royal Bengal Tiger, using best rank-{} approximation'.format(k))
plt.show()
