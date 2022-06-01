from src.Data.DataGenerator import ResizedDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio

test = ResizedDataGenerator()
im, label = test.load_sample(1)

print("Test getitem")
test[12]
print(len(test))

plt.subplot(1, 2, 1)
plt.imshow(im), plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(label), plt.axis("off")

plt.show()

for idx, (im, label) in enumerate(test(10)):
    plt.subplot(2, 5, idx + 1)
    plt.imshow(label), plt.axis("off")

plt.show()

# test_im_path = '/home/melih/Code/uni/sem6/datasets/train/AOI_11_Rotterdam/SAR-Intensity_128/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804113009_20190804113242_tile_4800.tif'
# test_label_path = '/home/melih/Code/uni/sem6/datasets/train/AOI_11_Rotterdam/Labels_128/20190804111851_20190804112030_tile_5176.tif'

# test_label = imageio.imread(test_label_path)
# test_im = imageio.imread(test_im_path)

# plt.subplot(1, 2, 1)
# plt.title(
#     f'Shape: {test_im.shape} \nMax: {test_im.max()}\n Min: {test_im.min()}')
# plt.imshow(test_im)

# plt.subplot(1, 2, 2)
# plt.imshow(test_label)
# plt.show()
