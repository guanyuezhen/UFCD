import torch
import numpy as np


def visulize_features(features):
    """
    可视化特征图，各维度make grid到一起
    """
    from torchvision.utils import make_grid
    assert features.ndim == 4
    b, c, h, w = features.shape
    features = features.view((b * c, 1, h, w))
    grid = make_grid(features)
    visualize_tensors(grid)


def visualize_tensors(*tensors):
    """
    可视化tensor，支持单通道特征或3通道图像
    :param tensors: tensor: C*H*W, C=1/3
    :return:
    """
    import matplotlib.pyplot as plt
    # from misc.torchutils import tensor2np
    images = []
    for tensor in tensors:
        assert tensor.ndim == 3 or tensor.ndim == 2
        if tensor.ndim == 3:
            assert tensor.shape[0] == 1 or tensor.shape[0] == 3
        images.append(tensor2np(tensor))
    nums = len(images)
    if nums > 1:
        fig, axs = plt.subplots(1, nums)
        for i, image in enumerate(images):
            axs[i].imshow(image, cmap='jet')
        plt.show()
    elif nums == 1:
        fig, ax = plt.subplots(1, nums)
        for i, image in enumerate(images):
            ax.imshow(image, cmap='jet')
        plt.show()


def tensor2np(input_image, if_normalize=True):
    """
    :param input_image: C*H*W / H*W
    :return: ndarray, H*W*C / H*W
    """
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array

    else:
        image_numpy = input_image
    if image_numpy.ndim == 2:
        return image_numpy
    elif image_numpy.ndim == 3:
        C, H, W = image_numpy.shape
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        #  如果输入为灰度图C==1，则输出array，ndim==2；
        if C == 1:
            image_numpy = image_numpy[:, :, 0]
        if if_normalize and C == 3:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            #  add to prevent extreme noises in visual images
            image_numpy[image_numpy < 0] = 0
            image_numpy[image_numpy > 255] = 255
            image_numpy = image_numpy.astype(np.uint8)
    return image_numpy
