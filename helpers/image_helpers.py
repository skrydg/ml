import matplotlib.pyplot as plt

def display_images(images, width):
    if (images.ndim == 4 and images.shape[-1] == 1):
        # Если пришла черно белая картинка с одним каналом, то мы просто удаляем лишнюю размерность
        images = images[:, :, :, 0]

    image_len = len(images)
    height = (image_len + width - 1) / width  # floor(image_len / width)

    fig = plt.figure(figsize=(20, 20))
    #fig.tight_layout(pad=1.0)
    for i in range(image_len):
        fig.add_subplot(height, width, i + 1)
        plt.imshow(images[i])
    
    plt.show()