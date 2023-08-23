class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        upper = (height - crop_size) // 2
        right = left + crop_size
        lower = upper + crop_size
        image = image.crop((left, upper, right, lower))

        resized_image = image.resize(self.size)
        return resized_image
    
class PadTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        upper = (height - crop_size) // 2
        right = left + crop_size
        lower = upper + crop_size
        image = image.crop((left, upper, right, lower))

        pad_width = max(self.size[0] - crop_size, 0)
        pad_height = max(self.size[1] - crop_size, 0)

        padded_image = Image.new('RGB', self.size, (0, 0, 0))
        padded_image.paste(image, (pad_width // 2, pad_height // 2))

        return padded_image
    
class MakeSameChannelAgain:
    def __init__(self):
        pass

    def __call__(self, image):
        if image.shape[0] == 1:
            image = torch.cat([image]*3, dim=0)
        elif image.shape[0] == 4:
            image = image[:3, :, :]
        elif image.shape[0] != 3:
            raise ValueError("Input image must have 1, 3, or 4 channels")
        return image