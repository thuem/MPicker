import torch

class AbstractDataset(torch.utils.data.Dataset):
    """
    Serves as a common interface to reduce boilerplate and help dataset
    customization

    A generic Dataset for the maskrcnn_benchmark must have the following
    non-trivial fields / methods implemented:
        CLASSES - list/tuple:
            A list of strings representing the classes. It must have
            "__background__" as its 0th element for correct id mapping.

        __getitem__ - function(idx):
            This has to return three things: img, target, idx.
            img is the input image, which has to be load as a PIL Image object
            implementing the target requires the most effort, since it must have
            multiple fields: the size, bounding boxes, labels (contiguous), and
            masks (either COCO-style Polygons, RLE or torch BinaryMask).
            Usually the target is a BoxList instance with extra fields.
            Lastly, idx is simply the input argument of the function.

    also the following is required:
        __len__ - function():
            return the size of the dataset
        get_img_info - function(idx):
            return metadata, at least width and height of the input image
    """

    def __init__(self, *args, **kwargs):
        return


    def __getitem__(self, idx):
        raise NotImplementedError


    def __len__(self):
        raise NotImplementedError
