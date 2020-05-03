import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import torchvision
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def preprocess(x, k = 5):
    log = torch.log(torch.abs(x) + 1e-7)
    clamped_log = (log / k).clamp(min = -1)
    sign = (x * np.exp(k)).clamp(min = -1, max = 1)
    return torch.cat([clamped_log.unsqueeze(-1), sign.unsqueeze(-1)], dim = -1)

def get_parameters_and_gradients(model)->list:
    parameters = list(map(lambda x: x.data.view(1, -1), model.parameters()))
    gradients = list(map(lambda x: x.data.view(1, -1), map(lambda x: x.grad, model.parameters())))
    parameters_name = list(map(lambda x: x[0], model.named_parameters()))
    return parameters, gradients, parameters_name

def copy_and_pad(tensors: list):
    real_lengths = list(map(lambda x: x.shape[1], tensors))
    max_length = max(real_lengths)
    def pad(tensor, l):
        new_tensor = torch.zeros((1, l), device = tensor.device)
        new_tensor[0, :tensor.shape[1]] = tensor
        return new_tensor
    new_tensors = map(lambda x: pad(x.clone(), max_length), tensors)

    return torch.cat(list(new_tensors), 0), real_lengths

def update_parameter(parameter, delta):
    parameter.add_(delta)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(target)
        if (target.dim() > 1):
            target = torch.argmax(target, 1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0].item()

def center_crop_with_flip(img, size, vertical_flip=False):
    crop_h, crop_w = size
    first_crop = F.center_crop(img, (crop_h, crop_w))
    if vertical_flip:
        img = F.vflip(img)
    else:
         img = F.hflip(img)
    second_crop = F.center_crop(img, (crop_h, crop_w))
    return (first_crop, second_crop)

class CenterCropWithFlip(object):
    """Center crops with its mirror version.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return center_crop_with_flip(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def preprocess_strategy(dataset):
    evaluate_transforms = None
    if dataset.startswith('CUB224'):
        train_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(0.4, 0, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize(224),
            CenterCropWithFlip(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    elif dataset.startswith('CUB'):
        train_transforms = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(0.4, 0, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize(448),
            CenterCropWithFlip(448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])

    elif dataset.startswith('Aircraft'):
        train_transforms = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.CenterCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize((512,512)),
            CenterCropWithFlip(448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    elif dataset.startswith('Cars'):
        train_transforms = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize((448,448)),
            CenterCropWithFlip(448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    elif dataset.startswith('Dogs'):
        train_transforms = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize((448, 448)),
            CenterCropWithFlip(448, 448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    elif dataset.startswith('ImageNet'):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    else:
        raise KeyError("=> transform method of '{}' does not exist!".format(dataset))
    return train_transforms, val_transforms, evaluate_transforms


if __name__ == '__main__':
    l = [torch.randn((1, 2)), torch.zeros((1,3))]
    print(copy_and_pad(l))