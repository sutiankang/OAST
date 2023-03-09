import torch


def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)


def create_iou_bce_loss(predict, target):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    iou_loss = IOU(size_average=True)
    criterion = bce_loss(predict, target) + iou_loss(torch.sigmoid(predict), target)
    return criterion


def create_online_iou_bce_loss(predict, target):
    # B C H W
    target = torch.sigmoid(target)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    iou_loss = IOU(size_average=True)
    criterion = bce_loss(predict, target) + iou_loss(torch.sigmoid(predict), target)
    return criterion