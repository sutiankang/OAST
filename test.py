import os
import os.path as osp
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.dataset import UVOSDataset
from models.mobilevit3d import MobileViTVOS
from utils.utils import get_list, Ensemble, get_size


def get_argparse():

    parser = argparse.ArgumentParser("Test Data.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="choose using device")
    parser.add_argument("--img_size", type=int, nargs="+", default=[384, 640], help="training image size")
    parser.add_argument("--mean", type=list, default=[0.485, 0.456, 0.406], help="imagenet mean")
    parser.add_argument("--std", type=list, default=[0.229, 0.224, 0.225], help="imagenet std")

    parser.add_argument("--data-dir", type=str, default="your/data/path", help="dataset path")
    parser.add_argument("--save_dir", type=str, default="runs", help="save test data dir")
    parser.add_argument("--pretrained", type=str, default=None, help="use pretrained weights")

    parser.add_argument("--test_datasets", type=str, nargs="+", default=["DAVIS-2016", "FBMS"])
    parser.add_argument("--weights", type=str, nargs="+", default=None)
    parser.add_argument("--model_scale", type=str, default=None, choices=["xxs", "xs", "s"], help="model size")
    parser.add_argument("--dropout", default=None, type=float, help="before segmentation head add dropout")

    return parser.parse_args()


def flip(x, dim):
    if x.is_cuda:
        # dim -> w dimension
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda())
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())


def test():
    args = get_argparse()
    device = torch.device(args.device)
    model = MobileViTVOS(args)

    model.to(device)
    model = Ensemble(model, args.weights)()

    test_datasets = get_list(args.test_datasets)

    with torch.no_grad():
        model.eval()

        for test_dataset in test_datasets:
            dataset = UVOSDataset(data_dir=args.data_dir, size=get_size(args.img_size), mean=args.mean, std=args.std,
                                  mode='test', datasets=[test_dataset])
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                                     pin_memory=True if args.device == "cuda" else False)
            with tqdm(total=len(data_loader)) as tbar:
                for batch_test in data_loader:

                    tbar.set_description(f"Dataset: {test_dataset}")

                    image, flow = batch_test['image'], batch_test['flow']
                    image, flow = image.to(device), flow.to(device)
                    orin_w, orin_h = batch_test["size"]
                    image_path = batch_test["path"]

                    predict, _ = model(image, flow).sigmoid()
                    predict = F.interpolate(predict, size=(orin_h, orin_w), mode="bilinear", align_corners=True)
                    predict = predict.cpu().detach().numpy()
                    predict[predict >= 0.5] = 1
                    predict[predict < 0.5] = 0
                    predict = predict[0, 0, :, :] * 255
                    predict = Image.fromarray(predict).convert("L")
                    save_path = osp.join(args.save_dir, "test", test_dataset, image_path[0].split("/")[-2])
                    os.makedirs(save_path, exist_ok=True)
                    save_file = osp.join(save_path, image_path[0].split("/")[-1][:-4] + ".png")
                    predict.save(save_file)
                    tbar.update(1)


if __name__ == '__main__':
    test()
