import torchvision
import torch
import torchsummary
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

def sample_mask(img_w, img_h, mask_w, mask_h, static_center=False):
    #return a sampled mask, a tensor of size ((mask_w*mask_h)+1, img_w, img_h)
    length = mask_w * mask_h + 1
    order = np.random.permutation(np.arange(0, mask_w * mask_h, 1))  # Sample an order
    mask = torch.ones(length, 3, mask_w, mask_h)
    mask = mask.view(length, 3, -1)
    for j in range(1, length):
        mask[j:, :, order[j - 1]] = 0
    mask = mask.view(length, 3, mask_w, mask_h)
    if static_center:
        mask[:, :, mask_w//2, mask_h//2] = 1
    mask = F.interpolate(mask.clone(), size=[img_w, img_h],mode="nearest").float()
    return mask, order


def getShapley_pixel(img, label, model, sample_times, mask_size, k=0):
    b, c, w, h = img.size()
    #assert b == 1 and label.size(0) == 1
    shap_value = torch.zeros((mask_size ** 2))
    with torch.no_grad():
        for i in range(sample_times):
            mask, order = sample_mask(w, h, mask_size, mask_size)
            base = img[k].expand(mask.size(0), 3, w, h).clone()
            masked_img = base * mask.cuda()
            output = model(masked_img)
            if torch.any(torch.isnan(output)):
                raise ValueError("NAN in output")
            y = output[:, label[k]]
            yy = y[:-1]
            dy = yy - y[1:]
            shap_value[order] += (dy.cpu())
        shap_value /= sample_times
    return shap_value


def transform_fft(img):
    img = np.array(img.cpu())

    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)

    return torch.tensor(img)


def transform_ifft(img):
    img = np.array(img.cpu())
    img = np.fft.ifftshift(img)
    img = np.fft.ifft2(img)
    if np.abs(np.sum(np.imag(img))) > 1e-5:
        raise ValueError(f"imag of reconstructed image is too big:{np.abs(np.sum(np.imag(img)))}")

    return torch.tensor(np.real(img)).float()


def getShapley_freq(img, label, model, sample_times, mask_size, k=0, n_per_batch=1, split_n=1, static_center=False, fix_masks=False, mask_path=None):
    b, c, w, h = img.size()
    length = mask_size ** 2 + 1
    # assert b == 1 and label.size(0) == 1
    shap_value = torch.zeros((mask_size ** 2))
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in range(sample_times // n_per_batch):
                maskes = []
                orders = []
                if not fix_masks:
                    for j in range(n_per_batch):
                        mask, order = sample_mask(w, h, mask_size, mask_size, static_center=static_center)
                        maskes.append(mask)
                        orders.append(order)
                    maskes = torch.cat(maskes, 0)
                    assert maskes.size(0) == n_per_batch * length
                else:
                    maskes = torch.load(os.path.join(mask_path, f"mask_{i}.pth"))
                    orders = torch.load(os.path.join(mask_path, f"order_{i}.pth"))
                if split_n > 1:
                    base = transform_fft(img[k])
                    bs = maskes.size(0) // split_n
                    outputs = []
                    for j in range(maskes.size(0)//bs):
                        if j == maskes.size(0) // bs -1:
                            current_mask = maskes[j*bs:]
                        else:
                            current_mask = maskes[j*bs:(j+1) * bs]
                        masked_img = base.expand(current_mask.size(0), 3, w, h).clone() * current_mask
                        masked_img = transform_ifft(masked_img)
                        masked_img = masked_img.cuda()
                        masked_img = torch.clamp(masked_img, 0., 1.)
                        outputs.append(model(masked_img))
                    output = torch.cat(outputs, dim=0)
                else:
                    base = transform_fft(img[k]).expand(maskes.size(0), 3, w, h).clone()
                    masked_img = base * maskes
                    masked_img = transform_ifft(masked_img)
                    masked_img = masked_img.cuda()
                    masked_img = torch.clamp(masked_img, 0., 1.)
                    output = model(masked_img)
                for j in range(n_per_batch):
                    y = output[j * length:(j + 1) * length, label[k]]
                    yy = y[:-1]
                    dy = yy - y[1:]
                    if torch.any(torch.isnan(dy)):
                        raise ValueError("Nan in dy")
                    shap_value[orders[j]] += (dy.cpu())
                if i % 100 == 0:
                    print(f"{i}/{sample_times // n_per_batch}")
        shap_value /= sample_times//n_per_batch * n_per_batch
    return shap_value


def main():
    # Load the pre-trained model
    # model = torchvision.models.densenet121(pretrained=True).cuda()
    # torchsummary inception_v3

    # Print the model summary
    # torchsummary.summary(model, (3, 224, 224))
    train_dir = '/datasets/imagenet(30)/train'
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    train_set = torchvision.datasets.ImageFolder(train_dir, train_transform)
    train_loader_match = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True,
                                                     num_workers=4, pin_memory=True)
    model = torchvision.models.inception_v3(pretrained=True).cuda()
    model.eval()

    for i, (img, label) in enumerate(train_loader_match):
        img = img.cuda()
        label = label.cuda()
        getShapley_pixel(img, label, model, 10, 16, 0)




if __name__ == "__main__":
    main()