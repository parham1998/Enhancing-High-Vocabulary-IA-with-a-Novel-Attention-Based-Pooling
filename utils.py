# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import skimage.transform

from datasets import get_mean_std

# checking the availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow(args, tensor):
    mean, std = get_mean_std(args)
    #
    tensor = tensor.numpy()
    # img shape => (3, h, w), img shape after transpose => (h, w, 3)
    tensor = tensor.transpose(1, 2, 0)
    image = tensor * np.array(std) + np.array(mean)
    image = image.clip(0, 1)
    plt.imshow(image)


def convertBinaryAnnotationsToClasses(annotations, classes):
    labels = []
    annotations = np.array(annotations, dtype='int').tolist()
    for i in range(len(classes)):
        if annotations[i] == 1:
            labels.append(classes[i])
    return labels


def predicted_batch_plot(args,
                         classes,
                         model,
                         images,
                         annotations,
                         threshold=0.5):
    model.to(device)
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        outputs, _ = model(images)
        outputs = torch.sigmoid(outputs)

    fig = plt.figure(figsize=(80, 30))
    for i in np.arange(args.batch_size):
        ax = fig.add_subplot(4, 8, i+1)
        imshow(args, images[i].cpu())
        #
        gt_anno = convertBinaryAnnotationsToClasses(annotations[i], classes)
        #
        o = np.array(outputs.cpu() > threshold, dtype='int')
        pre_anno = convertBinaryAnnotationsToClasses(o[i], classes)
        #
        string_gt = 'GT: '
        string_pre = 'Pre: '
        if len(gt_anno) != 0:
            for ele in gt_anno:
                string_gt += ele if ele == gt_anno[-1] else ele + ' - '
        #
        if len(pre_anno) != 0:
            for ele in pre_anno:
                string_pre += ele if ele == pre_anno[-1] else ele + ' - '

        ax.set_title(string_gt + '\n' + string_pre)
        plt.savefig(args.data_root_dir + 'batch_plot.jpg')


def visualize_att(args,
                  classes,
                  model,
                  images,
                  annotations,
                  threshold=0.5,
                  smooth=True):
    model.to(device)
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        outputs, attn_weights = model(images)
        outputs = torch.sigmoid(outputs)

    for i in np.arange(args.batch_size):
        o = np.array(outputs[i].cpu() > threshold, dtype='int')
        idxs = [idx for idx, val in enumerate(o) if val == 1]
        #
        pre_anno = convertBinaryAnnotationsToClasses(o, classes)
        pre_anno = ['raw_image'] + pre_anno
        #
        alphas = attn_weights[i][idxs, :].cpu()
        dim = int(np.sqrt(alphas.shape[-1]))
        alphas = alphas.view(-1, dim, dim)
        #
        plt.figure(figsize=(40, 20))
        for t in range(len(pre_anno)):
            if t > 50:
                break

            plt.subplot(np.int64(np.ceil(len(pre_anno) / 5.)), 5, t + 1)
            plt.text(0, 1, '%s' % (pre_anno[t]), color='black',
                     backgroundcolor='white', fontsize=20)

            if t == 0:
                imshow(args, images[i].cpu())
            else:
                imshow(args, images[i].cpu())
                current_alpha = alphas[t-1, :]
                if smooth:
                    alpha = skimage.transform.pyramid_expand(
                        current_alpha.numpy(), upscale=32, sigma=8)
                else:
                    alpha = skimage.transform.resize(
                        current_alpha.numpy(), [dim * 32, dim * 32])
                plt.imshow(alpha, alpha=0.8)

            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
        plt.show()
