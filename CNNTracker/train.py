"""
All functions in this code were written by Jelmer M. Wolterink
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import SimpleITK as sitk
import tqdm
import argparse
from glob import glob

BS = 64
PS = 19
VS = 0.5
NV = 500


def load_mhd_to_npy(filename):
    image = sitk.ReadImage(filename)
    spacing = image.GetSpacing()
    offset = image.GetOrigin()
    return np.swapaxes(sitk.GetArrayFromImage(image), 0, 2), spacing, offset


def getData(datadir):
    images = []
    vessels = []
    spacings = []
    offsets = []
    img_paths = sorted(glob(os.path.join(datadir, "images", "*")))

    for i, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path).split(".")[0]
        print("Loaded patient {}".format(img_name))
        image, spacing, offset = load_mhd_to_npy(img_path)
        image = image.astype("float32")
        if np.min(image) == 0:
            image = image - 1024.0
        image = np.clip(image, -1024.0, 3072)

        images.append(image)
        spacings.append(spacing)
        offsets.append(offset)
        ctls = sorted(glob(os.path.join(datadir, "centerlines", img_name, "*.txt")))

        pt_vessels = []
        for ctl in ctls:
            ctl_array = np.loadtxt(ctl)
            for j, arr in enumerate(ctl_array):
                ctl_array[j, :3] = ctl_array[j, :3] - np.asarray(offset)

            pt_vessels.append(ctl_array[:, :4])
            flipped_ctl = np.flipud(ctl_array[:, :4])
            pt_vessels.append(flipped_ctl)

        vessels.append(pt_vessels)

    return images, vessels, spacings


def directionToClass(vertices, target, rotMatrix=np.eye(4, dtype="float32")):
    vertexlength = np.linalg.norm(np.squeeze(vertices[0, :]))

    target = target.reshape((1, 3))
    target = np.dot(rotMatrix, np.array([target[0, 0], target[0, 1], target[0, 2], 0.0]))
    target = target[:3]

    target = target / (np.linalg.norm(target) / vertexlength)

    dist_to_vert = np.linalg.norm(vertices - target, axis=1)

    distro = np.zeros(dist_to_vert.shape, dtype="float32")

    # Set label at closest vertice
    distro[np.argmin(dist_to_vert)] = 1.0

    return distro


def getRotationMatrix(rotate=True):
    if not rotate:
        return np.eye(4, dtype="float32"), np.eye(4, dtype="float32")
    else:
        # Constrain angles between 0 and 90 degrees --> 31-07-17 No, to 360
        rotangle = np.random.randint(0, 3, 1)
        alpha = 0.0
        beta = 0.0
        gamma = 0.0
        if rotangle == 0:
            alpha = (np.squeeze(float(np.random.randint(0, 360, 1))) / 180.0) * np.pi
        if rotangle == 1:
            beta = (np.squeeze(float(np.random.randint(0, 360, 1))) / 180.0) * np.pi
        if rotangle == 2:
            gamma = (np.squeeze(float(np.random.randint(0, 360, 1))) / 180.0) * np.pi

        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)
        rotMatrix = np.array(
            [
                [cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg, 0.0],
                [cb * sg, ca * cg + sa * sb * sg, -1.0 * cg * sa + ca * sb * sg, 0.0],
                [-1 * sb, cb * sa, ca * cb, 0],
                [0, 0, 0, 1.0],
            ]
        )
        alpha = -1.0 * alpha
        beta = -1.0 * beta
        gamma = -1.0 * gamma
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)
        invMatrix = np.array(
            [
                [cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg, 0.0],
                [cb * sg, ca * cg + sa * sb * sg, -1.0 * cg * sa + ca * sb * sg, 0.0],
                [-1 * sb, cb * sa, ca * cb, 0],
                [0, 0, 0, 1.0],
            ]
        )
        return rotMatrix, invMatrix


def fast_nearest(input_array, x_indices, y_indices, z_indices):
    x_ind = (x_indices + 0.5).astype(np.int32)
    y_ind = (y_indices + 0.5).astype(np.int32)
    z_ind = (z_indices + 0.5).astype(np.int32)
    x_ind[np.where(x_ind >= input_array.shape[0])] = input_array.shape[0] - 1
    y_ind[np.where(y_ind >= input_array.shape[1])] = input_array.shape[1] - 1
    z_ind[np.where(z_ind >= input_array.shape[2])] = input_array.shape[2] - 1
    x_ind[np.where(x_ind < 0)] = 0
    y_ind[np.where(y_ind < 0)] = 0
    z_ind[np.where(z_ind < 0)] = 0
    return input_array[x_ind, y_ind, z_ind]


def fast_trilinear(input_array, x_indices, y_indices, z_indices):
    x0 = x_indices.astype(np.int32)
    y0 = y_indices.astype(np.int32)
    z0 = z_indices.astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # #Check if xyz1 is beyond array boundary:
    x0[np.where(x0 >= input_array.shape[0])] = input_array.shape[0] - 1
    y0[np.where(y0 >= input_array.shape[1])] = input_array.shape[1] - 1
    z0[np.where(z0 >= input_array.shape[2])] = input_array.shape[2] - 1
    x1[np.where(x1 >= input_array.shape[0])] = input_array.shape[0] - 1
    y1[np.where(y1 >= input_array.shape[1])] = input_array.shape[1] - 1
    z1[np.where(z1 >= input_array.shape[2])] = input_array.shape[2] - 1
    x0[np.where(x0 < 0)] = 0
    y0[np.where(y0 < 0)] = 0
    z0[np.where(z0 < 0)] = 0
    x1[np.where(x1 < 0)] = 0
    y1[np.where(y1 < 0)] = 0
    z1[np.where(z1 < 0)] = 0

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0
    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output


def draw_sample_3D_world_fast(
    image,
    x,
    y,
    z,
    imagespacing,
    patchsize,
    patchspacing,
    rotMatrix=np.eye(4, dtype="float32"),
    interpolation="nearest",
):
    patchmargin = (patchsize - 1) / 2
    unra = np.unravel_index(np.arange(np.prod(patchsize)), patchsize)
    xs = (x + (unra[0] - patchmargin[0]) * patchspacing[0]) / imagespacing[0]
    ys = (y + (unra[1] - patchmargin[1]) * patchspacing[1]) / imagespacing[1]
    zs = (z + (unra[2] - patchmargin[2]) * patchspacing[2]) / imagespacing[2]

    xs = xs - (x / imagespacing[0])
    ys = ys - (y / imagespacing[1])
    zs = zs - (z / imagespacing[2])

    coords = np.concatenate(
        (
            np.reshape(xs, (1, xs.shape[0])),
            np.reshape(ys, (1, ys.shape[0])),
            np.reshape(zs, (1, zs.shape[0])),
            np.zeros((1, xs.shape[0]), dtype="float32"),
        ),
        axis=0,
    )

    coords = np.dot(rotMatrix, coords)

    xs = np.squeeze(coords[0, :]) + (x / imagespacing[0])
    ys = np.squeeze(coords[1, :]) + (y / imagespacing[1])
    zs = np.squeeze(coords[2, :]) + (z / imagespacing[2])

    if interpolation == "linear":
        patch = fast_trilinear(image, xs, ys, zs)
    else:
        patch = fast_nearest(image, xs, ys, zs)

    return patch.reshape(patchsize)


# Get minibatch for training
def getMiniBatch(
    images,
    vessels,
    spacings,
    vertices,
    bs=64,
    nclass=50,
    rotate=False,
    train=False,
    pw=19,
    vw=0.5,
    step_size=2,
):
    batch = np.zeros((bs, 1, pw, pw, pw))

    # Target directions
    targets = np.zeros((bs, nclass))

    # Fill minibatch with samples
    for ba in range(bs):
        # Randomly determine training image
        im = np.random.randint(0, len(images), 1)[0]
        # Randomly determine training vessel
        vind = np.random.randint(0, len(vessels[im]), 1)[0]
        vessel = vessels[im][vind].copy()

        # Vessel radius at each location
        radii = vessel[:, 3]

        # Vessel coordinates
        vessel = vessel[:, :3]

        # Randomly select location
        start = np.random.randint(0, vessel.shape[0], 1)

        # Apply random displacement to location (only in training)
        locx = vessel[start, 0]
        locy = vessel[start, 1]
        locz = vessel[start, 2]

        if train:
            sigma = radii[start] * 0.25
            locx = locx + np.random.normal(0, sigma, 1)
            locy = locy + np.random.normal(0, sigma, 1)
            locz = locz + np.random.normal(0, sigma, 1)

        point = np.reshape(np.array([locx, locy, locz]), (1, 3))

        # Now from displaced point find closest point on vessel
        minpt = np.argmin(np.linalg.norm(vessel - point, axis=1))

        # What is the next point from that minpt?
        nextp = vessel[min(minpt + int((radii[minpt] / step_size)), vessel.shape[0] - 1), :]

        # And the other way around? What is the next point from that minpt?
        prevp = vessel[max(minpt - int((radii[minpt] / step_size)), 0), :]

        # Compute displacement between next point and current point
        displacement = nextp - point
        displacement = displacement / (np.linalg.norm(displacement) / radii[minpt])
        displacement_back = prevp - point
        displacement_back = displacement_back / (np.linalg.norm(displacement_back) / radii[minpt])

        # Put patches in sequence
        rotMatrix, invMatrix = getRotationMatrix(rotate)

        ## SUBTRACT VOXEL CENTER FROM POINT TO MATCH ROTTERDAM FRAMEWORK CONVENTION
        locx = locx - spacings[im][0] / 2.0
        locy = locy - spacings[im][1] / 2.0
        locz = locz - spacings[im][2] / 2.0

        batch[ba, 0, :, :, :] = draw_sample_3D_world_fast(
            images[im],
            locx,
            locy,
            locz,
            spacings[im],
            np.array([pw, pw, pw]),
            np.array([vw, vw, vw]),
            rotMatrix,
            interpolation="nearest",
        ).astype("float32")

        targets[ba, : (nclass - 1)] = directionToClass(vertices, displacement, rotMatrix=invMatrix)
        targets[ba, : (nclass - 1)] += directionToClass(vertices, displacement_back, rotMatrix=invMatrix)

        # NORMALIZE
        targets[ba, : (nclass - 1)] = targets[ba, : (nclass - 1)] / np.sum(targets[ba, : (nclass - 1)])
        targets[ba, nclass - 1] = radii[minpt]

    return batch, targets


class CNNTracking(nn.Module):
    def __init__(self):
        super(CNNTracking, self).__init__()
        C = 32
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=C, kernel_size=3, dilation=1)  # 17
        self.bn1 = nn.BatchNorm3d(num_features=C)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1)  # 15
        self.bn2 = nn.BatchNorm3d(num_features=C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=2)
        self.bn3 = nn.BatchNorm3d(num_features=C)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=4)
        self.bn4 = nn.BatchNorm3d(num_features=C)
        self.relu4 = nn.ReLU()

        self.conve = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=8)
        self.bne = nn.BatchNorm3d(num_features=C)
        self.relue = nn.ReLU()

        self.convf = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=16)
        self.bnf = nn.BatchNorm3d(num_features=C)
        self.reluf = nn.ReLU()

        self.conv5 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1)
        self.bn5 = nn.BatchNorm3d(num_features=C)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv3d(in_channels=C, out_channels=2 * C, kernel_size=1, dilation=1)
        self.bn6 = nn.BatchNorm3d(num_features=2 * C)
        self.relu6 = nn.ReLU()

        self.conv6_c = nn.Conv3d(in_channels=2 * C, out_channels=NV, kernel_size=1)

        self.conv6_r = nn.Conv3d(in_channels=2 * C, out_channels=1, kernel_size=1)

    def forward(self, input):
        h1 = self.relu1(self.bn1(self.conv1(input)))
        h2 = self.relu2(self.bn2(self.conv2(h1)))
        h3 = self.relu3(self.bn3(self.conv3(h2)))
        h4 = self.relu4(self.bn4(self.conv4(h3)))
        h5 = self.relu5(self.bn5(self.conv5(h4)))
        h6 = self.relu6(self.bn6(self.conv6(h5)))
        h6_c = self.conv6_c(h6)
        h6_r = self.conv6_r(h6)
        out = torch.cat((h6_c, h6_r), dim=1)
        return out


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


class CenterlineLoss(nn.Module):
    def __init__(self):
        super(CenterlineLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.squeeze(inputs)
        targets = torch.squeeze(targets)
        return cross_entropy(inputs[:, :-1], targets[:, :-1]) + 10.0 * F.mse_loss(
            inputs[:, -1], targets[:, -1]
        )  # contiguous?


def main(args):
    # Output classes: NV directions and 1 radius
    nclass = NV + 1
    network = CNNTracking()
    network.to(args.device)
    network.float()
    network.train()

    train_images, train_vessels, train_spacings = getData(args.datadir)
    vertices = np.loadtxt("vertices500.txt")

    criterion = CenterlineLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 20000, 30000, 40000], gamma=0.1)

    num_epochs = 50001

    for it in tqdm.tqdm(range(num_epochs)):
        batch, target = getMiniBatch(
            train_images,
            train_vessels,
            train_spacings,
            vertices,
            BS,
            nclass,
            rotate=True,
            train=True,
            pw=PS,
            vw=VS,
            step_size=args.step_size,
        )
        batch, target = (
            Variable(torch.from_numpy(batch).float().cuda()),
            Variable(torch.from_numpy(target).float().cuda()),
        )
        optimizer.zero_grad()
        outputs = network(batch)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

    torch.save(network.state_dict(), args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", help="Directory containing data", type=str)
    # step_size dictates how far the next and previous points are from reference point when creating training patches.
    # For ASOCA reference centerlines, 1 or 2 works well. For more densly annotated centerlines, chose a lower value.
    parser.add_argument(
        "--C",
        help="Dictates distance of nextp and prevp in getMiniBatch.",
        default=2,
        type=int,
    )
    parser.add_argument("--save_path", help="Save path for network", default="tracker.pt", type=str)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
