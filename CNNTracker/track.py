"""
All functions in this code were written by Jelmer M. Wolterink.
Slight adaptations were made to retrain on the ASOCA dataset.
"""
from __future__ import print_function
from tqdm import tqdm
import scipy.spatial as scsp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import SimpleITK as sitk
import scipy.interpolate as scin
import argparse
import warnings

PS = 19
PST = 19
VS = 0.5
VST = 1.0
NV = 500

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore", category=RuntimeWarning)


def fast_trilinear(input_array, x_indices, y_indices, z_indices):
    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
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


def draw_sample_3D_world_fast(image, x, y, z, imagespacing, patchsize, patchspacing):
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

    xs = np.squeeze(coords[0, :]) + (x / imagespacing[0])
    ys = np.squeeze(coords[1, :]) + (y / imagespacing[1])
    zs = np.squeeze(coords[2, :]) + (z / imagespacing[2])

    patch = fast_trilinear(image, xs, ys, zs)

    return patch.reshape(patchsize)


def prepare_image_new(image, spacing, largesize, patchspacing):
    image = np.clip(image, -1024, 3072)

    # PAD IMAGE TO PREVENT PATCHES GOING OUT OF IMAGE
    ps = np.zeros((3))
    ps[0] = np.ceil((largesize * spacing[0]) / patchspacing[0]) * 2
    if ps[0] % 2 == 0:
        ps[0] += 1
    ps[1] = np.ceil((largesize * spacing[1]) / patchspacing[1]) * 2
    if ps[1] % 2 == 0:
        ps[1] += 1
    ps[2] = np.ceil((largesize * spacing[2]) / patchspacing[2]) * 2
    if ps[2] % 2 == 0:
        ps[2] += 1
    ps = ps.astype("int16")

    image = np.pad(
        image,
        (
            (int((ps[0] - 1) / 2), int((ps[0] - 1) / 2)),
            (int((ps[1] - 1) / 2), int((ps[1] - 1) / 2)),
            (int((ps[2] - 1) / 2), int((ps[2] - 1) / 2)),
        ),
        "edge",
    ).astype("float32")

    ps = ps.astype("float32")

    return image, ps


def main_classify(
    image,
    move_fn,
    position,
    ps,
    spacing,
    offset,
    vertices,
    vertices_mask,
    allvessel,
    listvessels,
    ostiag,
    entropythreshold=0.9,
):
    patchspacing = [VS, VS, VS]
    largesize = PS
    position = (
        position
        - np.asarray(offset)
        + [((ps[0] - 1) / 2) * spacing[0], ((ps[1] - 1) / 2) * spacing[1], ((ps[2] - 1) / 2) * spacing[2]]
    )

    ostia = np.copy(ostiag)
    ostia[0, :] = (
        ostia[0, :]
        - np.asarray(offset)
        + [((ps[0] - 1) / 2) * spacing[0], ((ps[1] - 1) / 2) * spacing[1], ((ps[2] - 1) / 2) * spacing[2]]
    )
    ostia[1, :] = (
        ostia[1, :]
        - np.asarray(offset)
        + [((ps[0] - 1) / 2) * spacing[0], ((ps[1] - 1) / 2) * spacing[1], ((ps[2] - 1) / 2) * spacing[2]]
    )

    patchsize = [largesize, largesize, largesize]
    path1 = [np.asarray(position)]
    path2 = [np.asarray(position)]
    startpoint = np.asarray(position)

    # CUMULATIVE LENGTH
    cuml1 = [0.0]
    cuml2 = [0.0]
    radii1 = []
    radii2 = []
    entros1 = []
    entros2 = []

    nsteps = 3000
    firstdirection = 0

    for bidir in range(2):
        if bidir == 0:
            position = path1[0]
        if bidir == 1:
            position = path2[0]

        if bidir == 0:
            lasttarget = "Dog"
        if bidir == 1:
            lasttarget = firstdirection

        for step in range(nsteps):
            samp = draw_sample_3D_world_fast(
                image,
                position[0] - spacing[0] / 2,
                position[1] - spacing[1] / 2,
                position[2] - spacing[2] / 2,
                spacing,
                np.array(patchsize),
                np.array(patchspacing),
            )
            patch = np.zeros((1, 1, patchsize[0], patchsize[1], patchsize[2]), dtype="float32")

            patch[0, 0, :, :, :] = samp.astype("float32")

            patch = Variable(torch.from_numpy(patch).float().cuda())

            outputs = move_fn(patch)
            outputs[:, :-1] = F.softmax(outputs[:, :-1], dim=1)
            classprob = np.squeeze(outputs.data.cpu().numpy())

            entropy = -1.0 * np.sum(np.log2(classprob[:-1]) * classprob[:-1])

            # Normalize entropy
            entropy = entropy / np.log2(float(NV))
            radius = classprob[-1]

            if bidir == 0:
                radii1 = np.concatenate((radii1, [radius]))
            if bidir == 1:
                radii2 = np.concatenate((radii2, [radius]))

            # Termination criterion
            threntropy = entropy

            if bidir == 0:
                entros1 = np.concatenate((entros1, [entropy]))
                if step > 5:
                    threntropy = entros1[-1].mean()
                else:
                    threntropy = 0.5
            if bidir == 1:
                entros2 = np.concatenate((entros2, [entropy]))
                if step > 5:
                    threntropy = entros2[-1].mean()
                else:
                    threntropy = 0.5

            probz = classprob[:-1]

            # Determine targetvertice as the one with the smallest angle to the previous direction
            if step > 0 or bidir == 1:
                probz = probz * vertices_mask[lasttarget, :]
                picked = np.argsort(probz)[-1]
                lasttarget = picked
                ## SELECT STRONGEST
                targetvertice = vertices[picked, :]
            else:
                # Als step == 0 en bidir == 0
                picked = np.argsort(probz)[-1]
                lasttarget = picked
                targetvertice = vertices[picked, :]
                firstdirection = -1.0 * targetvertice
                mini = 0
                mindist = 100.0
                for cl in range(NV):
                    angle = np.arccos(
                        np.dot(
                            firstdirection / np.linalg.norm(firstdirection),
                            vertices[cl, :] / np.linalg.norm(vertices[cl, :]),
                        )
                    )
                    if angle < mindist:
                        mindist = angle
                        mini = cl
                firstdirection = mini

            direction = targetvertice

            step_factor = 1.0
            stepsize_h = step_factor * radius

            # NORMALIZE STEP
            direction = direction / (np.linalg.norm(direction) / stepsize_h)
            position = position + direction
            position = np.squeeze(position)

            stopnow = False
            if bidir == 0:
                # SELECT POINTS THAT HAVE A DISTANCE ALONG THE LINE LARGER DAN 0.6 mm FROM THE CURRENT POINT
                if cuml1[-1] > 0.6:
                    cands = path1[((cuml1[-1] + stepsize_h) - cuml1) > 0.6, :]
                    if np.sum(np.linalg.norm(cands - np.asarray(position), axis=1) < stepsize_h * 0.5) > 0:
                        stopnow = True
                path1 = np.concatenate((path1, [np.asarray(position)]), axis=0)
                cuml1 = np.concatenate((cuml1, [np.asarray(cuml1[-1] + stepsize_h)]))
            if bidir == 1:
                # SELECT POINTS THAT HAVE A DISTANCE ALONG THE LINE LARGER DAN 0.6 mm FROM THE CURRENT POINT
                if cuml2[-1] > 0.6:
                    cands = path2[((cuml2[-1] + stepsize_h) - cuml2) > 0.6, :]
                    if np.sum(np.linalg.norm(cands - np.asarray(position), axis=1) < stepsize_h * 0.5) > 0:
                        stopnow = True
                path2 = np.concatenate((path2, [np.asarray(position)]), axis=0)
                cuml2 = np.concatenate((cuml2, [np.asarray(cuml2[-1] + stepsize_h)]))

            if stopnow:
                break

            voxloc = np.round(position / spacing)
            margin = np.ceil((((patchsize[0] - 1) / 2) * patchspacing[0]) / spacing[0])

            # STOPPING CONDITIONS
            # Entropy too high
            if threntropy > entropythreshold:
                break
            # Outside image
            if (
                np.any(voxloc < margin)
                or voxloc[0] >= image.shape[0] - margin
                or voxloc[1] >= image.shape[1] - margin
                or voxloc[2] >= image.shape[2] - margin
            ):
                break

            # Ostia center point
            cpoint = [
                (ostia[0, 0] + ostia[1, 0]) / 2.0,
                (ostia[0, 1] + ostia[1, 1]) / 2.0,
                (ostia[0, 2] + ostia[1, 2]) / 2.0,
            ]
            radius = np.linalg.norm(ostia[0, :] - ostia[1, :]) * 0.5
            if np.linalg.norm(position - cpoint) < radius:
                break

    path = np.concatenate((np.flipud(path1[:-1, :]), path2[:-1, :]), axis=0)
    radii = np.concatenate((radii1[::-1], radii2))

    # Get right direction!
    if radii.shape[0] > 8:
        radii_tmp = radii[3:-3]
        tpoints = range(np.prod(radii_tmp.shape))
        par = np.polyfit(tpoints, radii_tmp, 1, full=True)
        slope = par[0][0]
        if slope > 0.0:
            path = np.flipud(path)
            radii = radii[::-1]

    cuml = np.cumsum(radii * step_factor)

    if np.max(cuml) > 275.0:
        cutpoint = np.argmin(np.abs(cuml - 275.0))
        if np.min(np.linalg.norm(path[:cutpoint] - startpoint, axis=1)) == 0.0:
            path = path[:cutpoint]
            radii = radii[:cutpoint]
        else:
            path = path[cutpoint:]
            radii = radii[cutpoint:]

    minradid = np.argmin(radii)
    if minradid > 2 and radii[minradid] < 1.0:
        if np.min(np.linalg.norm(path[:minradid] - startpoint, axis=1)) == 0.0:
            path = path[:minradid]
            radii = radii[:minradid]
        else:
            path = path[minradid:]
            radii = radii[minradid:]

    # Get right direction!
    if radii.shape[0] > 8:
        radii_tmp = radii[3:-3]
        tpoints = range(np.prod(radii_tmp.shape))
        par = np.polyfit(tpoints, radii_tmp, 1, full=True)
        slope = par[0][0]
        if slope > 0.0:
            path = np.flipud(path)
            radii = radii[::-1]

    po = (
        path
        + offset
        - [((ps[0] - 1) / 2) * spacing[0], ((ps[1] - 1) / 2) * spacing[1], ((ps[2] - 1) / 2) * spacing[2]]
    )
    radii = radii.reshape((radii.shape[0], 1))

    newvessel = np.concatenate((po, radii), axis=1)
    newvessel = resample(newvessel, resolution=0.03)

    if np.min(scsp.distance.cdist(path, ostia)) < 8.0:
        allvessel = np.concatenate((allvessel, newvessel), axis=0)
        listvessels.append(newvessel)

    return allvessel, listvessels


class CNNTracking(nn.Module):
    def __init__(self):
        super(CNNTracking, self).__init__()
        C = 32
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=C, kernel_size=3, dilation=1)
        self.bn1 = nn.BatchNorm3d(num_features=C)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1)
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


def resample(path, resolution):
    if path.shape[0] < 8:
        return path

    # Remove duplicate rows!
    pathc = np.zeros((1, 4), dtype="float32")
    pathc[0, :] = path[0, :]
    for i in range(1, path.shape[0]):
        if np.linalg.norm(path[i, :] - path[i - 1, :]) > 0.1:
            pathc = np.concatenate((pathc, path[i, :].reshape((1, 4))), axis=0)
    path = pathc

    # Resample to equidistance
    ptd = np.zeros((path.shape[0]))
    for i in range(1, path.shape[0]):
        distp = np.linalg.norm(path[i, :3] - path[i - 1, :3])
        ptd[i] = distp + ptd[i - 1]

    x = ptd
    y_loc = path[:, :3]
    f_loc = scin.interp1d(x, y_loc, kind="cubic", axis=0)

    y_rad = path[:, 3]
    f_rad = scin.interp1d(x, y_rad, kind="cubic", axis=0)

    xnew = np.arange(0.0, np.max(ptd), resolution)
    pathnew_loc = f_loc(xnew)
    pathnew_rad = f_rad(xnew)
    pathnew = np.concatenate((pathnew_loc, pathnew_rad.reshape((pathnew_rad.shape[0], 1))), axis=1)

    return pathnew


def track(args, image, ps, seeds, ostia):
    network = CNNTracking()
    parameters = torch.load(args.tracknet)
    newdict = {}
    for key in parameters:
        if key.replace(r"module.", "") in network.state_dict():
            newdict[key.replace(r".module", "")] = parameters[key]
    parameters = newdict
    network.load_state_dict(parameters)
    network.to(args.device)
    network.float()
    network.eval()

    vertices = np.loadtxt(r"vertices500.txt")
    vertices_mask = np.loadtxt(r"vertices500_mask.txt")

    allvessel = np.zeros((0, 4), dtype="float32")
    usedseeds = np.zeros((0, 3), dtype="float32")

    listvessels = []

    va = 0

    for v in tqdm(range(seeds.shape[0])):
        seed = seeds[v, :]
        processthis = True
        if allvessel.shape[0] > 0:
            distm = np.linalg.norm(allvessel[:, :3] - seed[:3], axis=1)
            if np.min(distm) < 1.5:
                processthis = False

        if processthis:
            allvessel, listvessels = main_classify(
                image,
                network,
                seed,
                ps,
                spacing,
                offset,
                vertices,
                vertices_mask,
                allvessel,
                listvessels,
                ostia,
                args.entropythreshold,
            )
            usedseeds = np.concatenate((usedseeds, np.reshape(seed, (1, 3))), axis=0)

            va = va + 1

    # Clean up listvessels
    toremove = []

    connected_ostia = np.zeros((len(listvessels)))
    for i in range(len(listvessels)):
        distm = scsp.distance.cdist(listvessels[i][:, :3], ostia)
        if np.min(distm) < 8.0:
            connected_ostia[i] = 1
        else:
            toremove.append(i)
        if listvessels[i].shape[0] < 500:
            toremove.append(i)

    selvessels = []
    for i in range(len(listvessels)):
        if i not in toremove:
            selvessels.append(listvessels[i])

    print("Found {} centerlines".format(len(selvessels)))
    for s in range(len(selvessels)):
        np.savetxt(r"{}/vessel{}.txt".format(args.outdir, s), selvessels[s])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impath", help="Path to image file (should be SimpleITK readable)", type=str)
    parser.add_argument("--outdir", help="Where should output files be written?", default="results", type=str)
    parser.add_argument("--ostia", help="Path to ostia file", type=str)
    parser.add_argument("--seeds", help="Path to seed file", type=str)
    parser.add_argument("--tracknet", help="Path to tracking network", default="tracker.pt", type=str)
    parser.add_argument("--entropythreshold", help="Entropy threshold", default=0.85, type=float)

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(args.outdir, os.path.basename(args.impath)), exist_ok=True)

    image = sitk.ReadImage(args.impath)

    offset = image.GetOrigin()
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, 2)
    image = image.astype("float32")

    if np.min(image) == 0:
        image = image - 1024.0

    imbase = os.path.split(args.impath)[-1].replace(".mhd", "")

    seeds = np.loadtxt(args.seeds)
    ostia = np.loadtxt(args.ostia)

    image, ps = prepare_image_new(image, spacing, PS, [VS, VS, VS])
    track(args, image, ps, seeds, ostia)
