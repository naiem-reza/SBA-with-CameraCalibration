import os
import copy
import time
import math
import scipy
import numpy as np
import pandas as pd
import xmltodict
from scipy.stats import norm
from ismember import ismember
import matplotlib.pyplot as plt
from matplotlib import image
from numpy.linalg import inv as npinv
from scipy.sparse import lil_matrix, bsr_matrix, bmat, hstack, vstack
from scipy.sparse.linalg import spsolve

# --------------------------------- Read Cameras EOPs and IOP File
def ReadEOPFile(eop_path):
    with open(eop_path, 'r') as ALL_EOP:
        EOPFile = ALL_EOP.readlines()
    neop = len(EOPFile) - 2
    PhotoID = [];    X = [];    Y = [];    Z = [];    omega = [];    phi = [];    kappa = [];    r11 = []
    r12 = [];    r13 = [];    r21 = [];    r22 = [];    r23 = [];    r31 = [];    r32 = [];    r33 = []
    for row in EOPFile[2: len(EOPFile)]:
        PhotoID.append(str(row.split('\t')[0]))
        X.append(float(row.split('\t')[1]))
        Y.append(float(row.split('\t')[2]))
        Z.append(float(row.split('\t')[3]))
        omega.append(float(row.split('\t')[4]))
        phi.append(float(row.split('\t')[5]))
        kappa.append(float(row.split('\t')[6]))

    #  imgeop = [XO, YO, ZO, Omega, Phi, Kappa, r11, r12, ... , r32, r33]
    imgeop = np.vstack((np.array(X),np.array(Y),np.array(Z),np.array(omega),np.array(phi),np.array(kappa))).transpose()
    return imgeop
def ReadIOPFile(cam_path):
    xml_data = open(cam_path, 'r').read()  # Read data
    data = xmltodict.parse(xml_data)  # Parse XML
    iop = []
    for i in data['calibration']:
        child = data['calibration'][i]
        iop.append(child)

    if len(iop) <= 5:
        w = float(iop[1])
        h = float(iop[2])
        f = float(iop[3])
        IOP = [w, h, f, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    else:
        w = float(iop[1])
        h = float(iop[2])
        f = float(iop[3])
        cx = float(iop[4])
        cy = float(iop[5])
        b1 = float(iop[6])
        b2 = float(iop[7])
        k1 = float(iop[8])
        k2 = float(iop[9])
        k3 = float(iop[10])
        p1 = float(iop[11])
        p2 = float(iop[12])
        IOP = [w, h, f, cx, cy, b1, b2, k1, k2, k3, p1, p2]
    return IOP
def ReadIOPFile_Australis(cam_path):
    with open(cam_path, 'r') as ALL_IOP:
        IOPFile = ALL_IOP.readlines()
    w = float(IOPFile[15][7:11])
    h = float(IOPFile[14][7:11])
    f = float(IOPFile[19][13:19])
    cx = float(IOPFile[20][12:19])
    cy = float(IOPFile[21][12:19])
    k1 = float(IOPFile[22][7:19])
    k2 = float(IOPFile[23][7:19])
    k3 = float(IOPFile[24][7:19])
    k4 = 0
    p1 = float(IOPFile[25][7:19])
    p2 = float(IOPFile[26][7:19])
    b1 = float(float(IOPFile[27][7:19]))
    b2 = float(float(IOPFile[28][7:19]))
    ps = float(float(IOPFile[14][22:32]))

    IOP = [h, w, f, cx, cy, b1, b2, k1, k2, k3, k4, p1, p2]
    return IOP
def ReadEOPFile1(eop_path):
    with open(eop_path, 'r') as ALL_EOP:
        EOPFile = ALL_EOP.readlines()
    neop = len(EOPFile) - 3
    PhotoID = []
    X = []
    Y = []
    Z = []
    omega = []
    phi = []
    kappa = []
    for row in EOPFile[2: len(EOPFile)]:
        PhotoID.append(str(row.split(',')[0]))
        X.append(float(row.split(',')[1]))
        Y.append(float(row.split(',')[2]))
        Z.append(float(row.split(',')[3]))
        omega.append(float(row.split(',')[4]))
        phi.append(float(row.split(',')[5]))
        kappa.append(float(row.split(',')[6]))

    #  imgeop = [XO, YO, ZO, Omega, Phi, Kappa, r11, r12, ... , r32, r33]
    imgeop = np.vstack((np.array(X), np.array(Y), np.array(Z), np.array(omega), np.array(phi), np.array(kappa))).transpose()
    return imgeop

# --------------------------------- XYZ of Tie points
def ReadXYZTieFile(xyz_path):
    ID = [];    xx = [];    yy = [];    zz = []
    with open(xyz_path, 'r') as ALL_Tie:
        TieFile = ALL_Tie.readlines()
    for row in TieFile[0: len(TieFile)]:
        ID.append(int(row.split(',')[0]))
        xx.append(float(row.split(',')[1]))
        yy.append(float(row.split(',')[2]))
        zz.append(float(row.split(',')[3]))
    #  XYZtie = [Number , X , Y , Z]
    XYZTie = np.vstack((np.array(ID), np.array(xx), np.array(yy), np.array(zz))).transpose()
    return XYZTie

# --------------------------------- Read image observation of Tie points
def readimgTieobs(obs_path,TieXYZ):
    os.chdir(obs_path)
    Tie_obs = []

    # iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{obs_path}/{file}"
            ID = []
            xx = []
            yy = []
            with open(file_path, 'r') as f:
                cor = f.readlines()
            for row in cor[0: len(cor)]:
                ID.append(int(row.split(',')[0]))
                xx.append(float(row.split(',')[1]))
                yy.append(float(row.split(',')[2]))
            xyimg = np.vstack((np.array(ID), np.array(xx), np.array(yy))).transpose()
            [Ixyz , Ixy] = ismember(TieXYZ[:,0] , xyimg[:,0])
            xyimg = xyimg[Ixy]
            Tie_obs.append(xyimg)
    return(Tie_obs)

# --------------------------------- Import Data
def Import_Data(dir_files):
    eop_path = dir_files + '/Step_1/eop_init.txt'
    cam_init_path = dir_files + '/Step_1/iop_init.xml'
    xyz_path = dir_files + '/Step_2/xyz.txt'
    Tieobs_path = dir_files + '/Step_2/imgdata'

    EOP_init = ReadEOPFile1(eop_path)
    IOP_init = ReadIOPFile(cam_init_path)
    TieXYZ = ReadXYZTieFile(xyz_path)
    TieObs = readimgTieobs(Tieobs_path,TieXYZ)

    return EOP_init, np.asarray(IOP_init[2:]), TieXYZ, TieObs

def Control_from_xml(dir_files, Control_start_id):
    gcp_path = dir_files + '/Step_1/gcp.xml'
    xml_data = open(gcp_path, 'r').read()  # Read data
    data = xmltodict.parse(xml_data)  # Parse XML
    marker = data['document']['chunk']['markers']['marker']
    obs = data['document']['chunk']['frames']['frame']['markers']['marker']
    for i in range(len(marker)):
        marker[i]['@id'] = str(i)
        marker[i]['@label'] = str(i)
        obs[i]['@marker_id'] = str(i)

    CP_LB, CH_LB, Control, Check, Control_Obs, Check_Obs = [], [], [], [], [], []
    for i in range(len(marker)):
        CP_LB.append(marker[i]['@label'])
        Control.append(np.asarray([marker[i]['@id'], marker[i]['reference']['@x'], marker[i]['reference']['@y'], marker[i]['reference']['@z']], dtype=float))
        for j in range(len(obs[i]['location'])):
            Control_Obs.append(np.asarray([obs[i]['location'][j]['@camera_id'], marker[i]['@id'], obs[i]['location'][j]['@x'], obs[i]['location'][j]['@y']], dtype=float))

    Control = np.asarray(Control)
    Cnt = pd.DataFrame(np.asarray(Control_Obs)).sort_values([1, 0])
    unique_index = pd.unique(Cnt[1])
    Cnt[1] = np.digitize(Cnt[1], unique_index) - 1 + Control_start_id
    Control[:, 0] = np.digitize(Control[:, 0], unique_index) - 1 + Control_start_id
    Control_Obs = Cnt.to_numpy()

    return Control, Control_Obs

def preprocess(TieXYZ, TieObs, EOP):
    PI = math.pi
    UV_ud = []
    imgnum = len(EOP)
    for i in range(imgnum):
        num = len(TieObs[i])
        if len(UV_ud) == 0:
            UV_ud = np.column_stack(((i * np.ones([num])).astype(int), TieObs[i]))
        else:
            UV_ud = np.vstack((UV_ud, np.column_stack(((i * np.ones([num])).astype(int), TieObs[i]))))

    Observation_Tie = pd.DataFrame(UV_ud).sort_values([1, 0])
    unique_index = pd.unique(Observation_Tie[1])
    Observation_Tie[1] = np.digitize(Observation_Tie[1], unique_index) - 1
    TieXYZ[:, 0] = np.digitize(TieXYZ[:, 0], unique_index) - 1
    camera_params = np.column_stack((EOP[:, :3], EOP[:, 3:]*(PI/180)))
    return Observation_Tie.to_numpy(), TieXYZ, camera_params

def PLOT_iter(RES_IMG, PHI, RES_OBJ):
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    ax1.plot(np.abs(np.asarray(RES_IMG[:, 0])), 'r', label='x')
    ax1.plot(np.abs(np.asarray(RES_IMG[:, 1])), 'b', label='y')
    ax1.set_title('RMSE of Image Residuals')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('image residuals (pixel)')
    ax1.legend(loc="best")
    ax1.grid(True)

    ax2.plot(PHI, 'r')
    ax2.set_title('Phi')
    ax2.set_xlabel('Iteration')
    ax2.grid(True)

    ax3.plot(np.abs(np.asarray(RES_OBJ[:, 0])), 'r', label='X')
    ax3.plot(np.abs(np.asarray(RES_OBJ[:, 1])), 'g', label='Y')
    ax3.plot(np.abs(np.asarray(RES_OBJ[:, 2])), 'b', label='Z')
    ax3.set_title('RMSE of Object Residuals')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Object residuals (m)')
    ax3.legend(loc="best")
    ax3.grid(True)

def Plot_residual_hist(dl_last):
    fig, [ax1 , ax2] = plt.subplots(1,2)
    ax1.hist(dl_last, bins=100, density=True)
    ax1.grid(True)
    [mean_fit, std_fit] = scipy.stats.norm.fit(dl_last)
    x = np.linspace(dl_last.min(), dl_last.max())
    ax1.plot(x, scipy.stats.norm.pdf(x, mean_fit, std_fit))
    ax1.set_xlabel('residuals of Tie Distance')
    ax1.set_ylabel('Number of Tie')
    ax1.set_title('Mean_fit:%0.2f' % mean_fit + '   std_fit:%0.2f' % std_fit)

    ax2.hist(dl_last, bins=100, density=True)
    ax2.grid(True)
    [scale_fit, mean_fit, std_fit] = scipy.stats.t.fit(dl_last)
    x = np.linspace(dl_last.min(), dl_last.max())
    ax2.plot(x, scipy.stats.t.pdf(x, scale_fit, mean_fit, std_fit))
    ax2.set_xlabel('residuals of Tie Distance')
    ax2.set_ylabel('Number of Tie')
    ax2.set_title('Scale_fit:%0.2f' % scale_fit + '   Mean_fit:%0.2f' % mean_fit + '   std_fit:%0.2f' % std_fit)

def show_residual(dir_files, Observation, Udpoint2d_pix, num):
    gcp_path = dir_files + '/Step_1/gcp.xml'
    xml_data = open(gcp_path, 'r').read()  # Read data
    data = xmltodict.parse(xml_data)  # Parse XML
    camera = data['document']['chunk']['cameras']['camera']
    cam_label = []
    for i in range(len(camera)):
        cam_label.append(camera[i]['@label'])

    udp = copy.deepcopy(Observation)
    udp[:, 2:] = Udpoint2d_pix
    udp = udp[np.argsort(udp[:, 0])]
    obs = copy.deepcopy(Observation)
    obs = obs[np.argsort(obs[:, 0])]

    fig = plt.figure()
    imgdir = dir_files + '/Image/' + cam_label[num] + '.JPG'
    data = image.imread(imgdir)
    [I1, I2] = ismember(udp[:, 0].astype(int), np.array(num).astype(int))

    xn = udp[I1, 2]
    yn = udp[I1, 3]
    x = obs[I1, 2]
    y = obs[I1, 3]

    x1 = [xn, x]
    y1 = [yn, y]
    plt.plot(x1, y1, color="black", linewidth=2)

    plt.plot(xn, yn, '.r', label='re-project')
    plt.plot(x, y, '.b', label='Observation')
    plt.imshow(data)
    plt.show()
    plt.legend()

def inverse_block_diag_sparse(A):
    V = bsr_matrix(A)
    size = V.blocksize
    iV = copy.deepcopy(V)

    data = V.data
    idata = np.zeros((len(data), size[0], size[1]))
    for k in range(len(data)):
        idata[k, :, :] = npinv(data[k])

    iV.data = idata
    return iV.tocsr()

def ComputeWeight(Info_BA, sigma_img, sigma_obj):
    Tie_indices = Info_BA.Tie_indices
    vec_sigma_obj = sigma_obj * np.ones(len(Info_BA.GCP))
    sigma_OBJ = np.column_stack((vec_sigma_obj, vec_sigma_obj, vec_sigma_obj)).ravel()
    m = (len(Tie_indices) * 2) + len(sigma_OBJ)
    Weight = lil_matrix((m, m), dtype=float)
    ii = np.arange(len(Tie_indices) * 2)
    Weight[ii, ii] = sigma_img
    ii = np.arange(len(Tie_indices) * 2, len(Tie_indices) * 2 + len(sigma_OBJ))
    Weight[ii, ii] = sigma_OBJ
    return Weight.tocsr()

class jac_element():
    def __init__(self, IOP, eop, XYZ):
        f = IOP[0]
        cx = IOP[1]
        cy = IOP[2]
        B1 = IOP[3]
        B2 = IOP[4]
        K1 = IOP[5]
        K2 = IOP[6]
        K3 = IOP[7]
        P1 = IOP[8]
        P2 = IOP[9]

        X = XYZ[:, 0]
        Y = XYZ[:, 1]
        Z = XYZ[:, 2]

        X0 = eop[:, 0]
        Y0 = eop[:, 1]
        Z0 = eop[:, 2]
        w = eop[:, 3]
        p = eop[:, 4]
        k = eop[:, 5]

        cw = np.cos(w)
        cp = np.cos(p)
        ck = np.cos(k)
        sw = np.sin(w)
        sp = np.sin(p)
        sk = np.sin(k)

        T1 = (cw * sk + ck * sp * sw)
        T2 = (sk * sw - ck * cw * sp)
        T3 = (ck * sw + cw * sk * sp)
        T4 = (ck * cw - sk * sp * sw)
        T5 = (Y - Y0) * T1 + (Z - Z0) * T2
        T9 = (sp * (X - X0) + cp * cw * (Z - Z0) - cp * sw * (Y - Y0))
        T12 = ((Y - Y0) * T4 + (Z - Z0) * T3 - cp * sk * (X - X0))
        T13 = (T5 + ck * cp * (X - X0))
        T6 = (T13 ** 2 / T9 ** 2 + T12 ** 2 / T9 ** 2)
        T7 = (ck * sp * (X - X0) + ck * cp * cw * (Z - Z0) - ck * cp * sw * (Y - Y0))
        T8 = (K1 * T6 + K2 * T6 ** 2 + K3 * T6 ** 3 + 1)
        T10 = (cp * cw * (Y - Y0) + cp * sw * (Z - Z0))
        T11 = (2 * cp * cw * T13 ** 2) / T9 ** 3

        t1 = (P2 * ((2 * T2 * T13) / T9 ** 2 + (6 * T3 * T12) / T9 ** 2 - T11 - (6 * cp * cw * T12 ** 2) / T9 ** 3) + (T3 * T8) / T9 + ((K1 * ((2 * T2 * T13) / T9 ** 2 + (2 * T3 * T12) / T9 ** 2 - T11 - (2 * cp * cw * T12 ** 2) / T9 ** 3) + 3 * K3 * T6 ** 2 * ((2 * T2 * T13) / T9 ** 2 + (2 * T3 * T12) / T9 ** 2 - T11 - (2 * cp * cw * T12 ** 2) / T9 ** 3) + 2 * K2 * T6 * ((2 * T2 * T13) / T9 ** 2 + (2 * T3 * T12) / T9 ** 2 - T11 - (2 * cp * cw * T12 ** 2) / T9 ** 3)) * T12) / T9 - (2 * P1 * T2 * T12) / T9 ** 2 - (2 * P1 * T3 * T13) / T9 ** 2 - (cp * cw * T12 * T8) / T9 ** 2 + (4 * P1 * cp * cw * T13 * T12) / T9 ** 3)
        t2 = (P2 * ((2 * T1 * T13) / T9 ** 2 + (6 * T4 * T12) / T9 ** 2 + (2 * cp * sw * T13 ** 2) / T9 ** 3 + (6 * cp * sw * T12 ** 2) / T9 ** 3) + (T4 * T8) / T9 + (T12 * (K1 * ((2 * T1 * T13) / T9 ** 2 + (2 * T4 * T12) / T9 ** 2 + (2 * cp * sw * T13 ** 2) / T9 ** 3 + (2 * cp * sw * T12 ** 2) / T9 ** 3) + 3 * K3 * T6 ** 2 * ((2 * T1 * T13) / T9 ** 2 + (2 * T4 * T12) / T9 ** 2 + (2 * cp * sw * T13 ** 2) / T9 ** 3 + (2 * cp * sw * T12 ** 2) / T9 ** 3) + 2 * K2 * T6 * ((2 * T1 * T13) / T9 ** 2 + (2 * T4 * T12) / T9 ** 2 + (2 * cp * sw * T13 ** 2) / T9 ** 3 + (2 * cp * sw * T12 ** 2) / T9 ** 3))) / T9 - (2 * P1 * T1 * T12) / T9 ** 2 - (2 * P1 * T4 * T13) / T9 ** 2 + (cp * sw * T12 * T8) / T9 ** 2 - (4 * P1 * cp * sw * T13 * T12) / T9 ** 3)
        t3 = (P2 * ((2 * sp * T13 ** 2) / T9 ** 3 + (6 * sp * T12 ** 2) / T9 ** 3 - (2 * ck * cp * T13) / T9 ** 2 + (6 * cp * sk * T12) / T9 ** 2) + (T12 * (K1 * ((2 * sp * T13 ** 2) / T9 ** 3 + (2 * sp * T12 ** 2) / T9 ** 3 - (2 * ck * cp * T13) / T9 ** 2 + (2 * cp * sk * T12) / T9 ** 2) + 2 * K2 * T6 * ((2 * sp * T13 ** 2) / T9 ** 3 + (2 * sp * T12 ** 2) / T9 ** 3 - (2 * ck * cp * T13) / T9 ** 2 + (2 * cp * sk * T12) / T9 ** 2) + 3 * K3 * T6 ** 2 * ((2 * sp * T13 ** 2) / T9 ** 3 + (2 * sp * T12 ** 2) / T9 ** 3 - (2 * ck * cp * T13) / T9 ** 2 + (2 * cp * sk * T12) / T9 ** 2))) / T9 + (sp * T12 * T8) / T9 ** 2 + (cp * sk * T8) / T9 - (2 * P1 * cp * sk * T13) / T9 ** 2 + (2 * P1 * ck * cp * T12) / T9 ** 2 - (4 * P1 * sp * T13 * T12) / T9 ** 3)
        t4 = ((T2 * T8) / T9 - P1 * ((6 * T2 * T13) / T9 ** 2 + (2 * T3 * T12) / T9 ** 2 - (6 * cp * cw * T13 ** 2) / T9 ** 3 - (2 * cp * cw * T12 ** 2) / T9 ** 3) + ((K1 * ((2 * T2 * T13) / T9 ** 2 + (2 * T3 * T12) / T9 ** 2 - T11 - (2 * cp * cw * T12 ** 2) / T9 ** 3) + 3 * K3 * T6 ** 2 * ((2 * T2 * T13) / T9 ** 2 + (2 * T3 * T12) / T9 ** 2 - T11 - (2 * cp * cw * T12 ** 2) / T9 ** 3) + 2 * K2 * T6 * ((2 * T2 * T13) / T9 ** 2 + (2 * T3 * T12) / T9 ** 2 - T11 - (2 * cp * cw * T12 ** 2) / T9 ** 3)) * T13) / T9 + (2 * P2 * T2 * T12) / T9 ** 2 + (2 * P2 * T3 * T13) / T9 ** 2 - (cp * cw * T13 * T8) / T9 ** 2 - (4 * P2 * cp * cw * T13 * T12) / T9 ** 3)
        t5 = ((T1 * T8) / T9 - P1 * ((6 * T1 * T13) / T9 ** 2 + (2 * T4 * T12) / T9 ** 2 + (6 * cp * sw * T13 ** 2) / T9 ** 3 + (2 * cp * sw * T12 ** 2) / T9 ** 3) + (T13 * (K1 * ((2 * T1 * T13) / T9 ** 2 + (2 * T4 * T12) / T9 ** 2 + (2 * cp * sw * T13 ** 2) / T9 ** 3 + (2 * cp * sw * T12 ** 2) / T9 ** 3) + 3 * K3 * T6 ** 2 * ((2 * T1 * T13) / T9 ** 2 + (2 * T4 * T12) / T9 ** 2 + (2 * cp * sw * T13 ** 2) / T9 ** 3 + (2 * cp * sw * T12 ** 2) / T9 ** 3) + 2 * K2 * T6 * ((2 * T1 * T13) / T9 ** 2 + (2 * T4 * T12) / T9 ** 2 + (2 * cp * sw * T13 ** 2) / T9 ** 3 + (2 * cp * sw * T12 ** 2) / T9 ** 3))) / T9 + (2 * P2 * T1 * T12) / T9 ** 2 + (2 * P2 * T4 * T13) / T9 ** 2 + (cp * sw * T13 * T8) / T9 ** 2 + (4 * P2 * cp * sw * T13 * T12) / T9 ** 3)
        t6 = ((T13 * (K1 * ((2 * sp * T13 ** 2) / T9 ** 3 + (2 * sp * T12 ** 2) / T9 ** 3 - (2 * ck * cp * T13) / T9 ** 2 + (2 * cp * sk * T12) / T9 ** 2) + 2 * K2 * T6 * ((2 * sp * T13 ** 2) / T9 ** 3 + (2 * sp * T12 ** 2) / T9 ** 3 - (2 * ck * cp * T13) / T9 ** 2 + (2 * cp * sk * T12) / T9 ** 2) + 3 * K3 * T6 ** 2 * ((2 * sp * T13 ** 2) / T9 ** 3 + (2 * sp * T12 ** 2) / T9 ** 3 - (2 * ck * cp * T13) / T9 ** 2 + (2 * cp * sk * T12) / T9 ** 2))) / T9 - P1 * ((6 * sp * T13 ** 2) / T9 ** 3 + (2 * sp * T12 ** 2) / T9 ** 3 - (6 * ck * cp * T13) / T9 ** 2 + (2 * cp * sk * T12) / T9 ** 2) + (sp * T13 * T8) / T9 ** 2 - (ck * cp * T8) / T9 + (2 * P2 * cp * sk * T13) / T9 ** 2 - (2 * P2 * ck * cp * T12) / T9 ** 2 + (4 * P2 * sp * T13 * T12) / T9 ** 3)
        t7 = (P2 * ((2 * T10 * T13 ** 2) / T9 ** 3 + (6 * T10 * T12 ** 2) / T9 ** 3 - (2 * ((Y - Y0) * T2 - (Z - Z0) * T1) * T13) / T9 ** 2 - (6 * ((Y - Y0) * T3 - (Z - Z0) * T4) * T12) / T9 ** 2) + ((K1 * ((2 * T10 * T13 ** 2) / T9 ** 3 + (2 * T10 * T12 ** 2) / T9 ** 3 - (2 * ((Y - Y0) * T2 - (Z - Z0) * T1) * T13) / T9 ** 2 - (2 * ((Y - Y0) * T3 - (Z - Z0) * T4) * T12) / T9 ** 2) + 2 * K2 * T6 * ((2 * T10 * T13 ** 2) / T9 ** 3 + (2 * T10 * T12 ** 2) / T9 ** 3 - (2 * ((Y - Y0) * T2 - (Z - Z0) * T1) * T13) / T9 ** 2 - (2 * ((Y - Y0) * T3 - (Z - Z0) * T4) * T12) / T9 ** 2) + 3 * K3 * T6 ** 2 * ((2 * T10 * T13 ** 2) / T9 ** 3 + (2 * T10 * T12 ** 2) / T9 ** 3 - (2 * ((Y - Y0) * T2 - (Z - Z0) * T1) * T13) / T9 ** 2 - (2 * ((Y - Y0) * T3 - (Z - Z0) * T4) * T12) / T9 ** 2)) * T12) / T9 - (((Y - Y0) * T3 - (Z - Z0) * T4) * T8) / T9 + (2 * P1 * ((Y - Y0) * T3 - (Z - Z0) * T4) * T13) / T9 ** 2 + (2 * P1 * ((Y - Y0) * T2 - (Z - Z0) * T1) * T12) / T9 ** 2 + (T10 * T12 * T8) / T9 ** 2 - (4 * P1 * T10 * T13 * T12) / T9 ** 3)
        t8 = (P2 * ((2 * T13 * T7) / T9 ** 2 - (6 * T12 * (sk * sp * (X - X0) + cp * cw * sk * (Z - Z0) - cp * sk * sw * (Y - Y0))) / T9 ** 2 + (2 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T13 ** 2) / T9 ** 3 + (6 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T12 ** 2) / T9 ** 3) + ((K1 * ((2 * T13 * T7) / T9 ** 2 - (2 * T12 * (sk * sp * (X - X0) + cp * cw * sk * (Z - Z0) - cp * sk * sw * (Y - Y0))) / T9 ** 2 + (2 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T13 ** 2) / T9 ** 3 + (2 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T12 ** 2) / T9 ** 3) + 2 * K2 * T6 * ((2 * T13 * T7) / T9 ** 2 - (2 * T12 * (sk * sp * (X - X0) + cp * cw * sk * (Z - Z0) - cp * sk * sw * (Y - Y0))) / T9 ** 2 + (2 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T13 ** 2) / T9 ** 3 + (2 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T12 ** 2) / T9 ** 3) + 3 * K3 * T6 ** 2 * ((2 * T13 * T7) / T9 ** 2 - (2 * T12 * (sk * sp * (X - X0) + cp * cw * sk * (Z - Z0) - cp * sk * sw * (Y - Y0))) / T9 ** 2 + (2 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T13 ** 2) / T9 ** 3 + (2 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T12 ** 2) / T9 ** 3)) * T12) / T9 - ((sk * sp * (X - X0) + cp * cw * sk * (Z - Z0) - cp * sk * sw * (Y - Y0)) * T8) / T9 - (2 * P1 * T12 * T7) / T9 ** 2 + (2 * P1 * T13 * (sk * sp * (X - X0) + cp * cw * sk * (Z - Z0) - cp * sk * sw * (Y - Y0))) / T9 ** 2 + ((cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T12 * T8) / T9 ** 2 - (4 * P1 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T13 * T12) / T9 ** 3)
        t9 = ((T13 * T8) / T9 - (2 * P1 * T13 ** 2) / T9 ** 2 + (2 * P1 * T12 ** 2) / T9 ** 2 + (4 * P2 * T13 * T12) / T9 ** 2)

        self.df1_dX0 = B1*t6 + f*t6 - B2*t3
        self.df2_dX0 = -f*t3

        self.df1_dY0 = B2*t2 - B1*t5 - f*t5
        self.df2_dY0 = f*t2

        self.df1_dZ0 = B2*t1 - B1*t4 - f*t4
        self.df2_dZ0 = f*t1

        self.df1_dw = - B1*(P1*((6*T10*T13**2)/T9**3 + (2*T10*T12**2)/T9**3 - (6*((Y - Y0)*T2 - (Z - Z0)*T1)*T13)/T9**2 - (2*((Y - Y0)*T3 - (Z - Z0)*T4)*T12)/T9**2) - ((K1*((2*T10*T13**2)/T9**3 + (2*T10*T12**2)/T9**3 - (2*((Y - Y0)*T2 - (Z - Z0)*T1)*T13)/T9**2 - (2*((Y - Y0)*T3 - (Z - Z0)*T4)*T12)/T9**2) + 2*K2*T6*((2*T10*T13**2)/T9**3 + (2*T10*T12**2)/T9**3 - (2*((Y - Y0)*T2 - (Z - Z0)*T1)*T13)/T9**2 - (2*((Y - Y0)*T3 - (Z - Z0)*T4)*T12)/T9**2) + 3*K3*T6**2*((2*T10*T13**2)/T9**3 + (2*T10*T12**2)/T9**3 - (2*((Y - Y0)*T2 - (Z - Z0)*T1)*T13)/T9**2 - (2*((Y - Y0)*T3 - (Z - Z0)*T4)*T12)/T9**2))*T13)/T9 + (((Y - Y0)*T2 - (Z - Z0)*T1)*T8)/T9 + (2*P2*((Y - Y0)*T3 - (Z - Z0)*T4)*T13)/T9**2 + (2*P2*((Y - Y0)*T2 - (Z - Z0)*T1)*T12)/T9**2 - (T10*T13*T8)/T9**2 - (4*P2*T10*T13*T12)/T9**3) - B2*t7 - f*(P1*((6*T10*T13**2)/T9**3 + (2*T10*T12**2)/T9**3 - (6*((Y - Y0)*T2 - (Z - Z0)*T1)*T13)/T9**2 - (2*((Y - Y0)*T3 - (Z - Z0)*T4)*T12)/T9**2) - ((K1*((2*T10*T13**2)/T9**3 + (2*T10*T12**2)/T9**3 - (2*((Y - Y0)*T2 - (Z - Z0)*T1)*T13)/T9**2 - (2*((Y - Y0)*T3 - (Z - Z0)*T4)*T12)/T9**2) + 2*K2*T6*((2*T10*T13**2)/T9**3 + (2*T10*T12**2)/T9**3 - (2*((Y - Y0)*T2 - (Z - Z0)*T1)*T13)/T9**2 - (2*((Y - Y0)*T3 - (Z - Z0)*T4)*T12)/T9**2) + 3*K3*T6**2*((2*T10*T13**2)/T9**3 + (2*T10*T12**2)/T9**3 - (2*((Y - Y0)*T2 - (Z - Z0)*T1)*T13)/T9**2 - (2*((Y - Y0)*T3 - (Z - Z0)*T4)*T12)/T9**2))*T13)/T9 + (((Y - Y0)*T2 - (Z - Z0)*T1)*T8)/T9 + (2*P2*((Y - Y0)*T3 - (Z - Z0)*T4)*T13)/T9**2 + (2*P2*((Y - Y0)*T2 - (Z - Z0)*T1)*T12)/T9**2 - (T10*T13*T8)/T9**2 - (4*P2*T10*T13*T12)/T9**3)
        self.df2_dw = -f*t7

        self.df1_dp = B2*t8 - f*((T7*T8)/T9 - P1*((6*T13*T7)/T9**2 - (2*T12*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + (6*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13**2)/T9**3 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T12**2)/T9**3) + ((K1*((2*T13*T7)/T9**2 - (2*T12*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13**2)/T9**3 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T12**2)/T9**3) + 2*K2*T6*((2*T13*T7)/T9**2 - (2*T12*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13**2)/T9**3 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T12**2)/T9**3) + 3*K3*T6**2*((2*T13*T7)/T9**2 - (2*T12*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13**2)/T9**3 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T12**2)/T9**3))*T13)/T9 + (2*P2*T12*T7)/T9**2 - (2*P2*T13*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + ((cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13*T8)/T9**2 + (4*P2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13*T12)/T9**3) - B1*((T7*T8)/T9 - P1*((6*T13*T7)/T9**2 - (2*T12*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + (6*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13**2)/T9**3 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T12**2)/T9**3) + ((K1*((2*T13*T7)/T9**2 - (2*T12*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13**2)/T9**3 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T12**2)/T9**3) + 2*K2*T6*((2*T13*T7)/T9**2 - (2*T12*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13**2)/T9**3 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T12**2)/T9**3) + 3*K3*T6**2*((2*T13*T7)/T9**2 - (2*T12*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13**2)/T9**3 + (2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T12**2)/T9**3))*T13)/T9 + (2*P2*T12*T7)/T9**2 - (2*P2*T13*(sk*sp*(X - X0) + cp*cw*sk*(Z - Z0) - cp*sk*sw*(Y - Y0)))/T9**2 + ((cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13*T8)/T9**2 + (4*P2*(cp*(X - X0) - cw*sp*(Z - Z0) + sp*sw*(Y - Y0))*T13*T12)/T9**3)
        self.df2_dp = f*t8

        self.df1_dk = B1*((T12*T8)/T9 - (2*P2*T13**2)/T9**2 + (2*P2*T12**2)/T9**2 - (4*P1*T13*T12)/T9**2) + f*((T12*T8)/T9 - (2*P2*T13**2)/T9**2 + (2*P2*T12**2)/T9**2 - (4*P1*T13*T12)/T9**2) + B2*t9
        self.df2_dk = f*t9

        ########################################################################
        self.df1_df = (T13*T8)/T9 - P1*((3*T13**2)/T9**2 + T12**2/T9**2) + (2*P2*T13*T12)/T9**2
        self.df2_df = (2*P1*T13*T12)/T9**2 - (T12*T8)/T9 - P2*(T13**2/T9**2 + (3*T12**2)/T9**2)

        self.df1_dcx = -1
        self.df2_dcx = 0

        self.df1_dcy = 0
        self.df2_dcy = -1

        self.df1_db1 = (T13*T8)/T9 - P1*((3*T13**2)/T9**2 + T12**2/T9**2) + (2*P2*T13*T12)/T9**2
        self.df2_db1 = 0

        self.df1_db2 = (2*P1*T13*T12)/T9**2 - (T12*T8)/T9 - P2*(T13**2/T9**2 + (3*T12**2)/T9**2)
        self.df2_db2 = 0

        self.df1_dk1 = (B1*T6*T13)/T9 - (B2*T6*T12)/T9 + (f*T6*T13)/T9
        self.df2_dk1 = -(f*T6*T12)/T9

        self.df1_dk2 = (B1*T6**2*T13)/T9 - (B2*T6**2*T12)/T9 + (f*T6**2*T13)/T9
        self.df2_dk2 = -(f*T6**2*T12)/T9

        self.df1_dk3 = (B1*T6**3*T13)/T9 - (B2*T6**3*T12)/T9 + (f*T6**3*T13)/T9
        self.df2_dk3 = -(f*T6**3*T12)/T9

        self.df1_dp1 = (2*B2*T13*T12)/T9**2 - f*((3*T13**2)/T9**2 + T12**2/T9**2) - B1*((3*T13**2)/T9**2 + T12**2/T9**2)
        self.df2_dp1 = (2*f*T13*T12)/T9**2

        self.df1_dp2 = (2*f*T13*T12)/T9**2 - B2*(T13**2/T9**2 + (3*T12**2)/T9**2) + (2*B1*T13*T12)/T9**2
        self.df2_dp2 = -f*(T13**2/T9**2 + (3*T12**2)/T9**2)


        ##################################################################
        self.df1_dX = B2*t3 - f*t6 - B1*t6
        self.df2_dX = f*t3

        self.df1_dY = B1*t5 - B2*t2 + f*t5
        self.df2_dY = -f*t2

        self.df1_dZ = B1*t4 - B2*t1 + f*t4
        self.df2_dZ = -f*t1

def rotation(Omega, Phi, Kapa):
    R = np.asarray([[np.cos(Kapa) * np.cos(Phi), np.cos(Omega) * np.sin(Kapa) + np.cos(Kapa) * np.sin(Omega) * np.sin(Phi), np.sin(Kapa) * np.sin(Omega) - np.cos(Kapa) * np.cos(Omega) * np.sin(Phi)],
                    [-np.cos(Phi) * np.sin(Kapa), np.cos(Kapa) * np.cos(Omega) - np.sin(Kapa) * np.sin(Omega) * np.sin(Phi), np.cos(Kapa) * np.sin(Omega) + np.cos(Omega) * np.sin(Kapa) * np.sin(Phi)],
                    [np.sin(Phi), -np.cos(Phi) * np.sin(Omega), np.cos(Omega) * np.cos(Phi)]])
    return R

class Bundle_information():
    def __init__(self,dir_files, gcp, TieXYZ, Observation_Tie, Observation_gcp, IOP, EOP):
        # 5 -------------------- Mange Observation and Target Table --------------------
        Observation = np.row_stack((Observation_Tie, Observation_gcp))
        points_3d = np.row_stack((TieXYZ[:, 1:], gcp[:, 1:]))

        # 6 -------------------- Approximate Value --------------------
        x0 = np.hstack((EOP.ravel(), IOP.ravel(), points_3d.ravel()))

        self.dir_files = dir_files
        self.Observation = Observation
        self.Observation_gcp = Observation_gcp
        self.Gcp_indices = gcp[:, 0].astype(int)
        self.Tie_indices = Observation[:, 1].astype(int)
        self.camera_indices = Observation[:, 0].astype(int)
        self.points_3d = np.row_stack((TieXYZ[:, 1:], gcp[:, 1:]))
        self.points_2d = Observation[:, 2:]
        self.n_cameras = len(EOP)
        self.n_tiepoints = len(points_3d)
        self.n_gcpoints = len(gcp)
        self.GCP = gcp
        self.x0 = x0

class SBA_point_const():
    def __init__(self, Weight, Info_BA, Max_iter, th, Show):
        print(' -------------------------------------- Sparse Bundle Adjustment --------------------------------------')
        print('  BA_loop      Gcp    Max D_residual   Max D_Correction    Sigma0_h2     Loss_img(pix)    Loss_Obj(m)  ')
        print(' ------------------------------------------------------------------------------------------------------')
        # 8 -------------------- Algorithm iterations --------------------
        DL, DX, Phi, DL_img, DL_obj, UV = [], [], [], [], [], []
        DL.append(0)
        DX.append(0)
        nTie = len(Info_BA.Tie_indices)
        n_tiepoints = Info_BA.n_tiepoints
        n_cameras = Info_BA.n_cameras
        m = len(Info_BA.Tie_indices) * 2 + len(Info_BA.Gcp_indices)
        n = 10 + n_cameras * 6 + n_tiepoints * 3
        prt = 1
        x0 = Info_BA.x0
        for l in range(Max_iter):
            # 9 -------------------- Design Matrix and residuals --------------------
            E, I, P = Jac_IOP_EOP_OC_GCP_PointConst(x0, Info_BA)
            dl, reproject = ObjFun_IOP_EOP_OC_GCP_PointConst(x0, Info_BA)

            # 9 -------------------- Normal Matrix -------------------
            U = bmat([[E.T @ Weight @ E, E.T @ Weight @ I], [I.T @ Weight @ E, I.T @ Weight @ I]]).tocsr()
            W = vstack((E.T @ Weight @ P, I.T @ Weight @ P)).tocsr()
            Wt = hstack((P.T @ Weight @ E, P.T @ Weight @ I)).tocsr()
            V = P.T @ Weight @ P
            iV = inverse_block_diag_sparse(V)

            rei = np.hstack((-(E.T @ Weight @ dl), -(I.T @ Weight @ dl)))
            rp = -(P.T @ Weight @ dl)

            # 10 -------------------- RNE --------------------
            AA = U - (W @ iV @ Wt)
            ll = rei - (W @ iV @ rp)
            dei = spsolve(AA, ll)
            dp = iV @ (rp - Wt @ dei)
            dx = np.hstack((dei, dp))
            x0 += dx

            ## ---------------------------- updated values ----------------------------
            Final_EOP = x0[:n_cameras * 6].reshape((n_cameras, 6))
            Final_IOP = x0[n_cameras * 6: n_cameras * 6 + 10]
            Final_Tie = np.column_stack((np.arange(n_tiepoints), x0[n_cameras * 6 + 10:].reshape((n_tiepoints, 3))))
            Final_Gcp = Final_Tie[Info_BA.Gcp_indices].reshape((len(Info_BA.Gcp_indices), 4))

            # 12 -------------------- Save image and object residuals --------------------
            dl_img = (dl[:nTie * 2]).reshape(len(Info_BA.Tie_indices), 2)
            dl_obj = (dl[nTie * 2:]).reshape(len(Info_BA.Gcp_indices), 3)
            phi = (dl @ Weight @ dl.T) / (m-n)

            rmse_img = np.sqrt(dl_img[:, 0]**2 + dl_img[:, 1]**2)
            rmse_img_total = np.sqrt(np.mean(rmse_img**2))
            rmse_obj_total = np.sqrt(np.mean(dl_obj**2))

            DL.append(dl)
            DX.append(dx)
            Phi.append(phi)
            UV.append(reproject)
            DL_img.append(np.sqrt(np.mean(dl_img ** 2, axis=0)))
            DL_obj.append(np.sqrt(np.mean(dl_obj ** 2, axis=0)))

            # 13 -------------------- Print image and object residuals --------------------
            if l >= 4:
                prt = 5
            prt_dl = np.abs(np.max(np.abs(DL[-2])) - np.max(np.abs(DL[-1])))
            if l == 0 or (l + 1) % prt == 0 or np.max(np.abs(DX[-2] - DX[-1])) < th or np.max(np.abs(DL[-2] - DL[-1])) < th:
                str1 = '   %4g     %6g       %2.2e         %2.2e' % (l + 1, len(Info_BA.GCP), prt_dl, np.max(np.abs(DX[-2] - DX[-1])))
                str2 = '         %2.3e       %2.3e        %4.4f' % (phi, rmse_img_total, rmse_obj_total)
                print(str1 + str2)

            if np.max(np.abs(DX[-2] - DX[-1])) < th or prt_dl < th or l == Max_iter - 1:
                break

        i1, _ = ismember(Info_BA.Observation[:, 1].astype(int), Info_BA.Gcp_indices)
        res_gcp_img = dl_img[i1]
        res_gcp_img_total = np.sqrt(res_gcp_img[:, 0] ** 2 + res_gcp_img[:, 1] ** 2)
        obs_gcp = Info_BA.Observation[i1, :]

        res_gcp_obj = Final_Gcp[:, 1:] - Info_BA.GCP[:, 1:]
        res_gcp_obj_total_horz = np.sqrt(res_gcp_obj[:, 0] ** 2 + res_gcp_obj[:, 1] ** 2)
        res_gcp_obj_total_vert = res_gcp_obj[:, 2]

        # 12 -------------------- Blunder Detection Object Space --------------------
        df = pd.DataFrame(res_gcp_obj_total_horz)
        idx_up_H = df[0].quantile(0.95)
        df = pd.DataFrame(res_gcp_obj_total_vert)
        idx_up_V = df[0].quantile(0.95)

        bl1 = np.where(res_gcp_obj_total_horz >= idx_up_H)
        bl2 = np.where(res_gcp_obj_total_vert >= idx_up_V)

        # 12 -------------------- Blunder Detection image Space -------------------
        df = pd.DataFrame(res_gcp_img_total)
        idx_up = df[0].quantile(0.95)

        bl3 = np.where(res_gcp_img_total >= idx_up)
        BL = np.unique(np.hstack((Info_BA.Gcp_indices[bl1[0]], Info_BA.Gcp_indices[bl2[0]], obs_gcp[bl3, 1][0])))


        if Show == True:
            PLOT_iter(np.asarray(DL_img), np.asarray(Phi), np.asarray(DL_obj))
            Plot_residual_hist(DL[1])
            Plot_residual_hist(DL[-1])
            show_residual(Info_BA.dir_files, Info_BA.Observation, UV[0], 3)
            show_residual(Info_BA.dir_files, Info_BA.Observation, UV[-1], 3)

        self.x_hat = x0
        self.Coordinate_point = Final_Tie
        self.Final_GCP = Final_Gcp
        self.Final_EOP = Final_EOP
        self.Final_IOP = Final_IOP
        self.Residuals = DL
        self.Corrections = DX
        self.Phi = np.asarray(Phi)
        self.blunder_id = BL
        print(' ')

def Jac_IOP_EOP_OC_GCP_PointConst(params, Info_BA):
    n_tiepoints = Info_BA.n_tiepoints
    n_cameras = Info_BA.n_cameras
    camera_indices = Info_BA.camera_indices
    Tie_indices = Info_BA.Tie_indices
    Gcp_indices = Info_BA.Gcp_indices

    m = len(Tie_indices) * 2 + len(Gcp_indices) * 3
    n = n_cameras * 6
    E = lil_matrix((m, n), dtype=float)

    m = len(Tie_indices) * 2 + len(Gcp_indices) * 3
    n = 10
    I = lil_matrix((m, n), dtype=float)

    m = len(Tie_indices) * 2 + len(Gcp_indices) * 3
    n = n_tiepoints * 3
    P = lil_matrix((m, n), dtype=float)

    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    IOP = params[n_cameras * 6: n_cameras * 6 + 10]
    points_3d = params[n_cameras * 6 + 10:].reshape((n_tiepoints, 3))
    J = jac_element(IOP, camera_params[camera_indices], points_3d[Tie_indices])

    i = np.arange(len(Tie_indices))
    E[2 * i, camera_indices * 6] = J.df1_dX0
    E[2 * i + 1, camera_indices * 6] = J.df2_dX0

    E[2 * i, camera_indices * 6 + 1] = J.df1_dY0
    E[2 * i + 1, camera_indices * 6 + 1] = J.df2_dY0

    E[2 * i, camera_indices * 6 + 2] = J.df1_dZ0
    E[2 * i + 1, camera_indices * 6 + 2] = J.df2_dZ0

    E[2 * i, camera_indices * 6 + 3] = J.df1_dw
    E[2 * i + 1, camera_indices * 6 + 3] = J.df2_dw

    E[2 * i, camera_indices * 6 + 4] = J.df1_dp
    E[2 * i + 1, camera_indices * 6 + 4] = J.df2_dp

    E[2 * i, camera_indices * 6 + 5] = J.df1_dk
    E[2 * i + 1, camera_indices * 6 + 5] = J.df2_dk

    ########################################################################
    I[2 * i, 0] = J.df1_df
    I[2 * i + 1, 0] = J.df2_df

    I[2 * i, 1] = -1
    I[2 * i + 1, 1] = 0

    I[2 * i, 2] = 0
    I[2 * i + 1, 2] = -1

    I[2 * i, 3] = J.df1_db1
    I[2 * i + 1, 3] = 0

    I[2 * i, 4] = J.df1_db2
    I[2 * i + 1, 4] = 0

    I[2 * i, 5] = J.df1_dk1
    I[2 * i + 1, 5] = J.df2_dk1

    I[2 * i, 6] = J.df1_dk2
    I[2 * i + 1, 6] = J.df2_dk2

    I[2 * i, 7] = J.df1_dk3
    I[2 * i + 1, 7] = J.df2_dk3

    I[2 * i, 8] = J.df1_dp1
    I[2 * i + 1, 8] = J.df2_dp1

    I[2 * i, 9] = J.df1_dp2
    I[2 * i + 1, 9] = J.df2_dp2

    ##################################################################
    P[2 * i, Tie_indices * 3] = J.df1_dX
    P[2 * i + 1, Tie_indices * 3] = J.df2_dX

    P[2 * i, Tie_indices * 3 + 1] = J.df1_dY
    P[2 * i + 1, Tie_indices * 3 + 1] = J.df2_dY

    P[2 * i, Tie_indices * 3 + 2] = J.df1_dZ
    P[2 * i + 1, Tie_indices * 3 + 2] = J.df2_dZ

    i = np.arange(len(Tie_indices) * 2, len(Tie_indices) * 2 + len(Gcp_indices) * 3, 3)
    P[i, Gcp_indices * 3 + 0] = 1
    P[i+1, Gcp_indices * 3 + 1] = 1
    P[i+2, Gcp_indices * 3 + 2] = 1
    return E, I, P

def ObjFun_IOP_EOP_OC_GCP_PointConst(params, Info_BA):
    W = 5472
    H = 3648

    n_tiepoints = Info_BA.n_tiepoints
    n_cameras = Info_BA.n_cameras
    camera_indices = Info_BA.camera_indices
    Tie_indices = Info_BA.Tie_indices
    Gcp_indices = Info_BA.Gcp_indices
    points_2d = Info_BA.points_2d
    gcp = Info_BA.GCP

    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    IOP = params[n_cameras * 6: n_cameras * 6 + 10]
    points_3d = params[n_cameras * 6 + 10:].reshape((n_tiepoints, 3))
    conpoints_3d = points_3d[Gcp_indices]

    XYZ = points_3d[Tie_indices]
    eop = camera_params[camera_indices]

    XYZ_camera = []
    for i, val in enumerate(eop):
        XYZ_cam = rotation(val[3], val[4], val[5]) @ (XYZ[i, :] - val[:3])
        XYZ_camera.append(XYZ_cam)

    XYZ_camera = np.asarray(XYZ_camera)
    f = IOP[0]
    cx = IOP[1]
    cy = IOP[2]
    B1 = IOP[3]
    B2 = IOP[4]
    K1 = IOP[5]
    K2 = IOP[6]
    K3 = IOP[7]
    P1 = IOP[8]
    P2 = IOP[9]

    xp = -(XYZ_camera[:, 0] / XYZ_camera[:, 2])
    yp = (XYZ_camera[:, 1] / XYZ_camera[:, 2])

    r = np.sqrt(xp ** 2 + yp ** 2)
    dx = xp * (1 + K1 * (r ** 2) + K2 * (r ** 4) + K3 * (r ** 6)) + (P1 * ((r ** 2) + 2 * (xp ** 2))) + (2 * P2 * (xp * yp))
    dy = yp * (1 + K1 * (r ** 2) + K2 * (r ** 4) + K3 * (r ** 6)) + (P2 * ((r ** 2) + 2 * (yp ** 2))) + (2 * P1 * (xp * yp))

    u = W * 0.5 + cx + dx * f + dx * B1 + dy * B2
    v = H * 0.5 + cy + dy * f

    f1 = points_2d[:, 0] - u
    f2 = points_2d[:, 1] - v

    res = np.column_stack((f1, f2)).ravel()
    resgcp = (conpoints_3d - gcp[:, 1:]).ravel()

    return np.hstack((res, resgcp)), np.column_stack((u, v))
