from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import map
from builtins import str
from builtins import range
from past.utils import old_div
import febio
import pickle
import subprocess
import os
import tkinter.filedialog
import platform
import fnmatch
import itertools
import time
import datetime
import vtk
from vtk.util import numpy_support
from tkinter import *
from tkinter.ttk import Notebook
from collections import OrderedDict
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pyplot import cm

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scipy.stats import histogram
from weighted import quantile_1D


class Application(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.lastdir = os.getcwd()
        self.op_sys = platform.platform()

        #Check for path to FEBio executable
        try:
            if "windows" in self.op_sys.lower():
                home = os.getenv("HOMEPATH")
            else:
                home = os.getenv("HOME")
            fid = open(os.path.join(home, ".pyCellAnalystFEA.pth"), "r")
            self._FEBIO_BIN = os.path.abspath(fid.readline())
            fid.close()
            try:
                p = subprocess.Popen(self._FEBIO_BIN)
                p.terminate()
            except:
                self.getFEBioLocation("File indicated as the FEBio executable is incorrect. Please reselect.", home)
        except:
            self.getFEBioLocation("Please select the FEBio executable file.", home)

        self.notebook = Notebook(self)
        self.tab1 = Frame(self.notebook)
        self.tab2 = Frame(self.notebook)
        self.notebook.add(self.tab1, text="Analysis")
        self.notebook.add(self.tab2, text="Material Model")
        self.notebook.grid(row=0, column=0, sticky=NW)

        self.pickles = []
        self.data = {}
        self.vtkMeshes = {}

        self.volumes = {}
        self.outputs = OrderedDict(
            [("Effective Strain (von Mises)", IntVar(value=1)),
             ("Maximum Compressive Strain", IntVar(value=1)),
             ("Maximum Tensile Strain", IntVar(value=1)),
             ("Maximum Shear Strain", IntVar(value=1)),
             ("Volumetric Strain", IntVar(value=1)),
             ("Effective Stress (von Mises)", IntVar(value=0)),
             ("Maximum Compressive Stress", IntVar(value=0)),
             ("Maximum Tensile Stress", IntVar(value=0)),
             ("Maximum Shear Stress", IntVar(value=0)),
             ("Pressure", IntVar(value=0))])

        self.analysis = OrderedDict(
            [("Generate Histograms", IntVar(value=1)),
             ("Tukey Boxplots", IntVar(value=1)),
             ("Calculate Differences", IntVar(value=1)),
             ("Convert to VTK", IntVar(value=1))])

        self.groundSubstance = IntVar(value=1)
        self.groundSubstances = ('neoHookean',
                                 'Mooney-Rivlin',
                                 'Transversely Isotropic')

        self.tensileNetwork = IntVar(value=1)
        self.tensileNetworks = ('None',
                                'Continuous Fibre Distribution')
        self.compressiveNetwork = IntVar(value=1)

        self.groundParameters = [
            OrderedDict([("Young's Modulus", DoubleVar(value=1.0e-9)),
                         ("Poisson's Ratio", DoubleVar(value=0.1))]),
            OrderedDict([("C_1", DoubleVar(value=2.3e-10)),
                         ("C_2", DoubleVar(value=0.0)),
                         ("Bulk Modulus", DoubleVar(value=4.17e-10))]),
            OrderedDict([
                ("Young's Modulus (radial)", DoubleVar(value=1.45e-6)),
                ("Young's Modulus (tangential)", DoubleVar(value=1.0e-9)),
                ("Poisson's Ratio (radial:tangential)", DoubleVar(value=0.4)),
                ("Poisson's Ratio (tangential:radial)", DoubleVar(value=0.0)),
                ("Shear Modulus (radial planes)", DoubleVar(value=5.0e-10))])]

        self.tensileParameters = [
            None,
            OrderedDict([("ksi1", DoubleVar(value=1.0e-7)),
                         ("ksi2=ksi3", DoubleVar(value=1.45e-6)),
                         ("beta1", DoubleVar(value=2.0)),
                         ("beta2=beta3", DoubleVar(value=2.0)),
                         ("Distance Cutoff [0, 1]", DoubleVar(value=0.5)),
                         ("Cutoff Grade (0, 1]", DoubleVar(value=0.1))])]

        self.results = {"Effective Strain (von Mises)": {},
                        "Maximum Compressive Strain": {},
                        "Maximum Tensile Strain": {},
                        "Maximum Shear Strain": {},
                        "Volumetric Strain": {},
                        "Effective Stress (von Mises)": {},
                        "Maximum Compressive Stress": {},
                        "Maximum Tensile Stress": {},
                        "Maximum Shear Stress": {},
                        "Pressure": {},
                        "Displacement": {}}

        self.histograms = {"Effective Strain (von Mises)": {},
                           "Maximum Compressive Strain": {},
                           "Maximum Tensile Strain": {},
                           "Maximum Shear Strain": {},
                           "Volumetric Strain": {},
                           "Effective Stress (von Mises)": {},
                           "Maximum Compressive Stress": {},
                           "Maximum Tensile Stress": {},
                           "Maximum Shear Stress": {},
                           "Pressure": {}}

        self.boxwhiskers = {"Effective Strain (von Mises)": {},
                            "Maximum Compressive Strain": {},
                            "Maximum Tensile Strain": {},
                            "Maximum Shear Strain": {},
                            "Volumetric Strain": {},
                            "Effective Stress (von Mises)": {},
                            "Maximum Compressive Stress": {},
                            "Maximum Tensile Stress": {},
                            "Maximum Shear Stress": {},
                            "Pressure": {}}

        self.differences = {"Effective Strain (von Mises)": {},
                            "Maximum Compressive Strain": {},
                            "Maximum Tensile Strain": {},
                            "Maximum Shear Strain": {},
                            "Volumetric Strain": {},
                            "Effective Stress (von Mises)": {},
                            "Maximum Compressive Stress": {},
                            "Maximum Tensile Stress": {},
                            "Maximum Shear Stress": {},
                            "Pressure": {}}

        self.grid()
        self.createWidgetsTab1()
        self.createWidgetsTab2()

    def getFEBioLocation(self, msg, home):
        filename = tkinter.filedialog.askopenfilename(
            parent=root, initialdir=home,
            title=msg)
        if filename:
            try:
                p = subprocess.Popen(filename)
                p.terminate()
                self._FEBIO_BIN = filename
                fid = open(os.path.join(home, ".pyCellAnalystFEA.pth"), "wt")
                fid.write(self._FEBIO_BIN)
                fid.close()
            except:
                self.getFEBioLocation("Incorrect FEBio executable file was selected. Please reselect.", home)
        else:
            raise SystemExit("You must indicate the location of the FEBio executable. Exiting...")

    def createWidgetsTab1(self):
        #pickles to load
        self.pickleFrame = LabelFrame(
            self.tab1, text="Pickles from which to generate models")
        self.pickleFrame.grid(row=0, column=0,
                              columnspan=3, rowspan=2,
                              padx=5, pady=5, sticky=NW)
        self.buttonAddFile = Button(
            self.pickleFrame, text="Add", bg='green', command=self.addPickle)
        self.buttonAddFile.grid(row=0, column=0, padx=5, pady=5, sticky=E + W)
        self.buttonRemoveFile = Button(
            self.pickleFrame,
            text="Remove", bg='red', command=self.removePickle)
        self.buttonRemoveFile.grid(row=1, column=0,
                                   padx=5, pady=5, sticky=E + W)
        self.listPickles = Listbox(self.pickleFrame,
                                   width=80, selectmode=MULTIPLE)
        self.listPickles.grid(row=0, column=1,
                              columnspan=2, rowspan=2,
                              padx=5, pady=5, sticky=E + W)

        #outputs
        self.outputsFrame = LabelFrame(self.tab1, text="Outputs")
        self.outputsFrame.grid(row=2, column=0,
                               rowspan=4, columnspan=2,
                               padx=5, pady=5, sticky=NW)
        for i, t in enumerate(self.outputs.keys()):
            if i < 5:
                Checkbutton(self.outputsFrame, text=t,
                            variable=self.outputs[t]).grid(row=i, column=0,
                                                           padx=5, pady=5,
                                                           sticky=NW)
            else:
                Checkbutton(self.outputsFrame, text=t,
                            variable=self.outputs[t]).grid(row=i - 5, column=1,
                                                           padx=5, pady=5,
                                                           sticky=NW)

        #analysis settings
        self.analysisSettingsFrame = LabelFrame(self.tab1,
                                                text="Analysis Options")
        self.analysisSettingsFrame.grid(row=2, column=2,
                                        padx=5, pady=5, sticky=E + W)
        for i, t in enumerate(self.analysis.keys()):
            Checkbutton(self.analysisSettingsFrame, text=t,
                        variable=self.analysis[t]).grid(row=i, column=2,
                                                        padx=5, pady=5,
                                                        sticky=NW)

        #run analysis
        self.buttonExecute = Button(self.tab1, bg='green',
                                    font=('Helvetica', '20', 'bold'),
                                    text='Perform Simulation',
                                    command=self.performFEA)
        self.buttonExecute.grid(row=7, column=0, columnspan=3,
                                padx=5, pady=5, sticky=E + W)

        self.buttonAnalyze = Button(self.tab1, bg='blue',
                                    font=('Helvetica', '20', 'bold'),
                                    text='Analyze Results',
                                    command=self.analyzeResults)
        self.buttonAnalyze.grid(row=8, column=0, columnspan=3,
                                padx=5, pady=5, sticky=E + W)

    def createWidgetsTab2(self):
        self.groundSubstanceFrame = LabelFrame(self.tab2,
                                               text="Ground Substance")
        self.groundSubstanceFrame.grid(
            row=0, column=0, padx=5, pady=5, sticky=NW)

        self.tensileFibreNetworkFrame = LabelFrame(
            self.tab2, text="Tensile Fibres")
        self.tensileFibreNetworkFrame.grid(
            row=0, column=1, padx=5, pady=5, sticky=NW)

        self.compressiveFibreNetworkFrame = LabelFrame(
            self.tab2, text="Compressive Fibres")
        self.compressiveFibreNetworkFrame.grid(
            row=0, column=3, padx=5, pady=5, sticky=NW)

        # Ground substance
        for i, t in enumerate(self.groundSubstances):
            Radiobutton(self.groundSubstanceFrame,
                        text=t,
                        indicatoron=0,
                        padx=5,
                        width=20,
                        command=self.populateGroundSubstance,
                        variable=self.groundSubstance,
                        value=i + 1).pack(anchor=NW)

        self.groundSubstanceSettingsFrame = LabelFrame(
            self.tab2, text="Ground Substance Parameters")
        self.groundSubstanceSettingsFrame.grid(
            row=1, column=0, padx=5, pady=5, sticky=NW)
        for i, (t, v) in enumerate(
                self.groundParameters[self.groundSubstance.get() - 1].items()):
            Label(self.groundSubstanceSettingsFrame, text=t).grid(row=i,
                                                                  column=0,
                                                                  sticky=NW)
            Entry(self.groundSubstanceSettingsFrame,
                  textvariable=v,
                  width=10).grid(row=i, column=1, sticky=NW)

        # Tensile Network
        for i, t in enumerate(self.tensileNetworks):
            Radiobutton(self.tensileFibreNetworkFrame,
                        text=t,
                        indicatoron=0,
                        padx=5,
                        width=25,
                        command=self.populateTensileNetwork,
                        variable=self.tensileNetwork,
                        value=i + 1).pack(anchor=NW)

        self.tensileNetworkSettingsFrame = LabelFrame(
            self.tab2, text="Tensile Network Parameters")
        self.tensileNetworkSettingsFrame.grid(
            row=1, column=1, padx=5, pady=5, sticky=NW)
        if self.tensileNetwork.get() == 1:
            Label(self.tensileNetworkSettingsFrame,
                  text="No tensile network will be modeled").grid(row=0,
                                                                  column=0,
                                                                  sticky=NW)
        else:
            for i, (t, v) in enumerate(
                    self.tensileParameters[self.tensileNetwork.get() - 1].items()):
                Label(self.tensileNetworkSettingsFrame, text=t).grid(row=i,
                                                                     column=0,
                                                                     sticky=NW)

                Entry(self.tensileNetworkSettingsFrame, textvariable=v,
                      width=10).grid(row=i, column=1, sticky=NW)

    def populateGroundSubstance(self):
        for child in self.groundSubstanceSettingsFrame.grid_slaves():
            child.destroy()
        for i, (t, v) in enumerate(
                self.groundParameters[self.groundSubstance.get() - 1].items()):
            Label(self.groundSubstanceSettingsFrame, text=t).grid(row=i,
                                                                  column=0,
                                                                  sticky=NW)
            Entry(self.groundSubstanceSettingsFrame,
                  textvariable=v,
                  width=10).grid(row=i, column=1, sticky=NW)

    def populateTensileNetwork(self):
        for child in self.tensileNetworkSettingsFrame.grid_slaves():
            child.destroy()
        if self.tensileNetwork.get() == 1:
            Label(self.tensileNetworkSettingsFrame,
                  text="No tensile network will be modeled").grid(row=0,
                                                                  column=0,
                                                                  sticky=NW)
        else:
            for i, (t, v) in enumerate(
                    self.tensileParameters[self.tensileNetwork.get() - 1].items()):
                Label(self.tensileNetworkSettingsFrame, text=t).grid(row=i,
                                                                     column=0,
                                                                     sticky=NW)
                Entry(
                    self.tensileNetworkSettingsFrame, textvariable=v,
                    width=10).grid(row=i, column=1, sticky=NW)
            Button(
                self.tensileNetworkSettingsFrame, text='Plot', width=10,
                command=self.plotWeight).grid(row=i + 1, column=0, sticky=NW)

    def plotWeight(self):
        f = Figure()
        a = f.add_subplot(111)
        x = np.arange(0.0, 1.0, 0.01)
        d = self.tensileParameters[1][
            "Distance Cutoff [0, 1]"].get()
        k = self.tensileParameters[1][
            "Cutoff Grade (0, 1]"].get()
        if k <= 0:
            k = 0.01
            print(("Cutoff Grade must be greater than zero"
                   " Using a value of 0.01."))
        s = 0.5 * (1 - np.tanh(old_div((x - d), k)))
        a.plot(x, s)
        a.set_xlabel("Normalized distance from object surface")
        a.set_ylabel("Tensile Network Weight")
        canvas = FigureCanvasTkAgg(f,
                                   master=self.tensileNetworkSettingsFrame)
        canvas.show()
        (cols, rows) = self.tensileNetworkSettingsFrame.grid_size()
        slaves = self.tensileNetworkSettingsFrame.grid_slaves()
        if rows == 7:
            canvas.get_tk_widget().grid(row=rows + 1, column=0,
                                        columnspan=2, sticky=NW)
        else:
            slaves[0].destroy()
            canvas.get_tk_widget().grid(rows=rows, column=0,
                                        columnspan=2, sticky=NW)

    def addPickle(self):
        filenames = tkinter.filedialog.askopenfilenames(
            parent=root, initialdir=self.lastdir,
            title="Please choose pickle file(s) for analysis.")
        filenames = root.tk.splitlist(filenames)
        for f in filenames:
            if '.pkl' in f:
                self.listPickles.insert(END, f)
                self.pickles.append(f)
            else:
                print(("WARNING - {:s} does not contain the .pkl extension."
                       " Ignoring this file."))

    def removePickle(self):
        index = self.listPickles.curselection()
        if index:
            for i in index[::-1]:
                self.listPickles.delete(i)
                del self.pickles[i]

    def performFEA(self):
        for filename in self.pickles:
            fid = open(filename, 'rb')
            self.data[filename] = pickle.load(fid)
            fid.close()

            mesh = febio.MeshDef()
            # would be good to vectorize these
            for i, e in enumerate(self.data[filename]['elements']):
                mesh.elements.append(['tet4', i + 1] + e)

            for i, n in enumerate(self.data[filename]['nodes']):
                mesh.nodes.append([i + 1] + n)

            mesh.addElementSet(setname='cell',
                               eids=list(range(1, len(mesh.elements) + 1)))

            modelname = filename.replace('.pkl', '.feb')
            model = febio.Model(modelfile=modelname,
                                steps=[{'Displace': 'solid'}])

            #lookup table for proper FEBio keyword vs GUI text
            keywords = {
                "neoHookean": "neo-Hookean",
                "Young's Modulus": "E",
                "Poisson's Ratio": "v",
                "Mooney-Rivlin": "Mooney-Rivlin",
                "C_1": "c1",
                "C_2": "c2",
                "Bulk Modulus": "k",
                "Transversely Isotropic": "orthotropic elastic",
                "Young's Modulus (radial)": "E1",
                "Young's Modulus (tangential)": ["E2", "E3"],
                "Poisson's Ratio (radial:tangential)": "v12",
                "Poisson's Ratio (tangential:radial)": ["v23", "v31"],
                "Shear Modulus (radial planes)": ["G12", "G23", "G31"],
                "Continuous Fibre Distribution":
                "ellipsoidal fiber distribution"}

            #Ground Substance
            gtype = keywords[
                self.groundSubstances[self.groundSubstance.get() - 1]]
            gattributes = {}
            for (k, v) in list(self.groundParameters[
                    self.groundSubstance.get() - 1].items()):
                # transversely isotropic case
                if isinstance(keywords[k], list):
                    for a in keywords[k]:
                        # G12 = E2 / (2 * (1 + v12))
                        if a == "G12":
                            gattributes[a] = str(
                                old_div(float(gattributes["E2"]),
                                (2.0 * (1 + float(gattributes["v12"])))))
                        else:
                            gattributes[a] = str(v.get())
                # any other case
                else:
                    gattributes[keywords[k]] = str(v.get())
            if self.tensileNetwork.get() == 1:
                mat = febio.MatDef(matid=1, mname='cell',
                                   mtype=gtype, elsets='cell',
                                   attributes=gattributes)
                model.addMaterial(mat)
                model.addGeometry(mesh=mesh, mats=[mat])
            #With a tensile network
            else:
                self.makeVTK(filename)
                self.findLocalCsys(filename)
                ind = self.tensileNetwork.get() - 1
                ttype = keywords[self.tensileNetworks[ind]]
                ksi = [self.tensileParameters[ind][k].get()
                       for k in ("ksi1", "ksi2=ksi3", "ksi2=ksi3")]
                beta = [self.tensileParameters[ind][k].get()
                        for k in ("beta1", "beta2=beta3", "beta2=beta3")]

                mats = []
                I1n2 = np.eye(3)[0:2, :]

                distances = numpy_support.vtk_to_numpy(
                    self.vtkMeshes[filename].GetCellData().GetArray(
                        "Signed Distances"))
                maxd = np.max(np.abs(distances))
                for i, e in enumerate(self.data[filename]['elements']):
                    normal = self.vtkMeshes[filename].GetCellData().GetArray(
                        "LevelSet Normals").GetTuple3(i)
                    # cross with both e1 and e2 to ensure linearly independent
                    crossed = np.cross(normal, I1n2)
                    norms = np.linalg.norm(crossed, axis=1)
                    # get the index of the maximum norm; so never zero
                    ind2 = np.argmax(norms)
                    b = old_div(crossed[ind, :], norms[ind2])
                    normal = list(map(str, list(normal)))
                    b = list(map(str, list(b)))
                    d = self.tensileParameters[ind]["Distance Cutoff [0, 1]"].get()
                    k = self.tensileParameters[ind]["Cutoff Grade (0, 1]"].get()
                    x = old_div(np.abs(distances[i]), maxd)
                    w = 0.5 * (1 - np.tanh(old_div((x - d), k)))
                    if w < 0.05:
                        w = 0.05
                    tmp_ksi = [w * v for v in ksi]
                    tattributes = {"ksi":
                                   ",".join(map(str, tmp_ksi)),
                                   "beta": ",".join(map(str, beta))}
                    mesh.addElementSet(setname='e{:d}'.format(i + 1),
                                       eids=[i + 1])
                    mats.append(febio.MatDef(
                        matid=i + 1, mname='cell', mtype="solid mixture",
                        elsets='e{:d}'.format(i + 1), attributes={
                            'mat_axis':
                            ['vector', ','.join(normal), ','.join(b)]}))

                    mats[-1].addBlock(branch=1, btype='solid', mtype=gtype,
                                      attributes=gattributes)
                    mats[-1].addBlock(branch=1, btype='solid', mtype=ttype,
                                      attributes=tattributes)
                    model.addMaterial(mats[-1])
                model.addGeometry(mesh=mesh, mats=mats)

            ctrl = febio.Control()
            ctrl.setAttributes({'title': 'cell', 'max_ups': '0'})

            model.addControl(ctrl, step=0)

            boundary = febio.Boundary(steps=1)
            for i, bc in enumerate(self.data[filename]['boundary conditions']):
                boundary.addPrescribed(
                    step=0, nodeid=self.data[filename]['surfaces'][i],
                    dof='x', lc='1', scale=str(bc[0]))
                boundary.addPrescribed(
                    step=0, nodeid=self.data[filename]['surfaces'][i],
                    dof='y', lc='1', scale=str(bc[1]))
                boundary.addPrescribed(
                    step=0, nodeid=self.data[filename]['surfaces'][i],
                    dof='z', lc='1', scale=str(bc[2]))

            model.addBoundary(boundary=boundary)
            model.addLoadCurve(lc='1', lctype='linear', points=[0, 0, 1, 1])
            model.writeModel()

            subprocess.call(self._FEBIO_BIN + " -i " + modelname, shell=True)

    def analyzeResults(self):
        self.matched = [] # for paired comparisons
        for i, f in enumerate(self.pickles):
            s = f.split(os.sep)[-1]
            if not(fnmatch.filter(
                    itertools.chain.from_iterable(self.matched), "*" + s)):
                self.matched.append(fnmatch.filter(self.pickles, "*" + s))
            plotname = f.replace('.pkl', '.xplt')
            print("\n... Analyzing results for {:s}".format(plotname))

            results = febio.FebPlt(plotname)
            stress = np.zeros((len(self.data[f]['elements']), 3, 3), float)
            strain = np.copy(stress)
            #material element volumes
            mvolumes = np.zeros(len(self.data[f]['elements']), float)
            #spatial element volumes
            svolumes = np.copy(mvolumes)
            nnodes = len(list(results.NodeData.keys()))
            displacement = np.zeros((nnodes, 3))
            for j, n in enumerate(self.data[f]['nodes']):
                tmp = results.NodeData[j + 1]['displacement'][-1, :]
                displacement[j, :] = [tmp[0], tmp[1], tmp[2]]
            pstress = []
            pstressdir = []
            pstrain = []
            pstraindir = []
            for j, e in enumerate(self.data[f]['elements']):
                tmp = results.ElementData[j + 1]['stress'][-1, :]
                stress[j, :, :] = [[tmp[0], tmp[3], tmp[5]],
                                   [tmp[3], tmp[1], tmp[4]],
                                   [tmp[5], tmp[4], tmp[2]]]
                #material coordinates
                X = np.zeros((4, 3), float)
                #spatial coordinates
                x = np.zeros((4, 3), float)
                for k in range(4):
                    X[k, :] = self.data[f]['nodes'][e[k] - 1]
                    x[k, :] = (X[k, :] +
                               results.NodeData[e[k]]['displacement'][-1, :])
                #set up tangent space
                W = np.zeros((6, 3), float)
                w = np.zeros((6, 3), float)
                for k, c in enumerate(
                        [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (1, 2)]):
                    W[k, :] = X[c[1], :] - X[c[0], :]
                    w[k, :] = x[c[1], :] - x[c[0], :]
                dX = np.zeros((6, 6), float)
                ds = np.zeros((6, 1), float)
                for k in range(6):
                    for l in range(3):
                        dX[k, l] = 2 * W[k, l] ** 2
                    dX[k, 3] = 4 * W[k, 0] * W[k, 1]
                    dX[k, 4] = 4 * W[k, 1] * W[k, 2]
                    dX[k, 5] = 4 * W[k, 0] * W[k, 2]
                    ds[k, 0] = (np.linalg.norm(w[k, :]) ** 2 -
                                np.linalg.norm(W[k, :]) ** 2)
                #solve for strain
                E = np.linalg.solve(dX, ds)
                #get volumes
                mvolumes[j] = old_div(np.abs(
                    np.dot(W[0, :], np.cross(W[1, :], W[2, :]))), 6.0)
                svolumes[j] = old_div(np.abs(
                    np.dot(w[0, :], np.cross(w[1, :], w[2, :]))), 6.0)
                strain[j, :, :] = [[E[0], E[3], E[5]],
                                   [E[3], E[1], E[4]],
                                   [E[5], E[4], E[2]]]
                #eigenvalues and eigenvectors of stress and strain tensors
                #eigenvectors are normalized
                eigstrain, eigstraindir = np.linalg.eigh(strain[j, :, :])
                order = np.argsort(eigstrain)
                eigstrain = eigstrain[order]
                eigstraindir /= np.linalg.norm(eigstraindir, axis=0, keepdims=True)
                eigstraindir = eigstraindir[:, order]
                pstrain.append(eigstrain)
                pstraindir.append(eigstraindir)
                eigstress, eigstressdir = np.linalg.eigh(stress[j, :, :])
                order = np.argsort(eigstress)
                eigstress = eigstress[order]
                eigstressdir /= np.linalg.norm(eigstressdir, axis=0, keepdims=True)
                eigstressdir = eigstressdir[:, order]
                pstress.append(eigstress)
                pstressdir.append(eigstressdir)
            pstress = np.array(pstress)
            pstressdir = np.array(pstressdir)
            pstrain = np.array(pstrain)
            pstraindir = np.array(pstraindir)
            #save reference volumes
            self.volumes.update({f: mvolumes})
            self.results['Effective Strain (von Mises)'].update(
                {f: np.sqrt(old_div(((pstrain[:, 2] - pstrain[:, 1]) ** 2 +
                             (pstrain[:, 1] - pstrain[:, 0]) ** 2 +
                             (pstrain[:, 2] - pstrain[:, 0]) ** 2),
                            2.0))})
            self.results['Maximum Compressive Strain'].update(
                {f: np.outer(pstrain[:, 0], [1 , 1, 1]) * pstraindir[:, :, 0]})
            self.results['Maximum Tensile Strain'].update(
                {f: np.outer(pstrain[:, 2], [1, 1, 1]) * pstraindir[:, :, 2]})
            self.results['Maximum Shear Strain'].update(
                {f: 0.5 * (pstrain[:, 2] - pstrain[:, 0])})
            self.results['Volumetric Strain'].update(
                {f: old_div(svolumes, mvolumes) - 1.0})

            self.results['Effective Stress (von Mises)'].update(
                {f: np.sqrt(old_div(((pstress[:, 2] - pstress[:, 1]) ** 2 +
                             (pstress[:, 1] - pstress[:, 0]) ** 2 +
                             (pstress[:, 2] - pstress[:, 0]) ** 2), 2.0))})
            self.results['Maximum Compressive Stress'].update(
                {f: np.outer(pstress[:, 0], [1 , 1, 1]) * pstressdir[:, :, 0]})
            self.results['Maximum Tensile Stress'].update(
                {f: np.outer(pstress[:, 2], [1, 1, 1]) * pstressdir[:, :, 2]})
            self.results['Maximum Shear Stress'].update(
                {f: 0.5 * (pstress[:, 2] - pstress[:, 0])})
            self.results['Pressure'].update(
                {f: old_div(np.sum(pstress, axis=1), 3.0)})

            self.results['Displacement'].update({f: displacement})

        for i, k in enumerate(self.outputs.keys()):
            if self.outputs[k].get():
                for m in self.matched:
                    weights = old_div(self.volumes[m[0]], np.sum(self.volumes[m[0]]))
                    for j, f in enumerate(m):
                        if len(self.results[k][f].shape) > 1:
                            dat = np.ravel(np.linalg.norm(self.results[k][f], axis=1))
                        else:
                            dat = np.ravel(self.results[k][f])
                        if self.analysis['Generate Histograms'].get():
                            IQR = np.subtract(*np.percentile(dat, [75, 25]))
                            nbins = (int(old_div(np.ptp(dat),
                                         (2 * IQR * dat.size ** (old_div(-1., 3.))))))
                            h = histogram(dat, numbins=nbins, weights=weights)
                            bins = np.linspace(h[1], h[1] + h[2] * nbins,
                                               nbins, endpoint=False)
                            self.histograms[k][f] = {'bins': bins,
                                                     'heights': h[0],
                                                     'width': h[2]}
                        if self.analysis['Tukey Boxplots'].get():
                            quantiles = np.zeros(3, float)
                            for n, q in enumerate([0.25, 0.5, 0.75]):
                                quantiles[n] = quantile_1D(dat, weights, q)
                            self.boxwhiskers[k][f] = {'quantiles': quantiles,
                                                      'data': dat}
                    if self.analysis['Calculate Differences'].get():
                        for c in itertools.combinations(m, 2):
                            if len(self.results[k][c[0]].shape) > 1:
                                dat1 = np.ravel(np.linalg.norm(self.results[k][c[0]], axis=1))
                                dat2 = np.ravel(np.linalg.norm(self.results[k][c[1]], axis=1))
                            else:
                                dat1 = np.ravel(self.results[k][c[0]])
                                dat2 = np.ravel(self.results[k][c[1]])
                            difference = dat2 - dat1
                            wrms = np.sqrt(np.average(difference ** 2,
                                                      weights=weights))
                            self.differences[k][c[1] + "MINUS" + c[0]] = {
                                'difference': difference, 'weighted RMS': wrms}
        self.saveResults()
        print("... ... Analysis Complete")

    def saveResults(self):
        top_dir = self.pickles[0].rsplit(os.sep, 2)[0]
        ts = datetime.datetime.fromtimestamp(
            time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.sep.join([top_dir, "FEA_analysis_" + ts])
        os.mkdir(output_dir)
        for output in list(self.histograms.keys()):
            if self.histograms[output] and self.analysis["Generate Histograms"].get():
                try:
                    os.mkdir(os.sep.join([output_dir, "histograms"]))
                except:
                    pass
                for m in self.matched:
                    fig = Figure()
                    FigureCanvasTkAgg(fig, self)
                    ax = fig.add_subplot(111)
                    #fig.set_size_inches(2.5, 2.5)
                    for f in m:
                        trunc_name = f.rsplit(os.sep, 2)
                        ax.bar(self.histograms[output][f]['bins'],
                               self.histograms[output][f]['heights'],
                               self.histograms[output][f]['width'],
                               label=trunc_name[1])
                    object_name = trunc_name[2].replace("cellFEA", "Cell ")
                    object_name = object_name.replace(".pkl", "")
                    ax.set_title(output + ' ' + object_name)
                    cell_directory = os.path.join(output_dir, "histograms", object_name.replace(" ", "_"))
                    try:
                        os.mkdir(cell_directory)
                    except:
                        pass
                    fig.savefig(
                        os.path.join(cell_directory,
                                     output.replace(" ", "_") + ".svg"), bbox_inches='tight')
                    fig.clear()
                    del fig

        for output in list(self.boxwhiskers.keys()):
            if self.boxwhiskers[output] and self.analysis["Tukey Boxplots"].get():
                try:
                    os.mkdir(os.path.join(output_dir, "boxplots"))
                except:
                    pass
                for m in self.matched:
                    fig = Figure()
                    FigureCanvasTkAgg(fig, self)
                    ax = fig.add_subplot(111)
                    x = []
                    q = []
                    labels = []
                    for f in m:
                        trunc_name = f.rsplit(os.sep, 2)
                        x.append(self.boxwhiskers[output][f]['data'])
                        q.append(self.boxwhiskers[output][f]['quantiles'])
                        labels.append(trunc_name[1])
                    q = np.array(q)
                    a = ax.boxplot(x)
                    object_name = trunc_name[2].replace("cellFEA", "Cell ")
                    object_name = object_name.replace(".pkl", "")
                    for i in range(q.shape[0]):
                        a['medians'][i].set_ydata(q[i, 1])
                        a['boxes'][i]._xy[[0, 1, 4], 1] = q[i, 0]
                        a['boxes'][i]._xy[[2, 3], 1] = q[i, 2]
                        iqr = q[i, 2] - q[i, 0]
                        y = x[i]
                        top = np.max(y[y < q[i, 2] + 1.5 * iqr])
                        bottom = np.min(y[y > q[i, 0] - 1.5 * iqr])
                        a['whiskers'][2 * i].set_ydata(
                            np.array([q[i, 0], bottom]))
                        a['whiskers'][2 * i + 1].set_ydata(
                            np.array([q[i, 2], top]))
                        a['caps'][2 * i].set_ydata(np.array([bottom, bottom]))
                        a['caps'][2 * i + 1].set_ydata(np.array([top, top]))

                        low_fliers = y[y < bottom]
                        high_fliers = y[y > top]
                        fliers = np.concatenate((low_fliers, high_fliers))
                        a['fliers'][2 * i].set_ydata(fliers)
                        a['fliers'][2 * i].set_xdata([i + 1] * fliers.size)

                    object_name = trunc_name[2].replace("cellFEA", "Cell ")
                    object_name = object_name.replace(".pkl", "")
                    ax.set_title(object_name)
                    ax.set_ylabel(output)
                    ax.set_xticklabels(labels, rotation=45)
                    cell_directory = os.path.join(output_dir, "boxplots",
                            object_name.replace(" ", "_"))
                    try:
                        os.mkdir(cell_directory)
                    except:
                        pass
                    fig.savefig(
                        os.path.join(cell_directory,
                                     output.replace(" ", "_") + ".svg"), bbox_inches="tight")
                    fig.clear()
                    del fig

        for output in list(self.differences.keys()):
            if self.differences[output] and self.analysis["Calculate Differences"].get():
                try:
                    os.mkdir(os.sep.join([output_dir, "paired_differences"]))
                except:
                    pass
                for m in self.matched:
                    labels = []
                    for f in m:
                        trunc_name = f.rsplit(os.sep, 2)
                        labels.append(trunc_name[1])
                    N = len(m)
                    grid = np.zeros((N, N), float)
                    combos = list(itertools.combinations(m, 2))
                    enum_combos = list(itertools.combinations(list(range(N)), 2))
                    for c, e in zip(combos, enum_combos):
                        grid[e[0], e[1]] = (self.differences[output]
                                            ["".join([c[1], 'MINUS', c[0]])]
                                            ['weighted RMS'])
                        grid[e[1], e[0]] = grid[e[0], e[1]]
                    fig = Figure()
                    FigureCanvasTkAgg(fig, self)
                    ax = fig.add_subplot(111)
                    high = np.max(np.ravel(grid))
                    low = np.min(np.ravel(grid))
                    if abs(low - high) < 1e-3:
                        high += 0.001
                    heatmap = ax.pcolormesh(
                        grid, cmap=cm.Blues, edgecolors='black',
                        vmin=low, vmax=high)
                    ax.set_xticks(np.arange(N) + 0.5, minor=False)
                    ax.set_yticks(np.arange(N) + 0.5, minor=False)
                    ax.invert_yaxis()
                    ax.xaxis.tick_top()
                    ax.set_xticklabels(labels, minor=False)
                    ax.set_yticklabels(labels, minor=False)
                    fig.colorbar(heatmap)

                    object_name = trunc_name[2].replace("cellFEA", "Cell ")
                    object_name = object_name.replace(".pkl", "")
                    ax.set_title(object_name, y=1.08)
                    cell_directory = os.path.join(
                        output_dir, "paired_differences",
                        object_name.replace(" ", "_"))
                    try:
                        os.mkdir(cell_directory)
                    except:
                        pass
                    fig.savefig(
                        os.path.join(cell_directory, output.replace(
                            " ", "_") + ".svg"), bbox_inches="tight")
                    fig.clear()
                    del fig

        if self.analysis['Convert to VTK'].get():
            try:
                os.mkdir(os.path.join(output_dir, "vtk"))
            except:
                pass
            for m in self.matched:
                try:
                    self.vtkMeshes[m[0]]
                except:
                    self.makeVTK(m[0])
                for output in list(self.outputs.keys()):
                    if self.outputs[output].get():
                        for f in m:
                            trunc_name = f.rsplit(os.sep, 2)
                            vtkArray = numpy_support.numpy_to_vtk(
                                np.ravel(self.results[output][f]).astype('f'),
                                deep=True, array_type=vtk.VTK_FLOAT)
                            shape = self.results[output][f].shape
                            if len(shape) == 2:
                                vtkArray.SetNumberOfComponents(shape[1])
                            elif len(shape) == 1:
                                vtkArray.SetNumberOfComponents(1)
                            else:
                                print(("WARNING: {:s} has rank {:d}".format(f, len(shape))))
                                continue
                            vtkArray.SetName(trunc_name[1] + ' ' + output)
                            self.vtkMeshes[m[0]].GetCellData().AddArray(
                                vtkArray)
                for f in m:
                    arr = numpy_support.numpy_to_vtk(np.ravel(self.results['Displacement'][f]).astype('f'),
                                                    deep=True, array_type=vtk.VTK_FLOAT)
                    arr.SetName(trunc_name[1] + ' ' + 'Displacement')
                    arr.SetNumberOfComponents(3)
                    self.vtkMeshes[m[0]].GetPointData().AddArray(arr)

                object_name = trunc_name[2].replace("cellFEA", "Cell")
                object_name = object_name.replace(".pkl", ".vtu")
                idWriter = vtk.vtkXMLUnstructuredGridWriter()
                idWriter.SetFileName(os.path.join(output_dir, 'vtk', object_name))
                idWriter.SetInputData(self.vtkMeshes[m[0]])
                idWriter.Write()

    def makeVTK(self, filename):
        nelements = len(self.data[filename]['elements'])
        n = np.array(self.data[filename]['nodes'])
        arr = numpy_support.numpy_to_vtk(
            n.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        arr.SetNumberOfComponents(3)
        tetraPoints = vtk.vtkPoints()
        tetraPoints.SetData(arr)

        vtkMesh = vtk.vtkUnstructuredGrid()
        vtkMesh.Allocate(nelements, nelements)
        vtkMesh.SetPoints(tetraPoints)

        e = np.array(self.data[filename]['elements'], np.uint32) - 1
        e = np.hstack((np.ones((e.shape[0], 1), np.uint32) * 4, e))
        arr = numpy_support.numpy_to_vtk(e.ravel(), deep=True,
                                         array_type=vtk.VTK_ID_TYPE)
        tet = vtk.vtkCellArray()
        tet.SetCells(old_div(e.size, 5), arr)
        vtkMesh.SetCells(10, tet)

        centroids = (n[e[:, 1]] + n[e[:, 2]] + n[e[:, 3]] + n[e[:, 4]])
        centroids /= 4.0
        arr = numpy_support.numpy_to_vtk(
            centroids.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        arr.SetName("Centroids")
        arr.SetNumberOfComponents(3)
        vtkMesh.GetCellData().AddArray(arr)
        self.vtkMeshes[filename] = vtkMesh

    def findLocalCsys(self, filename):
        poly = vtk.vtkGeometryFilter()
        poly.SetInputData(self.vtkMeshes[filename])
        poly.Update()

        distanceFilter = vtk.vtkImplicitPolyDataDistance()
        distanceFilter.SetInput(poly.GetOutput())

        gradients = vtk.vtkFloatArray()
        gradients.SetNumberOfComponents(3)
        gradients.SetName("LevelSet Normals")

        distances = vtk.vtkFloatArray()
        distances.SetNumberOfComponents(1)
        distances.SetName("Signed Distances")

        N = self.vtkMeshes[filename].GetCellData().GetArray(
            "Centroids").GetNumberOfTuples()
        for i in range(N):
            g = np.zeros(3, np.float32)
            p = self.vtkMeshes[filename].GetCellData().GetArray(
                "Centroids").GetTuple3(i)
            d = distanceFilter.EvaluateFunction(p)
            distanceFilter.EvaluateGradient(p, g)
            g = old_div(np.array(g), np.linalg.norm(np.array(g)))
            gradients.InsertNextTuple(g)
            distances.InsertNextValue(d)
        self.vtkMeshes[filename].GetCellData().AddArray(gradients)
        self.vtkMeshes[filename].GetCellData().AddArray(distances)


root = Tk()
root.title("Welcome to the FEBio pyCellAnalyst utility.")
app = Application(root)
root.mainloop()
