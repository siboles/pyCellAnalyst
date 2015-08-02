from Tkinter import *
from ttk import Notebook
import tkFileDialog
import os
import webbrowser
import pickle
import copy
import xlrd
import time
import datetime
import string
from pyCellAnalyst import (Volume, CellMech)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict


class Application(Frame):
    """ This is a GUI for the pyCellAnalyst Segmentation Feature """
    def __init__(self, master):
        Frame.__init__(self, master)
        self.lastdir = os.getcwd()
        self.notebook = Notebook(self)
        self.tab1 = Frame(self.notebook)
        self.tab2 = Frame(self.notebook)
        self.tab3 = Frame(self.notebook)
        self.tab4 = Frame(self.notebook)

        self.notebook.add(self.tab1, text="I/O")
        self.notebook.add(self.tab2, text="Filtering")
        self.notebook.add(self.tab3, text="Segmentation")
        self.notebook.add(self.tab4, text="Kinematics")
        self.notebook.grid(row=0, column=0, sticky=NW)
        self.directories = []
        self.ROI = []

        #default settings
        self.settings = {'xdim': DoubleVar(value=0.41),
                         'ydim': DoubleVar(value=0.41),
                         'zdim': DoubleVar(value=0.3),
                         'upsampling': DoubleVar(value=2.0),
                         'thresholdPercentage': DoubleVar(value=0.55),
                         'medianRadius': IntVar(value=1),
                         'gaussianSigma': DoubleVar(value=0.5),
                         'curvatureIterations': IntVar(value=10),
                         'curvatureConductance': DoubleVar(value=9.0),
                         'gradientIterations': IntVar(value=10),
                         'gradientConductance': DoubleVar(value=9.0),
                         'gradientTimeStep': DoubleVar(value=0.01),
                         'bilateralDomainSigma': DoubleVar(value=1.5),
                         'bilateralRangeSigma': DoubleVar(value=5.0),
                         'bilateralSamples': IntVar(value=100),
                         'patchRadius': IntVar(value=4),
                         'patchNumber': IntVar(value=20),
                         'patchNoiseModel': IntVar(value=2),
                         'patchIterations': IntVar(value=5),
                         'geodesicCannyVariance': DoubleVar(value=0.1),
                         'geodesicCannyUpper': DoubleVar(value=0.0),
                         'geodesicCannyLower': DoubleVar(value=0.0),
                         'geodesicPropagation': DoubleVar(value=0.15),
                         'geodesicCurvature': DoubleVar(value=0.1),
                         'geodesicAdvection': DoubleVar(value=1.0),
                         'geodesicIterations': IntVar(value=200),
                         'geodesicRMS': DoubleVar(value=0.01),
                         'edgeLambda1': DoubleVar(value=1.1),
                         'edgeLambda2': DoubleVar(value=1.0),
                         'edgeIterations': IntVar(value=20),
                         'edgeCurvature': DoubleVar(value=10.0),
                         'deformableIterations': IntVar(value=200),
                         'deformableRMS': DoubleVar(value=0.01),
                         'deformableSigma': DoubleVar(value=3.0),
                         'deformablePrecision': DoubleVar(value=0.02)}

        self.intSettings = {'stain': IntVar(value=0),
                            'display': IntVar(value=1),
                            'removeBright': IntVar(value=0),
                            'edgeEnhancement': IntVar(value=0),
                            'processAs2D': IntVar(value=0),
                            'opening': IntVar(value=1),
                            'fillHoles': IntVar(value=0),
                            'handleOverlap': IntVar(value=1),
                            'debug': IntVar(value=0),
                            'smoothingMethod': IntVar(value=4),
                            'thresholdMethod': IntVar(value=3),
                            'thresholdAdaptive': IntVar(value=1),
                            'activeMethod': IntVar(value=1),
                            'rigidInitial': IntVar(value=1),
                            'defReg': IntVar(value=0),
                            'saveFEA': IntVar(value=0),
                            'makePlots': IntVar(value=1),
                            'deformableDisplay': IntVar(value=1),
                            'depth_adjust': IntVar(value=0)}

        self.smoothingMethods = ['None',
                                 'Median',
                                 'Gaussian',
                                 'Curvature Diffusion',
                                 'Gradient Diffusion',
                                 'Bilateral',
                                 'Patch-based']

        self.thresholdMethods = ['Percentage',
                                 'Otsu',
                                 'MaxEntropy',
                                 'Li',
                                 'Huang',
                                 'IsoData',
                                 'KittlerIllingworth',
                                 'Moments',
                                 'Yen',
                                 'RenyiEntropy',
                                 'Shanbhag']

        for i in self.settings.keys():
            self.settings[i].trace("w", self.update)
        self.grid()
        self.create_widgets()
        #eliminates the focus switch to first widget
        #in frames when changing tabs
        self.notebook.focus()

    def create_widgets(self):
        #save/load settings
        self.buttonSaveSettings = Button(self.tab1, text="Save Settings",
                                         command=self.saveSettings)
        self.buttonSaveSettings.grid(row=1, column=0, padx=5, pady=5,
                                     sticky=W + E)
        self.buttonLoadSettings = Button(self.tab1, text="Load Settings",
                                         command=self.loadSettings)
        self.buttonLoadSettings.grid(row=1, column=1, padx=5, pady=5,
                                     sticky=W + E)
        self.buttonLoadROI = Button(self.tab1,
                                    text="Load Region of Interest File",
                                    command=self.loadROI)
        self.buttonLoadROI.grid(row=1, column=2, padx=5, pady=5, sticky=W + E)
        #create label frame for image directory selection
        self.directoryFrame = LabelFrame(self.tab1,
                                         text="Image directories to process")
        self.directoryFrame.grid(row=2, column=0, rowspan=2, columnspan=5,
                                 padx=5, pady=5, sticky=NW)
        #add directory
        self.buttonAddDirectory = Button(self.directoryFrame, bg='green')
        self.buttonAddDirectory["text"] = "Add"
        self.buttonAddDirectory["command"] = self.add_directory
        self.buttonAddDirectory.grid(row=2, column=0, padx=5, sticky=W + E)
        #remove directory
        self.buttonRemoveDirectory = Button(self.directoryFrame, bg='red')
        self.buttonRemoveDirectory["text"] = "Remove"
        self.buttonRemoveDirectory["command"] = self.remove_directory
        self.buttonRemoveDirectory.grid(row=3, column=0, padx=5, sticky=W + E)

        #directory list
        self.listDirectories = Listbox(self.directoryFrame)
        self.listDirectories["width"] = 80
        self.listDirectories["selectmode"] = MULTIPLE
        self.listDirectories.grid(row=2, column=1, rowspan=2, columnspan=4,
                                  padx=5, pady=5, sticky=E + W)

        # Image Settings
        self.imageSettingsFrame = LabelFrame(self.tab1, text="Image Settings")
        self.imageSettingsFrame.grid(row=4, column=0, columnspan=5, rowspan=2,
                                     padx=5, pady=5, sticky=E + W)
        settings = [('x-spacing', 'xdim'),
                    ('y-spacing', 'ydim'),
                    ('z-spacing', 'zdim'),
                    ('Upsampling Factor', 'upsampling')]
        for i, v in enumerate(settings):
            Label(self.imageSettingsFrame, text=v[0]).grid(row=4, column=i,
                                                           padx=5, pady=5,
                                                           sticky=E + W)
            Entry(self.imageSettingsFrame,
                  textvariable=self.settings[v[1]],
                  width=5).grid(row=5, column=i,
                                padx=5, pady=5, sticky=E + W)
        Checkbutton(self.imageSettingsFrame,
                    text='Objects are Dark',
                    variable=self.intSettings['stain']).grid(row=5,
                                                             column=i + 1,
                                                             padx=5, pady=5,
                                                             sticky=NW)

        # Other settings
        ####################################################################
        self.otherSettingsFrame = LabelFrame(self.tab1, text="Other Options")
        self.otherSettingsFrame.grid(row=6, column=0, columnspan=5,
                                     padx=5, pady=5, sticky=E + W)
        settings = [('Display Objects', 'display'),
                    ('Remove Bright Spots', 'removeBright'),
                    ('Edge Enhancement', 'edgeEnhancement'),
                    ('Consider Slices Independent', 'processAs2D'),
                    ('Perform Morphogical Opening', 'opening'),
                    ('Fill Holes', 'fillHoles'),
                    ('Handle Overlap', 'handleOverlap'),
                    ('Debug Mode', 'debug'),
                    ('Equalize Intensity Over Depth', 'depth_adjust')]
        row = 6
        shift = 0
        for i, v in enumerate(settings):
            Checkbutton(self.otherSettingsFrame,
                        text=v[0],
                        variable=self.intSettings[v[1]]).grid(row=row,
                                                              column=i - shift,
                                                              padx=5, pady=5,
                                                              sticky=NW)
            if (i + 1) % 3 == 0:
                row += 1
                shift = i + 1

        ######################################################################
        #button to execute segmentation(s)
        self.buttonExecute = Button(self.tab1,
                                    bg='green',
                                    font=('Helvetica', '20', 'bold'))
        self.buttonExecute["text"] = "Execute Segmentation"
        self.buttonExecute["command"] = self.run_segmentation
        self.buttonExecute.grid(row=row + 1, column=0, columnspan=5,
                                padx=5, pady=5, sticky=W + E)
        #smoothing/denoising
        methods = [("None", 1),
                   ("Median", 2),
                   ("Gaussian", 3),
                   ("Curvature-based\nAnisotropic Diffusion", 4),
                   ("Gradient-based\nAnisotropic Diffusion", 5),
                   ("Bilateral", 6),
                   ("Patch-based Denoising", 7)]
        self.smoothingFrame = LabelFrame(self.tab2, text="Smoothing/Denoising")
        self.smoothingFrame.grid(row=0, column=0, padx=5, pady=5, sticky=NW)

        for m, i in methods:
            Radiobutton(self.smoothingFrame,
                        text=m,
                        indicatoron=0,
                        padx=5,
                        width=20,
                        variable=self.intSettings['smoothingMethod'],
                        command=self.populateSmoothingSettings,
                        value=i).pack(anchor=W)

        self.smoothingHelpFrame = LabelFrame(self.tab2, text="Description")
        self.smoothingHelpFrame.grid(row=0, column=1, padx=5, pady=5,
                                     sticky=NW)
        self.textSmoothingHelp = Text(self.smoothingHelpFrame, wrap=WORD,
                                      height=11, width=40)
        self.textSmoothingHelp.insert(END, ("Apply an iterative curvature-bas"
                                            "ed anisotropic diffusion filter. "
                                            "Higher conductance will result in"
                                            " more change per iteration. More "
                                            "iterations will result in a "
                                            "smoother image. This filter shoul"
                                            "d preserve edges. It is better at"
                                            " retaining fine features than "
                                            "gradient-based anisotropic "
                                            "diffusion, and also better when "
                                            "the edge contrast is low."))
        self.textSmoothingHelp.pack(anchor=NW)
        self.textSmoothingHelp["state"] = DISABLED
        self.smoothingLink = r"http://www.itk.org/ItkSoftwareGuide.pdf"
        self.smoothingReference = Label(self.smoothingHelpFrame,
                                        text="Reference",
                                        fg="blue", cursor="hand2")
        self.smoothingReference.bind("<Button-1>",
                                     self.open_smoothing_reference)
        self.smoothingReference.pack(anchor=NW)

        self.smoothingSettingsFrame = LabelFrame(
            self.tab2,
            text="Smoothing/Denoising Settings")
        self.smoothingSettingsFrame.grid(row=0, column=2,
                                         padx=5, pady=5, sticky=NW)
        settings = [('Conductance', 'curvatureConductance'),
                    ('Iterations', 'curvatureIterations')]

        for t, v in settings:
            Label(self.smoothingSettingsFrame, text=t).pack(anchor=W)
            Entry(self.smoothingSettingsFrame,
                  textvariable=self.settings[v]).pack(anchor=W)

        #Threshold segmentation
        self.thresholdFrame = LabelFrame(self.tab3,
                                         text="Threshold segmentation")
        self.thresholdFrame.grid(row=0, column=0, padx=5, pady=5, sticky=NW)

        methods = [("Percentage", 1),
                   ("Otsu", 2),
                   ("Maximum Entropy", 3),
                   ("Li", 4),
                   ("Huang", 5),
                   ("IsoData (Ridler-Calvard)", 6),
                   ("KittlerIllingworth", 7),
                   ("Moments", 8),
                   ("Yen", 9),
                   ("RenyiEntropy", 10),
                   ("Shanbhag", 11)]

        for m, i in methods:
            Radiobutton(self.thresholdFrame,
                        text=m,
                        indicatoron=0,
                        padx=5,
                        width=20,
                        variable=self.intSettings['thresholdMethod'],
                        command=self.populateThresholdSettings,
                        value=i).pack(anchor=W)

        self.thresholdHelpFrame = LabelFrame(self.tab3, text="Description")
        self.thresholdHelpFrame.grid(row=0, column=1,
                                     padx=5, pady=5, sticky=NW)
        self.textThresholdHelp = Text(self.thresholdHelpFrame, wrap=WORD,
                                      height=11, width=40)
        self.textThresholdHelp.insert(END, ("Calculates the threshold such"
                                            " that entropy is maximized "
                                            "between the foreground and "
                                            "background. This has shown "
                                            "good performance even when "
                                            "objects are in close "
                                            "proximity."))
        self.textThresholdHelp["state"] = DISABLED
        self.thresholdLink = (r"http://dx.doi.org/10.1016/"
                              "0734-189X(85)90125-2")

        self.thresholdReference = Label(self.thresholdHelpFrame,
                                        text="", fg="blue", cursor="hand2")
        self.textThresholdHelp.pack(anchor=NW)

        self.thresholdSettingsFrame = LabelFrame(self.tab3,
                                                 text="Threshold Settings")
        Checkbutton(
            self.thresholdSettingsFrame,
            text="Iterative Threshold Adjustment",
            variable=self.intSettings['thresholdAdaptive']).pack(anchor=W)
        self.thresholdSettingsFrame.grid(row=0, column=2, padx=5, pady=5,
                                         sticky=NW)
        Label(self.thresholdSettingsFrame,
              text="     No additional settings needed.",
              fg='red').pack(anchor=W)

        #make Active Contour segmentation frame
        self.activeContourFrame = LabelFrame(
            self.tab3, text="Active contour segmentation")
        self.activeContourFrame.grid(row=1, column=0,
                                     padx=5, pady=5, sticky=NW)

        self.activeHelpFrame = LabelFrame(self.tab3, text="Description")
        self.activeHelpFrame.grid(row=1, column=1,
                                  padx=5, pady=5, sticky=NW)
        self.textActiveHelp = Text(self.activeHelpFrame,
                                   wrap=WORD, height=11, width=40)
        self.textActiveHelp.insert(END, ("Only a threshold-based "
                                         "segmentation will "
                                         "be performed."))

        self.textActiveHelp.pack(anchor=NW)
        self.textActiveHelp["state"] = DISABLED
        self.activeLink = ""
        self.activeReference = Label(self.activeHelpFrame,
                                     text="",
                                     fg="blue", cursor="hand2")
        self.activeReference.bind("<Button-1>", self.open_active_reference)
        self.activeReference.pack(anchor=NW)

        methods = [("None", 1),
                   ("Geodesic Levelset", 2),
                   ("Edge-free", 3)]

        for m, i in methods:
            Radiobutton(self.activeContourFrame,
                        text=m,
                        indicatoron=0,
                        padx=5,
                        width=20,
                        variable=self.intSettings['activeMethod'],
                        command=self.populateActiveContourSettings,
                        value=i).pack(anchor=W)

        self.activeSettingsFrame = LabelFrame(self.tab3,
                                              text="Active Contour Settings")
        self.activeSettingsFrame.grid(row=1, column=2,
                                      padx=5, pady=5, sticky=NW)
        Label(self.activeSettingsFrame,
              text="No additional settings needed.").pack(anchor=W)

        ##### Kinematics Tab ######
        #create label frame for image directory selection
        self.materialFrame = LabelFrame(
            self.tab4,
            text="Directory containing reference configuration data")
        self.materialFrame.grid(row=0, column=0,
                                rowspan=1, columnspan=5,
                                padx=5, pady=5, sticky=NW)
        #add directory
        self.buttonAddMaterialDirectory = Button(self.materialFrame,
                                                 bg='green')
        self.buttonAddMaterialDirectory["text"] = "Load"
        self.buttonAddMaterialDirectory["command"] = self.add_material_dir
        self.buttonAddMaterialDirectory.grid(row=0, column=0,
                                             padx=5, pady=5, sticky=W + E)
        self.materialDirectoryLabel = Label(self.materialFrame, text="")
        self.materialDirectoryLabel.grid(row=0, column=1,
                                         padx=5, pady=5, sticky=W + E)
        self.spatialFrame = LabelFrame(
            self.tab4,
            text="Directories containing deformed configuration data")
        self.spatialFrame.grid(row=1, column=0,
                               rowspan=2, columnspan=2,
                               padx=5, pady=5, sticky=NW)
        #add directory
        self.buttonAddSpatialDirectory = Button(self.spatialFrame,
                                                bg='green')
        self.buttonAddSpatialDirectory["text"] = "Add"
        self.buttonAddSpatialDirectory["command"] = self.add_spatial_dir
        self.buttonAddSpatialDirectory.grid(row=1, column=0,
                                            padx=5, sticky=W + E)
        #remove directory
        self.buttonRemoveSpatialDirectory = Button(self.spatialFrame, bg='red')
        self.buttonRemoveSpatialDirectory["text"] = "Remove"
        self.buttonRemoveSpatialDirectory["command"] = self.remove_spatial_dir
        self.buttonRemoveSpatialDirectory.grid(row=2, column=0,
                                               padx=5, pady=5, sticky=W + E)

        #directory list
        self.spatialDirectories = []
        self.listSpatialDirectories = Listbox(self.spatialFrame)
        self.listSpatialDirectories["width"] = 80
        self.listSpatialDirectories["selectmode"] = MULTIPLE
        self.listSpatialDirectories.grid(row=1, column=1,
                                         rowspan=2, padx=5, pady=5,
                                         sticky=E + W)

        #Options
        self.kinematicsOptionsFrame = LabelFrame(
            self.tab4, text="Kinematics Analysis Options")
        self.kinematicsOptionsFrame.grid(row=3, column=0,
                                         padx=5, pady=5, sticky=E + W)
        settings = [('Align with a rigid rotation initially', 'rigidInitial'),
                    ('Perform Deformable Image Registration', 'defReg'),
                    ('Save for Finite Element Analysis', 'saveFEA'),
                    ('Display Registration Results', 'deformableDisplay'),
                    ('Generate Plots', 'makePlots')]

        for i, (t, v) in enumerate(settings):
            if i == 1:
                Checkbutton(
                    self.kinematicsOptionsFrame,
                    text=t,
                    variable=self.intSettings[v],
                    command=self.populateDeformableSettings).grid(row=3 + i,
                                                                  column=0,
                                                                  padx=5,
                                                                  pady=5,
                                                                  sticky=NW)
            else:
                Checkbutton(self.kinematicsOptionsFrame,
                            text=t,
                            variable=self.intSettings[v]).grid(row=3 + i,
                                                               column=0,
                                                               padx=5,
                                                               pady=5,
                                                               sticky=NW)

        self.deformableSettingsFrame = LabelFrame(
            self.tab4, text="Deformable Image Registration Settings")
        self.deformableSettingsFrame.grid(row=3, column=1,
                                          padx=5, pady=5, sticky=E + W)

        #button to execute segmentation(s)
        self.buttonExecuteKinematics = Button(self.tab4, bg='green',
                                              font=('Helvetica', '20', 'bold'))
        self.buttonExecuteKinematics["text"] = "Execute Analysis"
        self.buttonExecuteKinematics["command"] = self.run_kinematics
        self.buttonExecuteKinematics.grid(row=7, column=0, columnspan=2,
                                          padx=5, pady=5, sticky=W + E)
        ##### End Kinematics Tab #####

    def populateSmoothingSettings(self):
        for child in self.smoothingSettingsFrame.pack_slaves():
            child.destroy()
        if self.intSettings['smoothingMethod'].get() == 1:
            self.smoothingReference.unbind("<Button-1>")
            self.smoothingReference["text"] = ""
        elif self.intSettings['smoothingMethod'].get() == 2:
            self.smoothingReference.bind("<Button-1>",
                                         self.open_smoothing_reference)
            self.smoothingReference["text"] = "Reference: page 80"
        elif self.intSettings['smoothingMethod'].get() == 3:
            self.smoothingReference.bind("<Button-1>",
                                         self.open_smoothing_reference)
            self.smoothingReference["text"] = "Reference: page 96"
        elif self.intSettings['smoothingMethod'].get() == 4:
            self.smoothingReference.bind("<Button-1>",
                                         self.open_smoothing_reference)
            self.smoothingReference["text"] = "Reference: page 106"
        else:
            self.smoothingReference.bind("<Button-1>",
                                         self.open_smoothing_reference)
            self.smoothingReference["text"] = "Reference"

        if self.intSettings['smoothingMethod'].get() == 1:
            Label(self.smoothingSettingsFrame,
                  text="No additional settings needed.").pack(anchor=W)
            self.textSmoothingHelp["state"] = NORMAL
            self.textSmoothingHelp.delete("0.0", END)
            self.textSmoothingHelp.insert(END, ("No smoothing or denoising "
                                                "will be applied to the "
                                                "image."))
            self.textSmoothingHelp["state"] = DISABLED
            self.smoothingLink = r""

        elif self.intSettings['smoothingMethod'].get() == 2:
            settings = [('Kernel Radius', 'medianRadius')]
            self.textSmoothingHelp["state"] = NORMAL
            self.textSmoothingHelp.delete("0.0", END)
            self.textSmoothingHelp.insert(END, ("Apply a median filter with "
                                                "the window size defined by "
                                                "Kernel Radius. A larger "
                                                "radius will result in a "
                                                "smoother image, but may "
                                                "degrade the edges."))
            self.textSmoothingHelp["state"] = DISABLED
            self.smoothingLink = r"http://www.itk.org/ItkSoftwareGuide.pdf"
        elif self.intSettings['smoothingMethod'].get() == 3:
            settings = [('Gaussian Variance', 'gaussianSigma')]
            self.textSmoothingHelp["state"] = NORMAL
            self.textSmoothingHelp.delete("0.0", END)
            self.textSmoothingHelp.insert(END, ("Apply a discrete Gaussian "
                                                "filter. Increasing Gaussian "
                                                "Variance will result in a "
                                                "smoother image, but will "
                                                "further blur edges."))
            self.textSmoothingHelp["state"] = DISABLED
            self.smoothingLink = r"http://www.itk.org/ItkSoftwareGuide.pdf"

        elif self.intSettings['smoothingMethod'].get() == 4:
            settings = [('Conductance', 'curvatureConductance'),
                        ('Iterations', 'curvatureIterations')]
            self.textSmoothingHelp["state"] = NORMAL
            self.textSmoothingHelp.delete("0.0", END)
            self.textSmoothingHelp.insert(END, ("Apply an iterative curvature-"
                                                "based anisotropic diffusion "
                                                "filter. Higher conductance "
                                                "will result in more change "
                                                "per iteration. More itera"
                                                "tions will result in a smooth"
                                                "er image. This filter should"
                                                " preserve edges. It is "
                                                "better at retaining fine "
                                                "features than gradient-based"
                                                " anisotropic diffusion, and "
                                                "also better when the edge "
                                                "contrast is low."))
            self.textSmoothingHelp["state"] = DISABLED
            self.smoothingLink = r"http://www.itk.org/ItkSoftwareGuide.pdf"

        elif self.intSettings['smoothingMethod'].get() == 5:
            settings = [('Conductance', 'gradientConductance'),
                        ('Iterations', 'gradientIterations'),
                        ('Time Step', 'gradientTimeStep')]
            self.textSmoothingHelp["state"] = NORMAL
            self.textSmoothingHelp.delete("0.0", END)
            self.textSmoothingHelp.insert(END, ("Apply an iterative gradient-"
                                                "based anisotropic diffusion "
                                                "filter. Higher conductance "
                                                "will result in more change "
                                                "per iteration. More itera"
                                                "tions will result in a smooth"
                                                "er image. This filter should"
                                                " preserve edges. This may "
                                                "perform better than curvature"
                                                "-based if edge contrast "
                                                "is good."))
            self.textSmoothingHelp["state"] = DISABLED
            self.smoothingLink = r"http://dx.doi.org/10.1109/34.56205"

        elif self.intSettings['smoothingMethod'].get() == 6:
            settings = [
                ('Domain Variance (costly)', 'bilateralDomainSigma'),
                ('Range Variance', 'bilateralRangeSigma'),
                ('Samples', 'bilateralSamples')]
            self.textSmoothingHelp["state"] = NORMAL
            self.textSmoothingHelp.delete("0.0", END)
            self.textSmoothingHelp.insert(END, ("A bilateral filter is applied"
                                                " both on a neighborhood "
                                                "defined by the Euclidean "
                                                "distance (the domain) from"
                                                " a given voxel and a "
                                                "'neighborhood' based on "
                                                "voxel intensities (the range)"
                                                ". Two Gaussian kernels are "
                                                "defined by Domain Variance "
                                                "and Range Variance and the "
                                                "actual weight of influence a "
                                                "particular neighbor voxel has"
                                                " on the current voxel is a "
                                                "combination of both. A voxel"
                                                " that is both close in "
                                                "distance and similar in "
                                                "intensity will have a high"
                                                " weight. This results in "
                                                "edge-preserving smoothing."))
            self.textSmoothingHelp["state"] = DISABLED
            self.smoothingLink = r"http://dx.doi.org/10.1109/ICCV.1998.710815"

        elif self.intSettings['smoothingMethod'].get() == 7:
            Label(
                self.smoothingSettingsFrame,
                text='Warning: CPU cost grows rapidly with increasing values.',
                fg='red').pack(anchor=W)
            settings = [('Patch Radius', 'patchRadius'),
                        ('Number of Patches', 'patchNumber'),
                        ('Iterations', 'patchIterations')]
            self.textSmoothingHelp["state"] = NORMAL
            self.textSmoothingHelp.delete("0.0", END)
            self.textSmoothingHelp.insert(END, ("This filter will denoise "
                                                "the image using an unsuper"
                                                "vised information-theoretic"
                                                " adaptive filter "
                                                "(SEE REFERENCE). The algo"
                                                "rithm attempts to extract "
                                                "the noise based on random "
                                                "sampling of the image with "
                                                "windows of size, Patch Radius"
                                                ", and number, Number of "
                                                "Patches. No a priori "
                                                "knowledge of the noise is"
                                                " needed, but a Noise Model"
                                                " can be specified. Since "
                                                "laser fluorescence microscopy"
                                                " is known to have Poisson "
                                                "noise, this is the default."
                                                " The drawback of this method"
                                                " is it becomes extremely "
                                                "costly with increasing "
                                                "any of its parameters."))
            self.textSmoothingHelp["state"] = DISABLED
            self.smoothingLink = r"http:/dx.doi.org/10.1109/TPAMI.2006.64"

        if not(self.intSettings['smoothingMethod'].get() in [1, 7]):
            for t, v in settings:
                Label(self.smoothingSettingsFrame, text=t).pack(anchor=W)
                Entry(self.smoothingSettingsFrame,
                      textvariable=self.settings[v]).pack(anchor=W)

        if self.intSettings['smoothingMethod'].get() == 7:
            for t, v in settings:
                Label(self.smoothingSettingsFrame, text=t).pack(anchor=W)
                Entry(self.smoothingSettingsFrame,
                      textvariable=self.settings[v]).pack(anchor=W)
            models = [("No Model", 1),
                      ("Poisson", 2),
                      ("Gaussian", 3),
                      ("Rician", 4)]
            Label(self.smoothingSettingsFrame,
                  text="Noise Model").pack(anchor=NW)
            for t, v in models:
                Radiobutton(self.smoothingSettingsFrame,
                            text=t,
                            indicatoron=0,
                            padx=5,
                            width=8,
                            variable=self.settings['patchNoiseModel'],
                            value=v).pack(side=LEFT)

    def populateThresholdSettings(self):
        for child in self.thresholdSettingsFrame.pack_slaves():
            child.destroy()
        Checkbutton(
            self.thresholdSettingsFrame,
            text="Iterative Threshold Adjustment",
            variable=self.intSettings['thresholdAdaptive']).pack(anchor=W)
        if self.intSettings['thresholdMethod'].get() == 1:
            self.thresholdReference.unbind("<Button-1>")
            self.thresholdReference['text'] = ""
        else:

            self.thresholdReference['text'] = "Reference"
            self.thresholdReference.bind("<Button-1>",
                                         self.open_threshold_reference)
            self.thresholdReference.pack(anchor=NW)

        if self.intSettings['thresholdMethod'].get() == 1:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("Thresholds at a user-specifie"
                                                "d ratio of the maximum voxel "
                                                "intensity."))
            self.textThresholdHelp["state"] = DISABLED
            Label(self.thresholdSettingsFrame, text="Ratio").pack(anchor=W)
            e = Entry(self.thresholdSettingsFrame,
                      textvariable=self.settings['thresholdPercentage'])
            e.pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 2:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("The method determines the "
                                                "threshold that maximizes the"
                                                " 'total variance of levels';"
                                                " equation 12 in reference. "
                                                "Performs poorly when objects "
                                                "are in close proximity."))
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = r"http://dx.doi.org/10.1109/TSMC.1979.4310076"
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed",
                  fg='red').pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 3:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("Calculates the threshold such"
                                                " that entropy is maximized "
                                                "between the foreground and "
                                                "background. This has shown "
                                                "good performance even when "
                                                "objects are in close "
                                                "proximity."))
            self.textThresholdHelp.pack(anchor=NW)
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = (r"http://dx.doi.org/10.1016/"
                                  "0734-189X(85)90125-2")
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed.",
                  fg='red').pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 4:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("An iterative method to "
                                                "minimize cross-entropy of "
                                                "the gray and binarized image."
                                                " Performs poorly when objects"
                                                " are in close proximity."))
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = (r"http://dx.doi.org/10.1016/"
                                  "S0167-8655(98)00057-9")
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed.",
                  fg='red').pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 5:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("Thresholds the image using "
                                                "fuzzy set theory. The "
                                                "'index of fuzziness' between "
                                                "the binarized and the gray "
                                                "image is calculated using "
                                                "Shannon's function. The "
                                                "threshold level that minimize"
                                                "s the index of fuzziness is "
                                                "chosen. Performs poorly when"
                                                " objects are in close "
                                                "proximity."))
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = (r"http://dx.doi.org/10.1016"
                                  "/0031-3203(94)E0043-K")
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed.",
                  fg='red').pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 6:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("An iterative method that uses"
                                                " a switching function to "
                                                "classify voxels as either "
                                                "foreground or background. "
                                                "The first iteration defines"
                                                " the switch based on the "
                                                "assumption that the image "
                                                "corners are background and "
                                                "the rest is foreground. The"
                                                " binarized image that results"
                                                " is then used to define the "
                                                "switching function in the "
                                                "next iteration. Performs "
                                                "poorly when objects are in"
                                                " close proximity."))
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = r"http://dx.doi.org/10.1109/TSMC.1978.4310039"
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed.",
                  fg='red').pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 7:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("This method approximates the "
                                                "image histogram as two "
                                                "Gaussian distributions estim"
                                                "ating the histogram above and"
                                                " below the current threshold "
                                                "level. The threshold that "
                                                "produces minimal overlap "
                                                "between the Gaussian fits "
                                                "is taken. Performs poorly "
                                                "when objects are in close "
                                                "proximity."))
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = (r"http://dx.doi.org/"
                                  "10.1016/0031-3203(86)90030-0")
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed.",
                  fg='red').pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 8:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("This method calculates the "
                                                "moments of the gray image and"
                                                " then determines the thresh"
                                                "old level that yields a thres"
                                                "holded image that has the "
                                                "same moments. Performs poorly"
                                                " when objects are in close "
                                                "proximity."))
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = (r"http://dx.doi.org/"
                                  "10.1016/0734-189X(85)90133-1")
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed.",
                  fg='red').pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 9:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("Performs a multilevel thres"
                                                "hold that minimizes a cost "
                                                "function based on similarity"
                                                " between the original and "
                                                "thresholded images and the "
                                                "total number of bits needed "
                                                "to represent the thresholded "
                                                "image. The balance of these "
                                                "two terms typically has a "
                                                "unique minimum. This method "
                                                "is much less expensive than "
                                                "entropy based methods, but "
                                                "has shown poor performance "
                                                "when objects are in close "
                                                "proximity."))
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = r"http://dx.doi.org/10.1109/83.366472"
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed.",
                  fg='red').pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 10:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("The same as the Maximum "
                                                "Entropy measure, but uses "
                                                "the Renyi entropy definition."
                                                " In practice this sets the "
                                                "threshold value lower than "
                                                "the Maximum Entropy approach,"
                                                " so is more likely to capture"
                                                " faint voxels. This can "
                                                "cause problems when objects"
                                                " are close."))
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = (r"http://dx.doi.org/"
                                  "10.1016/0734-189X(85)90125-2")
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed.",
                  fg='red').pack(anchor=W)
        elif self.intSettings['thresholdMethod'].get() == 11:
            self.textThresholdHelp["state"] = NORMAL
            self.textThresholdHelp.delete("0.0", END)
            self.textThresholdHelp.insert(END, ("A modification of the Maximum"
                                                " Entropy method. This method "
                                                "additionally considers the "
                                                "voxel's 'distance' from the "
                                                "determined threshold in its "
                                                "analytics. This results in a "
                                                "more aggressive thresholding "
                                                "with fewer faint voxels "
                                                "classified as white. For "
                                                "microscopy images, voxels "
                                                "with partial volume effects "
                                                "are more likely to not be "
                                                "considered an object with "
                                                "this approach. This will "
                                                "perform better still than "
                                                "Maximum Entropy when objects"
                                                " are close."))
            self.textThresholdHelp["state"] = DISABLED
            self.thresholdLink = r"http://dx.doi.org/10.1006/cgip.1994.1037"
            Label(self.thresholdSettingsFrame,
                  text="     No additional settings needed.",
                  fg='red').pack(anchor=W)

    def populateActiveContourSettings(self):
        for child in self.activeSettingsFrame.pack_slaves():
            child.destroy()
        if self.intSettings['activeMethod'].get() == 1:
            Label(self.activeSettingsFrame,
                  text="No additional settings needed.").pack(anchor=W)
            self.textActiveHelp["state"] = NORMAL
            self.textActiveHelp.delete("0.0", END)
            self.textActiveHelp.insert(END, ("Only a threshold-based "
                                             "segmentation will "
                                             "be performed."))
            self.textActiveHelp["state"] = DISABLED
            self.activeReference["text"] = ""
            self.activeLink = ""
        elif self.intSettings['activeMethod'].get() == 2:
            self.textActiveHelp["state"] = NORMAL
            self.textActiveHelp.delete("0.0", END)
            self.textActiveHelp.insert(END,
                                       ("Segments object using a geodesic "
                                        "active contour levelset algorithm. "
                                        "Uses the Canny edge detector; Canny"
                                        "Variance governs degree of Gaussian"
                                        " smoothing of edge detector. Upper"
                                        "Threshold will guarantee detection "
                                        "of edges with intensity greater than"
                                        " setting. LowerThreshold will disca"
                                        "rd intensities lower than setting. "
                                        "Propagation is the contour inflation"
                                        " force (deflation if negative); "
                                        "higher values result in skipping "
                                        "weaker edges or small islands of "
                                        "contrast change. Curvature governs "
                                        "how smooth the contour will be; "
                                        "higher values result in smoother. "
                                        "Advection causes the contour to "
                                        "attract to edges; higher values "
                                        "help prevent leakage. Iterations "
                                        "are the maximum iterations to take."
                                        " Maximum RMS Error Change is the "
                                        "change in RMS at which iterations "
                                        "will terminate if Iterations has "
                                        "not yet been reached."))
            self.textActiveHelp.pack(anchor=NW)
            self.textActiveHelp["state"] = DISABLED
            self.activeReference["text"] = "Reference"
            self.activeLink = r"http://dx.doi.org/10.1023/A:1007979827043"
            settings = [('CannyVariance', 'geodesicCannyVariance'),
                        ('UpperThreshold', 'geodesicCannyUpper'),
                        ('LowerThreshold', 'geodesicCannyLower'),
                        ('Propagation', 'geodesicPropagation'),
                        ('Curvature', 'geodesicCurvature'),
                        ('Advection', 'geodesicAdvection'),
                        ('Iterations', 'geodesicIterations'),
                        ('Maximum RMS Error Change', 'geodesicRMS')]
            for t, v in settings:
                Label(self.activeSettingsFrame, text=t).pack(anchor=W)
                Entry(self.activeSettingsFrame,
                      textvariable=self.settings[v]).pack(anchor=W)
        else:
            self.textActiveHelp["state"] = NORMAL
            self.textActiveHelp.delete("0.0", END)
            self.textActiveHelp.insert(END, ("An active contour model that "
                                             "requires no edge information. "
                                             "This algorithm employs a convex "
                                             "objective function and is "
                                             "therefore, very robust. "
                                             "Unfortunately, there is only "
                                             "a single phase implementation "
                                             "in SimpleITK, so this tends to "
                                             "have trouble with objects in "
                                             "close proximity. If a multiphase"
                                             " version is released in the "
                                             "future, this will be the ideal "
                                             "approach to segment close or "
                                             "touching objects. The contour "
                                             "is evolved iteratively using "
                                             "curvature flow. Lambda1 and "
                                             "Lambda2 are energy term weights "
                                             "for voxels inside and outside "
                                             "the contour, respectively. A "
                                             "strategy to help resolve close "
                                             "objects is to slightly increase "
                                             "Lambda1, penalizing voxels "
                                             "inside the contour."))
            self.textActiveHelp.pack(anchor=NW)
            self.textActiveHelp["state"] = DISABLED
            self.activeReference["text"] = "Reference"
            self.activeLink = r"http://dx.doi.org/10.1109/83.902291"
            settings = [('Lambda1 (internal weight)', 'edgeLambda1'),
                        ('Lambda2 (external weight)', 'edgeLambda2'),
                        ('Curvature Weight', 'edgeCurvature'),
                        ('Iterations', 'edgeIterations')]
            for t, v in settings:
                Label(self.activeSettingsFrame, text=t).pack(anchor=W)
                Entry(self.activeSettingsFrame,
                      textvariable=self.settings[v]).pack(anchor=W)

    def populateDeformableSettings(self):
        for child in self.deformableSettingsFrame.pack_slaves():
            child.destroy()
        if self.intSettings['defReg'].get() == 1:
            settings = [
                ('Displacement Field Smoothing Variance', 'deformableSigma'),
                ('Maximum RMS error', 'deformableRMS'),
                ('Maximum Iterations', 'deformableIterations'),
                ('Precision', 'deformablePrecision')]
            for t, v in settings:
                Label(self.deformableSettingsFrame, text=t).pack(anchor=W)
                Entry(self.deformableSettingsFrame,
                      textvariable=self.settings[v]).pack(anchor=W)

    def add_directory(self):
        dir_name = tkFileDialog.askdirectory(
            parent=root,
            initialdir=self.lastdir,
            title="Select directory containing images.")
        self.lastdir = dir_name
        self.directories.append(dir_name)
        self.listDirectories.insert(END, dir_name)

    def remove_directory(self):
        index = self.listDirectories.curselection()
        if index:
            for i in index[::-1]:
                self.listDirectories.delete(i)
                del self.directories[i]

    def add_material_dir(self):
        dir_name = tkFileDialog.askdirectory(
            parent=root,
            initialdir=self.lastdir,
            title="Select directory containing reference configuration data.")
        if dir_name:
            self.lastdir = dir_name
            self.materialDirectory = dir_name
            self.materialDirectoryLabel["text"] = dir_name

    def add_spatial_dir(self):
        dir_name = tkFileDialog.askdirectory(
            parent=root,
            initialdir=self.lastdir,
            title="Select directory containing reference configuration data.")
        if dir_name:
            self.lastdir = dir_name
            self.spatialDirectories.append(dir_name)
            self.listSpatialDirectories.insert(END, dir_name)

    def remove_spatial_dir(self):
        index = self.listSpatialDirectories.curselection()
        if index:
            for i in index[::-1]:
                self.listSpatialDirectories.delete(i)
                del self.spatialDirectories[i]

    def saveSettings(self):
        filename = tkFileDialog.asksaveasfilename(defaultextension=".pkl")
        if filename:
            fid = open(filename, 'wb')
            tmp_settings = copy.copy(self.settings)
            for key in self.settings.keys():
                tmp_settings[key] = self.settings[key].get()
            tmp_int_settings = copy.copy(self.intSettings)
            for key in self.intSettings.keys():
                tmp_int_settings[key] = self.intSettings[key].get()
            values = {"Settings": tmp_settings,
                      "ButtonStates": tmp_int_settings}
            pickle.dump(values, fid)
            fid.close()

    def loadSettings(self):
        filename = tkFileDialog.askopenfilename(
            parent=root,
            initialdir=os.getcwd(),
            title="Select a saved settings file.")
        if filename:
            fid = open(filename, 'rb')
            values = pickle.load(fid)
            fid.close()
            for key in self.settings:
                self.settings[key].set(values['Settings'][key])
            for key in self.intSettings:
                self.intSettings[key].set(values['ButtonStates'][key])
        self.populateSmoothingSettings()
        self.populateThresholdSettings()
        self.populateActiveContourSettings()

    def loadROI(self):
        filename = tkFileDialog.askopenfilename(
            parent=root,
            initialdir=os.getcwd(),
            title="Select an .xls file containing Regions of Interest.")
        if '.xls' in filename:
            wb = xlrd.open_workbook(filename)
            N = wb.nsheets
            #clear any previously loaded regions
            self.ROI = []
            for i in xrange(N):
                self.ROI.append([])
                s = wb.sheet_by_index(i)
                #skip the first row
                for r in xrange(s.nrows - 1):
                    v = s.row_values(r + 1, start_colx=1)
                    #even row
                    if r % 2 == 0:
                        tmp = [0] * 6
                        tmp[0] = int(v[0])
                        tmp[1] = int(v[1])
                        tmp[2] = int(v[4])
                        tmp[3] = int(v[2])
                        tmp[4] = int(v[3])
                    else:
                        tmp[5] = int(v[4]) - tmp[2]
                        self.ROI[i].append(tmp)
        else:
            print(("{:s} does not have proper extension. Currently supporting"
                   " only .xls filetypes.").format(filename))

    def run_segmentation(self):
        if not self.directories:
            print(("WARNING: no directories have been indicated; "
                  "nothing has been done."))
            return
        if not self.ROI:
            print(("WARNING: no region of interest loaded; "
                   "assuming only 1 cell in the image."))
        # translate smoothing parameters to pyCellAnalyst dictionary syntax
        if self.intSettings['smoothingMethod'].get() == 1:
            smoothingParameters = {}
        elif self.intSettings['smoothingMethod'].get() == 2:
            smoothingParameters = {
                'radius': (self.settings['medianRadius'].get(),) * 3}
        elif self.intSettings['smoothingMethod'].get() == 3:
            smoothingParameters = {
                'sigma': self.settings['gaussianSigma'].get()}
        elif self.intSettings['smoothingMethod'].get() == 4:
            smoothingParameters = {
                'iterations': self.settings['curvatureIterations'].get(),
                'conductance': self.settings['curvatureConductance'].get()}
        elif self.intSettings['smoothingMethod'].get() == 5:
            smoothingParameters = {
                'iterations': self.settings['gradientIterations'].get(),
                'conductance': self.settings['gradientConductance'].get(),
                'time step': self.settings['gradientTimeStep'].get()}
        elif self.intSettings['smoothingMethod'].get() == 6:
            smoothingParameters = {
                'domainSigma': self.settings['bilateralDomainSigma'].get(),
                'rangeSigma': self.settings['bilateralRangeSigma'].get(),
                'samples': self.settings['bilateralSamples'].get()}
        elif self.intSettings['smoothingMethod'].get() == 7:
            noiseModel = ['no model', 'poisson', 'gaussian', 'rician']
            smoothingParameters = {
                'radius': self.settings['patchRadius'].get(),
                'iterations': self.settings['patchIterations'].get(),
                'patches': self.settings['patchNumber'].get(),
                'noise model': noiseModel[
                    self.settings['patchNoiseModel'].get() - 1]}

        objects = ['Foreground', 'Background']
        for i, d in enumerate(self.directories):
            try:
                regions = self.ROI[i]
            except:
                regions = None
            vol = Volume(d,
                         pixel_dim=[self.settings['xdim'].get(),
                                    self.settings['ydim'].get(),
                                    self.settings['zdim'].get()],
                         regions=regions,
                         segmentation='User',
                         handle_overlap=self.intSettings[
                             'handleOverlap'].get(),
                         display=self.intSettings['display'].get(),
                         stain=objects[self.intSettings['stain'].get()],
                         two_dim=self.intSettings['processAs2D'].get(),
                         bright=self.intSettings['removeBright'].get(),
                         enhance_edge=self.intSettings[
                             'edgeEnhancement'].get(),
                         depth_adjust=self.intSettings['depth_adjust'].get(),
                         smoothing_method=self.smoothingMethods[
                             self.intSettings['smoothingMethod'].get() - 1],
                         debug=self.intSettings['debug'].get(),
                         fillholes=self.intSettings['fillHoles'].get(),
                         opening=self.intSettings['opening'].get(),
                         smoothing_parameters=smoothingParameters)

            if self.intSettings['activeMethod'].get() == 1:
                vol.thresholdSegmentation(
                    method=self.thresholdMethods[self.intSettings[
                        'thresholdMethod'].get() - 1],
                    ratio=self.settings['thresholdPercentage'].get(),
                    adaptive=self.intSettings['thresholdAdaptive'].get())
            elif self.intSettings['activeMethod'].get() == 2:
                vol.geodesicSegmentation(
                    upsampling=int(self.settings['upsampling'].get()),
                    seed_method=self.thresholdMethods[self.intSettings[
                        'thresholdMethod'].get() - 1],
                    adaptive=self.intSettings['thresholdAdaptive'].get(),
                    ratio=self.settings['thresholdPercentage'].get(),
                    canny_variance=(self.settings['geodesicCannyVariance']
                                    .get(),) * 3,
                    cannyUpper=self.settings['geodesicCannyUpper'].get(),
                    cannyLower=self.settings['geodesicCannyLower'].get(),
                    propagation=self.settings['geodesicPropagation'].get(),
                    curvature=self.settings['geodesicCurvature'].get(),
                    advection=self.settings['geodesicAdvection'].get(),
                    active_iterations=self.settings[
                        'geodesicIterations'].get())
            elif self.intSettings['activeMethod'].get() == 3:
                vol.edgeFreeSegmentation(
                    upsampling=int(self.settings['upsampling'].get()),
                    seed_method=self.thresholdMethods[self.intSettings[
                        'thresholdMethod'].get() - 1],
                    adaptive=self.intSettings['thresholdAdaptive'].get(),
                    ratio=self.settings['thresholdPercentage'].get(),
                    lambda1=self.settings['edgeLambda1'].get(),
                    lambda2=self.settings['edgeLambda2'].get(),
                    curvature=self.settings['edgeCurvature'].get(),
                    iterations=self.settings['edgeIterations'].get())
            vol.writeLabels()
            vol.writeSurfaces()

    def run_kinematics(self):
        #timestamp
        ts = datetime.datetime.fromtimestamp(
            time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        pardir = os.path.dirname(self.spatialDirectories[0])
        #uniform strain
        ofid = open(pardir + os.sep + "Kinematics_Analysis" + ts + ".csv", 'w')
        #ellipsoidal approximation
        efid = open(pardir + os.sep + "Ellipsoidal_Analysis" + ts
                    + ".csv", 'w')
        if self.intSettings["makePlots"].get():
            self.results = OrderedDict()
            self.results["Tissue"] = OrderedDict()
            self.aggregate = OrderedDict()
        for i, d in enumerate(self.spatialDirectories):
            shortname = d.split(os.sep)[-1]
            try:
                shortname = shortname.replace('_results', '')
            except:
                pass
            mech = CellMech(
                ref_dir=self.materialDirectory, def_dir=d,
                rigidInitial=self.intSettings['rigidInitial']
                .get(),
                deformable=self.intSettings['defReg'].get(),
                saveFEA=self.intSettings['saveFEA'].get(),
                deformableSettings={
                    'Iterations': self.settings[
                        'deformableIterations'].get(),
                    'Maximum RMS': self.settings[
                        'deformableRMS'].get(),
                    'Displacement Smoothing': self.settings[
                        'deformableSigma'].get(),
                    'Precision': self.settings[
                        'deformablePrecision'].get()},
                display=self.intSettings['deformableDisplay'].get())

            ofid.write(d + '\n')
            ofid.write(("Object ID, E11, E22, E33, E12, E13, E23, "
                        "Volumetric, Effective, Maximum Tensile, "
                        "Maximum Compressive, Maximum Shear\n"))
            if np.any(mech.ecm_strain):
                N = len(mech.ecm_strain)
                ecm_w, ecm_v = np.linalg.eigh(mech.ecm_strain)
                ecm_w = np.sort(ecm_w)
                tissue_tensile = ecm_w[2]
                tissue_compressive = ecm_w[0]
                tissue_shear = 0.5 * np.abs(ecm_w[2] - ecm_w[0])
                tissue_vol = np.linalg.det(mech.ecm_strain) - 1.0
                ofid.write(("Tissue, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, "
                            "{:f}, {:f}, {:f}, {:f}, {:f}\n")
                           .format(mech.ecm_strain[0, 0],
                                   mech.ecm_strain[1, 1],
                                   mech.ecm_strain[2, 2],
                                   mech.ecm_strain[0, 1],
                                   mech.ecm_strain[0, 2],
                                   mech.ecm_strain[1, 2],
                                   np.linalg.det(mech.ecm_strain) - 1.0,
                                   np.sqrt((ecm_w[2] - ecm_w[1]) ** 2 +
                                           (ecm_w[1] - ecm_w[0]) ** 2 +
                                           (ecm_w[2] - ecm_w[0]) ** 2),
                                   tissue_tensile,
                                   tissue_compressive,
                                   tissue_shear))
                if self.intSettings['makePlots'].get():
                    self.results["Tissue"][shortname] = OrderedDict([
                        ("Max Compression", tissue_compressive),
                        ("Max Tension", tissue_tensile),
                        ("Max Shear", tissue_shear),
                        ("Volume", tissue_vol)])

            N = len(mech.cell_strains)
            if i == 0 and self.intSettings["makePlots"].get():
                for j in xrange(N):
                    self.results[
                        "Cell {:d}".format(j + 1)] = OrderedDict()
            if self.intSettings["makePlots"].get():
                self.aggregate[shortname] = OrderedDict([
                    ("Max Compression", np.zeros(N, float)),
                    ("Max Tension", np.zeros(N, float)),
                    ("Max Shear", np.zeros(N, float)),
                    ("Volume", np.zeros(N, float)),
                    ("Height", np.zeros(N, float)),
                    ("Width", np.zeros(N, float)),
                    ("Depth", np.zeros(N, float))])

            for j, c in enumerate(mech.cell_strains):
                w, v = np.linalg.eigh(c)
                w = np.sort(w)
                ofid.write(("Cell {:d}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, "
                            "{:f}, {:f}, {:f}, {:f}, {:f}\n")
                           .format(j + 1,
                                   c[0, 0],
                                   c[1, 1],
                                   c[2, 2],
                                   c[0, 1],
                                   c[0, 2],
                                   c[1, 2],
                                   mech.vstrains[j],
                                   np.sqrt((w[2] - w[1]) ** 2 +
                                           (w[1] - w[0]) ** 2 +
                                           (w[2] - w[0]) ** 2),
                                   w[2],
                                   w[0],
                                   0.5 * np.abs(w[2] - w[0])))
                if self.intSettings["makePlots"].get():
                    key = "Cell {:d}".format(j + 1)
                    self.results[key][shortname] = OrderedDict([
                        ("Max Compression", w[0]),
                        ("Max Tension", w[2]),
                        ("Max Shear", 0.5 * np.abs(w[2] - w[0])),
                        ("Volume", mech.vstrains[j]),
                        ("Height", None),
                        ("Width", None),
                        ("Depth", None)])
                    self.aggregate[shortname]['Max Compression'][j] = w[0]
                    self.aggregate[shortname]['Max Tension'][j] = w[2]
                    self.aggregate[shortname]['Max Shear'][j] = 0.5 * np.abs(
                        w[2] - w[0])
                    self.aggregate[shortname]['Volume'][j] = mech.vstrains[j]

            efid.write(d + '\n')
            efid.write(("Object ID, Reference Major Axis, Reference Middle "
                        "Axis, Reference Minor Axis, Deformed Major Axis, "
                        "Deformed Middle Axis, Deformed Minor Axis, Reference"
                        " Volume, Deformed Volume\n"))

            for j, (rvol, dvol, raxes, daxes) in enumerate(
                    zip(mech.rvols, mech.dvols, mech.raxes, mech.daxes)):
                raxes = np.sort(raxes)
                daxes = np.sort(daxes)
                efid.write(("Cell {:d}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f},"
                            " {:f}, {:f}\n").format(j + 1,
                                                    raxes[2],
                                                    raxes[1],
                                                    raxes[0],
                                                    daxes[2],
                                                    daxes[1],
                                                    daxes[0],
                                                    rvol,
                                                    dvol))

                if self.intSettings['makePlots'].get():
                    key = "Cell {:d}".format(j + 1)
                    self.results[key][shortname]["Height"] = (daxes[0] /
                                                              raxes[0] - 1)
                    self.results[key][shortname]["Depth"] = (daxes[1] /
                                                             raxes[1] - 1)
                    self.results[key][shortname]["Width"] = (daxes[2] /
                                                             raxes[2] - 1)
                    self.aggregate[shortname]["Height"][j] = (daxes[0] /
                                                              raxes[0] - 1)
                    self.aggregate[shortname]["Depth"][j] = (daxes[1] /
                                                             raxes[1] - 1)
                    self.aggregate[shortname]["Width"][j] = (daxes[2] /
                                                             raxes[2] - 1)

        if self.intSettings['makePlots'].get():
            self.makePlots(ts)

        ofid.close()
        efid.close()

    def makePlots(self, ts):
        pardir = os.path.dirname(self.spatialDirectories[0])
        #boxplots of all cells for different cases
        N = len(self.aggregate.keys())
        M = len(self.aggregate.values()[0]["Height"])
        for m in self.aggregate.values()[0].keys():
            data = np.zeros((M, N), float)
            for j, k in enumerate(self.aggregate.keys()):
                data[:, j] = self.aggregate[k][m]
            fig, ax = plt.subplots()
            ax.boxplot(data)
            ax.set_xticklabels(self.aggregate.keys())
            ax.set_ylabel(m + " Strain")
            fig.savefig(string.join([pardir, os.sep, "AllCells_",
                                     m, "_", ts, ".svg"], ''))

        for k in self.results.keys():
            if not(self.results[k].keys()):
                continue
            fig, ax = plt.subplots()
            fig.set_size_inches([3.34646, 3.34646])
            N = len(self.results[k].keys())
            ind = np.arange(4)
            width = 0.8 / float(N)
            cm = plt.get_cmap('jet')
            colors = plt.cm.Set3(np.linspace(0, 1, N))
            plt.rcParams.update({'font.size': 8})
            rects = []
            # affine transformation approach
            for j, case in enumerate(self.results[k].keys()):
                dat = np.array([self.results[k][case]["Max Compression"],
                                self.results[k][case]["Max Tension"],
                                self.results[k][case]["Max Shear"],
                                self.results[k][case]["Volume"]]).ravel()
                rects.append(ax.bar(ind + j * width, dat, width,
                                    color=colors[j], label=case))
            ax.set_ylabel('Green-Lagrange Strain')
            ax.set_title(k)
            ax.set_xticks(np.arange(4) + 0.4)
            ax.set_xticklabels(['Compression', 'Tension', 'Shear', 'Volume'])
            ax.axhline(y=0.0, color='k')
            fig.savefig(string.join([pardir, os.sep, "Kinematics_",
                                     k, "_", ts, ".svg"], ''))
            # ellipsoidal approach
            fig2, ax2 = plt.subplots()
            fig2.set_size_inches([3.34646, 3.34646])
            rects = []
            for j, case in enumerate(self.results[k].keys()):
                if k == "Tissue":
                    continue
                dat = np.array([self.results[k][case]["Height"],
                                self.results[k][case]["Width"],
                                self.results[k][case]["Depth"],
                                self.results[k][case]["Volume"]]).ravel()
                rects.append(ax2.bar(ind + j * width, dat, width,
                                     color=colors[j], label=case))
            ax2.set_ylabel('Green-Lagrange Strain')
            ax2.set_title(k)
            ax2.set_xticks(np.arange(4) + 0.4)
            ax2.set_xticklabels(['Height', 'Width', 'Depth', 'Volume'])
            ax2.axhline(y=0.0, color='k')
            fig2.savefig(string.join([pardir, os.sep, "Ellipsoidal_",
                                      k, "_", ts, ".svg"], ''))
            plt.close('all')
        figLegend1 = plt.figure()
        plt.figlegend(*ax.get_legend_handles_labels(), loc='upper left')
        figLegend1.savefig(string.join(
            [pardir, os.sep, "Kinematics_Legend_", ts, ".svg"], ''))
        figLegend2 = plt.figure()
        plt.figlegend(*ax.get_legend_handles_labels(), loc='upper left')
        figLegend2.savefig(string.join(
            [pardir, os.sep, "Ellipsoidal_Legend_", ts, ".svg"], ''))
        plt.close('all')

    def open_smoothing_reference(self, *args):
        webbrowser.open_new(self.smoothingLink)

    def open_threshold_reference(self, *args):
        webbrowser.open_new(self.thresholdLink)

    def open_active_reference(self, *args):
        webbrowser.open_new(self.activeLink)

    def update(self, *args):
        """dummy function to use trace feature"""
        pass

root = Tk()
root.title("Welcome to the pyCellAnalyst segmentation GUI.")
app = Application(root)

root.mainloop()
