import febio, pickle, string, subprocess, os, tkFileDialog, platform, fnmatch, itertools
import time, datetime, vtk
from vtk.util import numpy_support
from Tkinter import *
from collections import OrderedDict
from FebPlt import FebPlt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import histogram
from weighted import quantile_1D

class Application(Frame):
    def __init__(self,master):
        Frame.__init__(self,master)
        self.lastdir = os.getcwd()
        self.op_sys = platform.platform()

        self.pickles = []
        self.data = {}

        self.volumes = {}
        self.outputs = OrderedDict([('Effective Strain (von Mises)', IntVar(value=1)),
                        ('Maximum Compressive Strain',   IntVar(value=1)),
                        ('Maximum Tensile Strain',       IntVar(value=1)),
                        ('Maximum Shear Strain',         IntVar(value=1)),
                        ('Volumetric Strain',            IntVar(value=1)),
                        ('Effective Stress (von Mises)', IntVar(value=0)),
                        ('Maximum Compressive Stress',   IntVar(value=0)),
                        ('Maximum Tensile Stress',       IntVar(value=0)),
                        ('Maximum Shear Stress',         IntVar(value=0)),
                        ('Pressure',                     IntVar(value=0))])

        self.analysis = OrderedDict([('Generate Histograms',   IntVar(value=1)),
                                     ('Tukey Boxplots',        IntVar(value=1)),
                                     ('Calculate Differences', IntVar(value=1)),
                                     ('Convert to VTK',        IntVar(value=1))])


        self.results = {'Effective Strain (von Mises)': {},
                        'Maximum Compressive Strain':   {},
                        'Maximum Tensile Strain':       {},
                        'Maximum Shear Strain':         {},
                        'Volumetric Strain':            {},
                        'Effective Stress (von Mises)': {},
                        'Maximum Compressive Stress':   {},
                        'Maximum Tensile Stress':       {},
                        'Maximum Shear Stress':         {},
                        'Pressure':                     {}}

        self.histograms = {'Effective Strain (von Mises)': {},
                           'Maximum Compressive Strain':   {},
                           'Maximum Tensile Strain':       {},
                           'Maximum Shear Strain':         {},
                           'Volumetric Strain':            {},
                           'Effective Stress (von Mises)': {},
                           'Maximum Compressive Stress':   {},
                           'Maximum Tensile Stress':       {},
                           'Maximum Shear Stress':         {},
                           'Pressure':                     {}}
        
        self.boxwhiskers = {'Effective Strain (von Mises)': {},
                            'Maximum Compressive Strain':   {},
                            'Maximum Tensile Strain':       {},
                            'Maximum Shear Strain':         {},
                            'Volumetric Strain':            {},
                            'Effective Stress (von Mises)': {},
                            'Maximum Compressive Stress':   {},
                            'Maximum Tensile Stress':       {},
                            'Maximum Shear Stress':         {},
                            'Pressure':                     {}}

        self.differences = {'Effective Strain (von Mises)': {},
                            'Maximum Compressive Strain':   {},
                            'Maximum Tensile Strain':       {},
                            'Maximum Shear Strain':         {},
                            'Volumetric Strain':            {},
                            'Effective Stress (von Mises)': {},
                            'Maximum Compressive Stress':   {},
                            'Maximum Tensile Stress':       {},
                            'Maximum Shear Stress':         {},
                            'Pressure':                     {}}

        self.grid()
        self.createWidgets()

    def createWidgets(self):
        #pickles to load
        self.pickleFrame = LabelFrame(self,text="Pickles from which to generate models")
        self.pickleFrame.grid(row=0,column=0,columnspan=3,rowspan=2,padx=5,pady=5,sticky=NW)
        self.buttonAddFile = Button(self.pickleFrame,text="Add",bg='green',command=self.addPickle)
        self.buttonAddFile.grid(row=0,column=0,padx=5,pady=5,sticky=E+W)
        self.buttonRemoveFile = Button(self.pickleFrame,text="Remove",bg='red',command=self.removePickle)
        self.buttonRemoveFile.grid(row=1,column=0,padx=5,pady=5,sticky=E+W)
        self.listPickles = Listbox(self.pickleFrame,width=80,selectmode=MULTIPLE)
        self.listPickles.grid(row=0,column=1,columnspan=2,rowspan=2,padx=5,pady=5,sticky=E+W)

        #outputs
        self.outputsFrame = LabelFrame(self,text="Outputs")
        self.outputsFrame.grid(row=2,column=0,rowspan=4,columnspan=2,padx=5,pady=5,sticky=NW)
        for i,t in enumerate(self.outputs.keys()):
            if i<5:
                Checkbutton(self.outputsFrame,text=t,variable=self.outputs[t]).grid(row=i,column=0,padx=5,pady=5,sticky=NW)
            else:
                Checkbutton(self.outputsFrame,text=t,variable=self.outputs[t]).grid(row=i-5,column=1,padx=5,pady=5,sticky=NW)

        #analysis settings
        self.analysisSettingsFrame = LabelFrame(self,text="Analysis Options")
        self.analysisSettingsFrame.grid(row=2,column=2,padx=5,pady=5,sticky=E+W)
        for i,t in enumerate(self.analysis.keys()):
            Checkbutton(self.analysisSettingsFrame,text=t,variable=self.analysis[t]).grid(row=i,column=2,padx=5,pady=5,sticky=NW)

        #run analysis
        self.buttonExecute = Button(self,bg='green',font=('Helvetica','20','bold'),text='Perform Simulation',command=self.performFEA)
        self.buttonExecute.grid(row=7,column=0,columnspan=3,padx=5,pady=5,sticky=E+W)

        self.buttonAnalyze = Button(self,bg='blue',font=('Helvetica','20','bold'),text='Analyze Results',command=self.analyzeResults)
        self.buttonAnalyze.grid(row=8,column=0,columnspan=3,padx=5,pady=5,sticky=E+W)

    def addPickle(self):
        filenames = tkFileDialog.askopenfilenames(parent=root,initialdir=self.lastdir,title="Please choose pickle file(s) for analsis.")
        filenames = root.tk.splitlist(filenames)
        for f in filenames:
            if '.pkl' in f:
                self.listPickles.insert(END,f)
                self.pickles.append(f)
            else:
                print("WARNING - {:s} does not contain the .pkl extension. Ignoring this file.")
        

    def removePickle(self):
        pass

    def performFEA(self):
        for filename in self.pickles:
            fid = open(filename,'rb')
            self.data[filename] = pickle.load(fid)
            fid.close()

            mesh = febio.MeshDef()
            for i,e in enumerate(self.data[filename]['elements']):
                mesh.elements.append(['tet4',i+1]+e)

            for i,n in enumerate(self.data[filename]['nodes']):
                mesh.nodes.append([i+1]+n)

            mesh.addElementSet(setname='cell',eids=range(1,len(self.data[filename]['elements'])))

            modelname = string.replace(filename,'.pkl','.feb')
            model = febio.Model(modelfile=modelname,steps=[{'Displace':'solid'}])

            mat = febio.MatDef(matid=1,mname='cell',mtype='neo-Hookean',elsets='cell',
                               attributes={'density':'0.001','E':'1.0','v':'0.3'})

            model.addMaterial(mat)

            model.addGeometry(mesh=mesh,mats=[mat])

            ctrl = febio.Control()
            ctrl.setAttributes({'title': 'cell'})

            model.addControl(ctrl,step=0)

            boundary = febio.Boundary(steps=1)
            for i,bc in enumerate(self.data[filename]['boundary conditions']):
                boundary.addPrescribed(step=0,nodeid=self.data[filename]['surfaces'][i],dof='x',lc='1',scale=str(bc[0]))
                boundary.addPrescribed(step=0,nodeid=self.data[filename]['surfaces'][i],dof='y',lc='1',scale=str(bc[1]))
                boundary.addPrescribed(step=0,nodeid=self.data[filename]['surfaces'][i],dof='z',lc='1',scale=str(bc[2]))

            model.addBoundary(boundary=boundary)
            model.addLoadCurve(lc='1',lctype='linear',points=[0,0,1,1])

            model.writeModel()

            subprocess.call("/home/scott/febio-2.0/bin/febio2.lnx64 -i "+modelname,shell=True)

    def analyzeResults(self):
        self.matched = [] # for paired comparisons
        for i,f in enumerate(self.pickles):
            if "Windows" in self.op_sys:
                s = f.split("\\")[-1]
            else:
                s = f.split("/")[-1]
            if not(fnmatch.filter(itertools.chain.from_iterable(self.matched),"*"+s)):
                self.matched.append(fnmatch.filter(self.pickles,"*"+s))
            plotname = string.replace(f,'.pkl','.xplt')

            results = FebPlt(plotname)
            stress = np.zeros((len(self.data[f]['elements']),3,3),float)
            strain = np.copy(stress)
            mvolumes = np.zeros(len(self.data[f]['elements']),float) #material element volumes
            svolumes = np.copy(mvolumes) #spatial element volumes
            for j,e in enumerate(self.data[f]['elements']):
                tmp = results.ElementData[j+1]['stress'][-1,:]
                stress[j,:,:] = [[tmp[0], tmp[3], tmp[5]],
                                 [tmp[3], tmp[1], tmp[4]],
                                 [tmp[5], tmp[4], tmp[2]]]
                X = np.zeros((4,3),float) #material coordinates
                x = np.zeros((4,3),float) #spatial coordinates
                for k in xrange(4):
                    X[k,:] = self.data[f]['nodes'][e[k]-1]
                    x[k,:] = X[k,:]+results.NodeData[e[k]]['displacement'][-1,:]
                #set up tangent space
                W = np.zeros((6,3),float)
                w = np.zeros((6,3),float)
                for k,c in enumerate([(0,1),(0,2),(0,3),(1,3),(2,3),(1,2)]):
                    W[k,:] = X[c[1],:] - X[c[0],:]
                    w[k,:] = x[c[1],:] - x[c[0],:]
                dX = np.zeros((6,6),float)
                ds = np.zeros((6,1),float)
                for k in xrange(6):
                    for l in xrange(3):
                        dX[k,l] = 2*W[k,l]**2
                    dX[k,3] = 4*W[k,0]*W[k,1]
                    dX[k,4] = 4*W[k,1]*W[k,2]
                    dX[k,5] = 4*W[k,0]*W[k,2]
                    ds[k,0] = np.linalg.norm(w[k,:])**2 - np.linalg.norm(W[k,:])**2
                E = np.linalg.solve(dX,ds) #solve for strain
                #get volumes
                mvolumes[j] = np.abs(np.dot(W[0,:],np.cross(W[1,:],W[2,:])))/6.0
                svolumes[j] = np.abs(np.dot(w[0,:],np.cross(w[1,:],w[2,:])))/6.0
                strain[j,:,:] = [[E[0], E[3], E[5]],
                                 [E[3], E[1], E[4]],
                                 [E[5], E[4], E[2]]]
            #eigenvalues for principals - vectorized over fist dimension
            pstress = np.linalg.eigvals(stress)
            pstrain = np.linalg.eigvals(strain)

            pstress = np.sort(pstress,axis=1)
            pstrain = np.sort(pstrain,axis=1)

            self.volumes.update({f:mvolumes}) #save reference volumes
            
            self.results['Effective Strain (von Mises)'].update({f:np.sqrt(((pstrain[:,2]-pstrain[:,1])**2+(pstrain[:,1]-pstrain[:,0])**2+(pstrain[:,2]-pstrain[:,0])**2)/2.0)})
            self.results['Maximum Compressive Strain'].update({f:pstrain[:,0]})
            self.results['Maximum Tensile Strain'].update({f:pstrain[:,2]})
            self.results['Maximum Shear Strain'].update({f:0.5*(pstrain[:,2] - pstrain[:,0])})
            self.results['Volumetric Strain'].update({f:svolumes/mvolumes - 1.0})

            self.results['Effective Stress (von Mises)'].update({f:np.sqrt(((pstress[:,2]-pstress[:,1])**2+(pstress[:,1]-pstress[:,0])**2+(pstress[:,2]-pstress[:,0])**2)/2.0)})
            self.results['Maximum Compressive Stress'].update({f:pstress[:,0]})
            self.results['Maximum Tensile Stress'].update({f:pstress[:,2]})
            self.results['Maximum Shear Stress'].update({f:0.5*(pstress[:,2] - pstress[:,0])})
            self.results['Pressure'].update({f:np.sum(pstress,axis=1)/3.0})
            
        for i,k in enumerate(self.outputs.keys()):
            if self.outputs[k].get():
                for m in self.matched:
                    weights = self.volumes[m[0]]/np.sum(self.volumes[m[0]])
                    for j,f in enumerate(m):
                        dat = np.ravel(self.results[k][f])
                        if self.analysis['Generate Histograms'].get():
                            IQR = np.subtract(*np.percentile(dat,[75,25]))
                            nbins = int(np.ptp(dat)/(2*IQR*dat.size**(-1./3.)))
                            h = histogram(dat,numbins=nbins,weights=weights)
                            bins = np.linspace(h[1],h[1]+h[2]*nbins,nbins,endpoint=False)
                            self.histograms[k][f] = {'bins': bins, 'heights': h[0], 'width': h[2]}
                        if self.analysis['Tukey Boxplots'].get():
                            quantiles = np.zeros(3,float)
                            for n,q in enumerate([0.25,0.5,0.75]):
                                quantiles[n] = quantile_1D(dat,weights,q)
                            self.boxwhiskers[k][f] = {'quantiles': quantiles, 'data': dat}
                    if self.analysis['Calculate Differences'].get():
                        for c in itertools.combinations(m,2):
                            dat1 = np.ravel(self.results[k][c[0]])
                            dat2 = np.ravel(self.results[k][c[1]])
                            difference = dat2-dat1
                            wrms = np.sqrt(np.average(difference**2,weights=weights))
                            self.differences[k][c[1]+"MINUS"+c[0]] = {'difference': difference, 'weighted RMS': wrms}
        self.saveResults()

    def saveResults(self):
        if "Windows" in self.op_sys:
            top_dir = self.pickles[0].rsplit('\\',2)[0]+'\\'
        else:
            top_dir = self.pickles[0].rsplit('/',2)[0]+'/'
        output_dir = top_dir+'FEA_analysis_' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        os.mkdir(output_dir)
        for output in self.histograms.keys():
            if self.histograms[output]:
                try:
                    os.mkdir(output_dir+'/histograms')
                except:
                    pass
                for m in self.matched:
                    plt.figure()
                    for f in m:
                        if "Windows" in self.op_sys: 
                            trunc_name = f.rsplit("\\",2)
                        else:
                            trunc_name = f.rsplit("/",2)

                        plt.bar(self.histograms[output][f]['bins'],
                                self.histograms[output][f]['heights'],
                                self.histograms[output][f]['width'],
                                label = trunc_name[1])
                    plt.legend()
                    object_name = string.replace(trunc_name[2],"cellFEA","Cell ")
                    object_name = string.replace(object_name,".pkl","")
                    plt.title(output+' '+object_name)
                    try:
                        cell_directory = output_dir+'/histograms/'+string.replace(object_name,' ','_')
                        os.mkdir(cell_directory)
                    except:
                        pass
                    plt.tight_layout()
                    plt.savefig(cell_directory+'/'+string.replace(output,' ','_')+'.svg')
                    plt.clf()

        for output in self.boxwhiskers.keys():
            if self.boxwhiskers[output]:
                try:
                    os.mkdir(output_dir+'/boxplots')
                except:
                    pass
                for m in self.matched:
                    ax = plt.subplot(111)
                    x = []
                    q = []
                    labels = []
                    for f in m:
                        if "Windows" in self.op_sys: 
                            trunc_name = f.rsplit("\\",2)
                        else:
                            trunc_name = f.rsplit("/",2)
                        x.append(self.boxwhiskers[output][f]['data'])
                        q.append(self.boxwhiskers[output][f]['quantiles'])
                        labels.append(trunc_name[1])
                    q = np.array(q)
                    a = ax.boxplot(x)
                    object_name = string.replace(trunc_name[2],"cellFEA","Cell ")
                    object_name = string.replace(object_name,".pkl","")
                    #a.title(output+' '+object_name)
                    for i in xrange(q.shape[0]):
                        a['medians'][i].set_ydata(q[i,1])
                        a['boxes'][i]._xy[[0,1,4],1] = q[i,0]
                        a['boxes'][i]._xy[[2,3],1] = q[i,2]
                        iqr = q[i,2]-q[i,0]
                        y = x[i]
                        top = np.max(y[y<q[i,2]+1.5*iqr])
                        bottom = np.min(y[y>q[i,0]-1.5*iqr])
                        a['whiskers'][2*i].set_ydata(np.array([q[i,0],bottom]))
                        a['whiskers'][2*i+1].set_ydata(np.array([q[i,2],top]))
                        a['caps'][2*i].set_ydata(np.array([bottom,bottom]))
                        a['caps'][2*i+1].set_ydata(np.array([top,top]))

                        low_fliers = y[y<bottom]
                        high_fliers = y[y>top]
                        a['fliers'][2*i].set_ydata(low_fliers)
                        a['fliers'][2*i].set_xdata([i+1]*low_fliers.size)
                        a['fliers'][2*i+1].set_ydata(high_fliers)
                        a['fliers'][2*i+1].set_xdata([i+1]*high_fliers.size)
                    object_name = string.replace(trunc_name[2],"cellFEA","Cell ")
                    object_name = string.replace(object_name,".pkl","")
                    ax.set_title(object_name)
                    ax.set_ylabel(output)
                    ax.set_xticklabels(labels, rotation=45)
                    try:
                        cell_directory = output_dir+'/boxplots/'+string.replace(object_name,' ','_')
                        os.mkdir(cell_directory)
                    except:
                        pass
                    plt.tight_layout()
                    plt.savefig(cell_directory+'/'+string.replace(output,' ','_')+'.svg')
                    plt.clf()
        for output in self.differences.keys():
            if self.differences[output]:
                try:
                    os.mkdir(output_dir+'/paired_differences')
                except:
                    pass
                for m in self.matched:
                    labels = []
                    for f in m:
                        if "Windows" in self.op_sys: 
                            trunc_name = f.rsplit("\\",2)
                        else:
                            trunc_name = f.rsplit("/",2)
                        labels.append(trunc_name[1])
                    N = len(m)
                    grid = np.zeros((N,N),float)
                    combos = list(itertools.combinations(m,2))
                    enum_combos = list(itertools.combinations(range(N),2))
                    for c,e in zip(combos,enum_combos):
                        grid[e[0],e[1]] = self.differences[output][c[1]+'MINUS'+c[0]]['weighted RMS']
                        grid[e[1],e[0]] = grid[e[0],e[1]]
                    fig, ax = plt.subplots()
                    high = np.max(np.ravel(grid))
                    low = np.min(np.ravel(grid))
                    if abs(low-high) < 1e-3:
                        high += 0.001
                    heatmap = ax.pcolormesh(grid, cmap=plt.cm.Blues, edgecolors='black',vmin=low,vmax=high)
                    ax.set_xticks(np.arange(N)+0.5,minor=False)
                    ax.set_yticks(np.arange(N)+0.5,minor=False)
                    ax.invert_yaxis()
                    ax.xaxis.tick_top()
                    ax.set_xticklabels(labels,minor=False)
                    ax.set_yticklabels(labels,minor=False)
                    plt.colorbar(heatmap)

                    object_name = string.replace(trunc_name[2],"cellFEA","Cell ")
                    object_name = string.replace(object_name,".pkl","")
                    ax.set_title(object_name, y=1.08)
                    try:
                        cell_directory = output_dir+'/paired_differences/'+string.replace(object_name,' ','_')
                        os.mkdir(cell_directory)
                    except:
                        pass
                    plt.tight_layout()
                    plt.savefig(cell_directory+'/'+string.replace(output,' ','_')+'.svg')
                    plt.clf()
        if self.analysis['Convert to VTK'].get():
            try:
                os.mkdir(output_dir+'/vtk')
            except:
                pass
            for m in self.matched:
                nnodes = len(self.data[m[0]]['nodes'])
                nelements = len(self.data[m[0]]['elements'])
                tetraPoints = vtk.vtkPoints()
                tetraPoints.SetNumberOfPoints(nnodes)
                for i,p in enumerate(self.data[m[0]]['nodes']):
                    tetraPoints.InsertPoint(i,p[0],p[1],p[2])

                tetraElements = []
                for i,e in enumerate(self.data[m[0]]['elements']):
                    tetraElements.append(vtk.vtkTetra())
                    for j in xrange(4):
                        tetraElements[i].GetPointIds().SetId(j,e[j]-1)
                vtkMesh = vtk.vtkUnstructuredGrid()
                vtkMesh.Allocate(i+1,i+1)
                for i,e in enumerate(tetraElements):
                    vtkMesh.InsertNextCell(e.GetCellType(),e.GetPointIds())
                vtkMesh.SetPoints(tetraPoints)
                for output in self.outputs.keys():
                    if self.outputs[output].get():
                        for f in m:
                            if "Windows" in self.op_sys: 
                                trunc_name = f.rsplit("\\",2)
                            else:
                                trunc_name = f.rsplit("/",2)
                            vtkArray = numpy_support.numpy_to_vtk(np.ravel(self.results[output][f]).astype('f'), deep=True, array_type=vtk.VTK_FLOAT)
                            vtkArray.SetName(trunc_name[1]+' '+output)
                            vtkMesh.GetCellData().AddArray(vtkArray)

                object_name = string.replace(trunc_name[2],"cellFEA","Cell")
                object_name = string.replace(object_name,".pkl",".vtu")
                idWriter = vtk.vtkXMLUnstructuredGridWriter()
                idWriter.SetFileName(output_dir+'/vtk/'+object_name)
                idWriter.SetInputData(vtkMesh)
                idWriter.Write()
                        
root = Tk()
root.title("Welcome to the FEBio pyCellAnalyst utility.")
app = Application(root)

root.mainloop()
