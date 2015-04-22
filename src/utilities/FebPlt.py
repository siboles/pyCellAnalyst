import numpy as np
import os, re

class FebPlt(object):
    '''
    DESCRIPTION: FebPlt parses FEBio binary plotfiles and stores
        them as a FebPlt object.
    USAGE: object_name = FebPlt.FebPlt(filename.xplt)
    MEMBERS:
        EASY ACCESS:
            object_name.TIME - list of output time points (one-to-one to rows
            of field variable arrays)
            object_name.ElementData - dictionary containing field variables
                stored on ELEMENTS
                EXAMPLE: object_name.ElementData[1]['stress']
                    returns a Nx6 numpy array where each row is voigt notation
                    stress tensor at an output time point
            object_name.NodeData - dictionary containing field variables
                stored on NODES
                EXAMPLE: object_name.NodeData[1]['displacement']
                    returns a Nx3 numpy array where each row is the
                    displacement vector at an output time point
            object_name.Mesh['Nodes'] - numpy array of nodal coordinates (nodes are always enumerated by 1, so ID is not stored)
            object_name.Mesh['Elements'] - dictionary with element IDs serving as keys
                each item is {'Type': element type, 'Connectivity': numpy array of node IDs}
            TODO:
                Add support for surface sets
        LOW LEVEL:
            Data are initially read and stored in a hierarchial block format
            similar to the original plot file.  This will be documented in
            the future.
    '''
    def __init__(self,filename):
        self._fid = open(filename,'rb')
        self._filesize = os.path.getsize(filename)
        self._read_size = 0
        self._matnames = {}
        self._lookup = {'01010001': 'VERSION',
                '01010002':'NODES',
                '01020000':'DICTIONARY', 
                '01021000':'GLOBAL_DATA',
                '01022000': 'MATERIAL_DATA',
                '01023000': 'NODESET_DATA',
                '01024000': 'DOMAIN_DATA',
                '01025000': 'SURFACE_DATA',
                '01020001': 'DICTIONARY_ITEM',
                '01020002': 'ITEM_TYPE',
                '01020003': 'ITEM_FORMAT',
                '01020004': 'ITEM_NAME',
                '01030001': 'MATERIAL',
                '01030002': 'MAT_ID',
                '01030003': 'MAT_NAME',
                '01041000': 'NODE_SECTION',
                '01041001': 'NODE_COORDS',
                '01042000': 'DOMAIN_SECTION',
                '01042100': 'DOMAIN',
                '01042101': 'DOMAIN_HEADER',
                '01042102': 'ELEM_TYPE',
                '01042103': 'MAT_ID',
                '01032104': 'ELEMENTS',
                '01042200': 'ELEMENT_LIST',
                '01042201': 'ELEMENT',
                '01043000': 'SURFACE_SECTION',
                '01043102': 'SURFACE_ID',
                '01043103': 'FACETS',
                '01043201': 'FACET',
                '02000000': 'STATE_SECTION',
                '02010000': 'STATE_HEADER',
                '02010002': 'TIME',
                '02020000': 'STATE_DATA',
                '02020001': 'STATE_VAR',
                '02020002': 'VARIABLE_ID',
                '02020003': 'VARIABLE_DATA',
                '02020100': 'GLOBAL_DATA',
                '02020200': 'MATERIAL_DATA',
                '02020300': 'NODESET_DATA',
                '02020400': 'DOMAIN_DATA',
                '02020500': 'SURFACE_DATA'
                }
        self.VERSION = None
        self.NODES = None
        self.DICTIONARY = {'GLOBAL_DATA': [], 
                        'MATERIAL_DATA':  [],
                        'NODESET_DATA':  [],
                        'DOMAIN_DATA':  [],
                        'SURFACE_DATA':  []}
        self.MATERIAL = []
        self.DOMAIN_SECTION = []
        self.SURFACE_SECTION = []
        self.STATE_SECTION = []

        self.TIME = []
        self.Mesh = {'Nodes': None, 'Elements': {}}
        self.NodeSets = {}
        self.ElementSets = {}
        self.SurfaceSets = {}
        self.ElementData = {}
        self.NodeData = {}

        self._parseModel()
        self._cleanData()
        self._makeSets()

    def _parseModel(self):
        chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
        if '%08x' % chunk != '00464542':
            print 'File passed to FebPlt() is not an FEBio xplt. Exiting...'
            raise SystemExit
        self._read_size += 4
        while self._read_size < self._filesize:
            chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
            self._read_size += 4
            try:
                keyword = self._lookup['%08x' % chunk]
                if keyword == 'VERSION':
                    self.VERSION = '%08x' % np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]
                    self._read_size += 8
                elif keyword == 'NODES':
                    self.NODES = np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]
                    self._read_size += 8
                elif keyword == 'DICTIONARY':
                    chunksize = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size
                    self._read_size += 4
                    while self._read_size < chunksize:
                        chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
                        self._read_size += 4
                        try:
                            keyword = self._lookup['%08x' % chunk]
                            if keyword in ['GLOBAL_DATA','MATERIAL_DATA','NODESET_DATA','DOMAIN_DATA','SURFACE_DATA']:
                                self._readDictSect(np.fromfile(self._fid,dtype=np.uint32,count=1)+self._read_size,keyword)
                        except:
                            continue
                elif keyword == 'MATERIAL':
                    chunksize = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size
                    self._read_size += 4
                    self._addMaterial()
                    while self._read_size < chunksize:
                        chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
                        self._read_size += 4
                        try:
                            keyword = self._lookup['%08x' % chunk]
                            if keyword == 'MAT_ID':
                                self.MATERIAL[-1]['MAT_ID'] = np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]
                                self._read_size += 8
                            elif keyword == 'MAT_NAME':
                                np.fromfile(self._fid,dtype=np.uint32,count=1)
                                self._read_size += 4
                                mname = np.fromfile(self._fid,dtype=np.dtype((str,64)),count=1)
                                self._read_size += 64
                                z = re.compile(r'(\x00)')
                                mname = str(z.split(mname[0])[0])
                                self.MATERIAL[-1]['MAT_NAME'] = mname
                                self._matnames[mname] = self.MATERIAL[-1]['MAT_ID']
                        except:
                            continue
                elif keyword == 'NODE_SECTION':
                    chunksize = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size
                    self._read_size += 4
                    while self._read_size < chunksize:
                        chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
                        self._read_size += 4
                        try:
                            keyword = self._lookup['%08x' % chunk]
                            if keyword == 'NODE_COORDS':
                                np.fromfile(self._fid,dtype=np.uint32,count=1)
                                self._read_size += 4
                                N = self.NODES
                                self.Mesh['Nodes'] = np.reshape(np.fromfile(self._fid,dtype=np.float32,count=3*N),(N,3))
                                self._read_size += 12*N
                        except:
                            continue
                elif keyword == 'DOMAIN_SECTION':
                    chunksize = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size
                    self._read_size += 4
                    while self._read_size < chunksize:
                        chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
                        self._read_size += 4
                        try:
                            keyword = self._lookup['%08x' % chunk]
                            if keyword == 'DOMAIN':
                                self._readDomain(np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size)
                        except:
                            continue
                elif keyword == 'SURFACE_SECTION':
                    chunksize = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size
                    self._read_size += 4
                    while self._read_size < chunksize:
                        chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
                        self._read_size += 4
                        try:
                            keyword = self._lookup['%08x' % chunk]
                            if keyword == 'SURFACE':
                                self._readSurface(np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size)
                        except:
                            continue
                elif keyword == 'STATE_SECTION':
                    vlengths = {'FLOAT': 1, 'VEC3F': 3, 'MAT3FS': 6}
                    chunksize = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size
                    self._read_size += 4
                    self._addStateSection()
                    while self._read_size < chunksize:
                        chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
                        self._read_size += 4
                        try:
                            keyword = self._lookup['%08x' % chunk]
                            if keyword == 'TIME':
                                self.STATE_SECTION[-1]['TIME'] = np.fromfile(self._fid,dtype=np.float32,count=2)[1]
                                self.TIME.append(self.STATE_SECTION[-1]['TIME'])
                                self._read_size += 8
                            elif keyword in ['GLOBAL_DATA','MATERIAL_DATA','NODESET_DATA','DOMAIN_DATA','SURFACE_DATA']:
                                size = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size
                                self._read_size += 4
                                while self._read_size < size:
                                    chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]
                                    self._read_size += 4
                                    try:
                                        subkeyword = self._lookup['%08x' % chunk]
                                        if subkeyword == 'VARIABLE_ID':
                                            self._addStateData(keyword)
                                            self.STATE_SECTION[-1][keyword][-1]['VARIABLE_ID'].append(np.fromfile(self._fid,dtype=np.uint32,count=2)[-1])
                                            self._read_size += 8
                                            itype = self.DICTIONARY[keyword][self.STATE_SECTION[-1][keyword][-1]['VARIABLE_ID'][-1]-1]['ITEM_TYPE']
                                        elif subkeyword == 'VARIABLE_DATA':
                                            block_size = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]+self._read_size
                                            self._read_size += 4
                                            while self._read_size < block_size:
                                                region_id = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]
                                                region_size = np.fromfile(self._fid,dtype=np.uint32,count=1)[0]
                                                self._read_size += 8
                                                self.STATE_SECTION[-1][keyword][-1]['DATA'].append({'REGION_ID': region_id, 
                                                    'DATA': np.reshape(np.fromfile(self._fid,dtype=np.float32,count=region_size/4),(-1,vlengths[itype]))})
                                                self._read_size += region_size
                                    except:
                                        continue
                        except:
                            continue
            except:
                continue
        self._fid.close()

    def _readDictSect(self,size,key):
        itype = ['FLOAT','VEC3F','MAT3FS']
        ifmt = ['NODE','ITEM','MULT']
        self._read_size += 4
        while self._read_size < size:
            chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
            self._read_size += 4
            try:
                keyword = self._lookup['%08x' % chunk]
                if keyword == 'DICTIONARY_ITEM':
                    self._addDictionaryItem(key)
                elif keyword == 'ITEM_TYPE':
                    self.DICTIONARY[key][-1]['ITEM_TYPE'] = itype[np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]]
                    self._read_size += 8
                elif keyword == 'ITEM_FORMAT':
                    self.DICTIONARY[key][-1]['ITEM_FORMAT'] = ifmt[np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]]
                    self._read_size += 8
                elif keyword == 'ITEM_NAME':
                    np.fromfile(self._fid,dtype=np.uint32,count=1) #read a 4 byte chunk to move to char64 array
                    self._read_size += 4
                    dmy = np.fromfile(self._fid,dtype=np.dtype((str,64)),count=1)
                    z = re.compile(r'(\x00)')
                    dmy = z.split(dmy[0])
                    self._read_size += 64
                    dmy = str(dmy[0])
                    self.DICTIONARY[key][-1]['ITEM_NAME'] = dmy
            except:
                continue

    def _readDomain(self,size):
        elmtypes = ['HEX8', 'PENTA6', 'TET4', 'QUAD4', 'TRI3', 'TRUSS2']
        numnodes = [8,6,4,4,3,2]
        self._addDomain()
        self._read_size += 4
        while self._read_size < size:
            chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
            self._read_size += 4
            try:
                keyword = self._lookup['%08x' % chunk]
                if keyword == 'ELEM_TYPE':
                    etype = np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]
                    self.DOMAIN_SECTION[-1]['DOMAIN_HEADER']['ELEM_TYPE'] = elmtypes[etype]
                    self._read_size += 8
                elif keyword == 'MAT_ID':
                    self.DOMAIN_SECTION[-1]['DOMAIN_HEADER']['MAT_ID'] = np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]
                    self._read_size += 8
                elif keyword == 'ELEMENTS':
                    self.DOMAIN_SECTION[-1]['DOMAIN_HEADER']['ELEMENTS'] = np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]
                    self._read_size += 8
                elif keyword == 'ELEMENT':
                    self.DOMAIN_SECTION[-1]['ELEMENT_LIST'].append(np.fromfile(self._fid,dtype=np.uint32,count=numnodes[etype]+2)[1:])
                    self._read_size += 4*(numnodes[etype]+2)
            except:
                continue

    def _readSurface(self,size):
        self._addSurface()
        self._read_size += 4
        while self._read_size < size:
            chunk = np.fromfile(self._fid,dtype=np.uint32,count=1)
            self._read_size += 4
            try:
                keyword = self._lookup['%08x' % chunk]
                if keyword == 'SURFACE_ID':
                    self.SURFACE_SECTION[-1]['SURFACE_HEADER']['SURFACE_ID'] = np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]
                    self._read_size += 8
                elif keyword == 'FACETS':
                    self.SURFACE_SECTION[-1]['SURFACE_HEADER']['FACETS'] = np.fromfile(self._fid,dtype=np.uint32,count=2)[-1]
                    self._read_size += 8
                elif keyword == 'FACET':
                    self.SURFACE_SECTION[-1]['FACET_LIST'].append(np.fromfile(self._fid,dtype=np.uint32,count=6)[1:])
                    self._read_size += 24
            except:
                continue

    def _addDictionaryItem(self,key):
        self.DICTIONARY[key].append({'ITEM_TYPE': 0, 'ITEM_FORMAT': 0, 'ITEM_NAME': ''})

    def _addMaterial(self):
        self.MATERIAL.append({'MAT_ID': 0, 'MAT_NAME': ''})
    
    def _addDomain(self):
        self.DOMAIN_SECTION.append({'DOMAIN_HEADER': {'ELEM_TYPE': 0, 'MAT_ID': 0, 'ELEMENTS': 0},
                        'ELEMENT_LIST': []})
    def _addSurface(self):
        self.SURFACE_SECTION.append({'SURFACE_HEADER': {'SURFACE_ID': 0, 'FACETS': 0}, 'FACET_LIST': []})

    def _addStateSection(self):
        self.STATE_SECTION.append({'TIME': 0.0,
                        'GLOBAL_DATA': [],
                        'MATERIAL_DATA':  [],
                        'NODESET_DATA':  [],
                        'DOMAIN_DATA':  [],
                        'SURFACE_DATA':  []})
    def _addStateData(self,key):
        self.STATE_SECTION[-1][key].append({'VARIABLE_ID': [], 'DATA': []})

    def _cleanData(self):
        mapping = []
        for d in self.DOMAIN_SECTION:
            mapping.append({'ITEM': [], 'NODE': [], 'MULT': []})
            used_node = {}
            for item in d['ELEMENT_LIST']:
                mapping[-1]['ITEM'].append(item[0])
                for n in item[1:]:
                    try:
                        used_node[n]
                    except:
                        mapping[-1]['NODE'].append(n)
                        used_node[n] = True
                    mapping[-1]['MULT'].append(n)
        N = len(self.STATE_SECTION)
        for i in xrange(N):
            for j in xrange(len(self.STATE_SECTION[i]['NODESET_DATA'][0]['DATA'][0]['DATA'])):
                try:
                    self.NodeData[j+1]['displacement'][i,:] = self.STATE_SECTION[i]['NODESET_DATA'][0]['DATA'][0]['DATA'][j]
                except:
                    self.NodeData[j+1] = {} 
                    self.NodeData[j+1]['displacement'] = np.zeros((N,3),dtype=np.float32)
                    self.NodeData[j+1]['displacement'][i,:] = self.STATE_SECTION[i]['NODESET_DATA'][0]['DATA'][0]['DATA'][j]

            for j in xrange(len(self.STATE_SECTION[i]['DOMAIN_DATA'])):
                vname = self.DICTIONARY['DOMAIN_DATA'][j]['ITEM_NAME']
                vformat = self.DICTIONARY['DOMAIN_DATA'][j]['ITEM_FORMAT']
                if vformat == 'ITEM':
                    for k in xrange(len(self.STATE_SECTION[i]['DOMAIN_DATA'][j]['DATA'])):
                        dat = self.STATE_SECTION[i]['DOMAIN_DATA'][j]['DATA'][k]['DATA']
                        m = mapping[k][vformat]
                        M,L = dat.shape    #rows in dat
                        for l in xrange(M):
                            try:
                                self.ElementData[m[l]][vname][i,:] = dat[l,:]
                            except:
                                try:
                                    self.ElementData[m[l]][vname] = np.zeros((N,L),dtype=np.float32)
                                except:
                                    self.ElementData[m[l]] = {}
                                    self.ElementData[m[l]][vname] = np.zeros((N,L),dtype=np.float32)
                                self.ElementData[m[l]][vname][i,:] =  dat[l,:]
                elif vformat == 'NODE':
                    for k in xrange(len(self.STATE_SECTION[i]['DOMAIN_DATA'][j]['DATA'])):
                        dat = self.STATE_SECTION[i]['DOMAIN_DATA'][j]['DATA'][k]['DATA']
                        m = mapping[k][vformat]
                        M,L = dat.shape    #rows in dat
                        for l in xrange(M):
                            n = m[l]+1
                            try:
                                self.NodeData[n][vname][i,:] = dat[l,:]
                            except:
                                try:
                                    self.NodeData[n][vname] = np.zeros((N,L),dtype=np.float32)
                                except:
                                    self.NodeData[n] = {}
                                    self.NodeData[n][vname] = np.zeros((N,L),dtype=np.float32)
                                self.NodeData[n][vname][i,:] =  dat[l,:]
                elif vformat == 'MULT':
                    accessed = {}
                    for k in xrange(len(self.STATE_SECTION[i]['DOMAIN_DATA'][j]['DATA'])):
                        dat = self.STATE_SECTION[i]['DOMAIN_DATA'][j]['DATA'][k]['DATA']
                        m = mapping[k][vformat]
                        M,L = dat.shape    #rows in dat
                        for l in xrange(M):
                            n = m[l]+1
                            try:
                                self.NodeData[n][vname][i,:] = self.NodeData[n][vname][i,:] + dat[l,:]
                                try:
                                    accessed[n] += 1
                                except:
                                    accessed[n] = 1
                            except:
                                try:
                                    self.NodeData[n][vname] = np.zeros((N,L),dtype=np.float32)
                                except:
                                    self.NodeData[n] = {}
                                    self.NodeData[n][vname] = np.zeros((N,L),dtype=np.float32)
                                self.NodeData[n][vname][i,:] = self.NodeData[n][vname][i,:] + dat[l,:]
                                accessed[n] = 1
                    for nid in self.NodeData.keys():
                        try:
                            self.NodeData[nid][vname][i,:] = self.NodeData[nid][vname][i,:]/accessed[nid]
                        except:
                            continue
    def _makeSets(self):
        names = {}
        for m in self.MATERIAL:
            try:
                names[m['MAT_ID']] = m['MAT_NAME']
                self.ElementSets[m['MAT_NAME']] = []
                self.NodeSets[m['MAT_NAME']] = []
            except:
                names[m['MAT_ID']] = m['MAT_ID']  #no name given to material
                self.ElementSets[m['MAT_ID']] = []
                self.NodeSets[m['MAT_NAME']] = []

        for s in self.DOMAIN_SECTION:
            setname = names[s['DOMAIN_HEADER']['MAT_ID']]
            used_nodes = {}
            for e in s['ELEMENT_LIST']:
                self.ElementSets[setname].append(e[0])
                self.Mesh['Elements'][e[0]] ={'Type': s['DOMAIN_HEADER']['ELEM_TYPE'],'Connectivity': e[1:]}
                for n in e[1:]:
                    try:
                        used_nodes[n]
                    except:
                        self.NodeSets[setname].append(n)
                        used_nodes[n] = True
