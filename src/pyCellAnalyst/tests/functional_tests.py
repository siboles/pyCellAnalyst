import unittest
import tempfile
import shutil
import os

import pyCellAnalyst
from pyCellAnalyst.util import generateImages
import SimpleITK as sitk
import vtk
import numpy as np

class ImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._imageRootDir, cls._regions = generateImages.generateTestImages(number=2, noiseLevel=0.5,
                                                                            spacing=[0.1, 0.1, 0.1],
                                                                            output=tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._imageRootDir)

    def setUp(self):
        self.im = pyCellAnalyst.FloatImage(os.path.join(self._imageRootDir, 'ref.nii'), spacing=[0.1, 0.1, 0.1])

    def test_image_from_file(self):
        self.assertIsInstance(self.im, pyCellAnalyst.Image)
        # test that spacing can be set after instantiation
        self.im.spacing = [1.0, 1.0, 1.0]
        # here we try to set the spacing incorrectly for 3d image; if an error isn't thrown, test fails
        try:
            self.im.spacing=[1.0, 1.0]
            self.fail()
        except:
            pass

    def test_image_from_stack(self):
        # create an image stack
        tmp_dir = tempfile.mkdtemp()
        size = self.im.image.GetSize()
        for i in range(size[2]):
            islice = sitk.Extract(self.im.image, [size[0], size[1], 0], [0, 0, i])
            sitk.WriteImage(islice, os.path.join(tmp_dir, 'slice{:03d}.tif'.format(i)))
        im = pyCellAnalyst.FloatImage(tmp_dir, spacing=[0.1, 0.1, 0.1])
        self.assertIsInstance(im, pyCellAnalyst.Image)
        shutil.rmtree(tmp_dir)

    def test_image_to_and_from_numpy(self):
        self.im.convertToNumpy()
        self.assertIsInstance(self.im.numpyimage, np.ndarray)
        im2 = pyCellAnalyst.FloatImage(self.im.numpyimage, spacing=[0.1, 0.1, 0.1])
        self.assertIsInstance(im2, pyCellAnalyst.Image)

    def test_image_to_vtk(self):
        vtk_filepath = os.path.join(self._imageRootDir, 'img')
        self.im.writeAsVTK(name=vtk_filepath)
        self.assertTrue(os.path.isfile('.'.join([vtk_filepath, 'vti'])))

    def test_invert_image(self):
        self.im.invert()

    def test_filter_gaussian(self):
        f = pyCellAnalyst.Gaussian(inputImage=self.im)
        f.parameters['variance'] = 1.0
        f.execute()

    def test_filter_curvature_anisotropic_diffusion(self):
        f = pyCellAnalyst.CurvatureAnisotropicDiffusion(inputImage=self.im)
        f.parameters['iterations'] = 15
        f.parameters['conductance'] = 10
        f.execute()

    def test_filter_gradient_anisotropic_diffusion(self):
        f = pyCellAnalyst.GradientAnisotropicDiffusion(inputImage=self.im)
        f.parameters['iterations'] = 15
        f.parameters['conductance'] = 10
        f.parameters['time_step'] = 0.005
        f.execute()

    def test_filter_bilateral(self):
        f = pyCellAnalyst.Bilateral(inputImage=self.im)
        f.parameters['domainSigma'] = 0.2
        f.parameters['rangeSigma'] = 5.0
        f.parameters['numberOfRangeGaussianSamples'] = 50
        f.execute()

    def test_filter_patchbased_denoising(self):
        f = pyCellAnalyst.PatchBasedDenoising(inputImage=self.im)
        f.parameters['radius'] = 2
        f.parameters['patches'] = 10
        f.parameters['iterations'] = 2
        f.parameters['noise_model'] = 'gaussian'
        f.execute()

    def test_filter_enhance_edges(self):
        f = pyCellAnalyst.EnhanceEdges(inputImage=self.im)
        f.execute()

    def test_filtering_pipeline(self):
        p = pyCellAnalyst.FilteringPipeline(inputImage=self.im)
        p.addFilter(pyCellAnalyst.CurvatureAnisotropicDiffusion(iterations=10))
        p.addFilter(pyCellAnalyst.EnhanceEdges())
        p.execute()

    def test_regions_of_interest(self):
        rois = pyCellAnalyst.RegionsOfInterest(inputImage=self.im, regions_of_interest=self._regions["reference"][0])
        for i in rois.images:
            self.assertIsInstance(i, type(self.im))

    def test_regions_of_interest_2dFrom3d(self):
        regions = np.copy(self._regions["reference"][0])
        regions[:,-1] = 1
        regions[:,2] += np.array(self._regions["reference"][0])[:,-1] // 2
        rois = pyCellAnalyst.RegionsOfInterest(inputImage=self.im, regions_of_interest=regions)
        for i in rois.images:
            self.assertIsInstance(i, type(self.im))

    def test_slice_viewer(self):
        f = pyCellAnalyst.CurvatureAnisotropicDiffusion(inputImage=self.im, iterations=20)
        f.execute()
        sv = pyCellAnalyst.SliceViewer([self.im, f.outputImage], titles=['Original', 'CurvatureAnisotropicDiffusion'])
        sv.view()

class SegmentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._imageRootDir, cls._regions = generateImages.generateTestImages(number=2, noiseLevel=0.5,
                                                                            spacing=[0.1, 0.1, 0.1],
                                                                            output=tempfile.mkdtemp())
        cls._im = pyCellAnalyst.FloatImage(os.path.join(cls._imageRootDir, 'ref.nii'), spacing=[0.1, 0.1, 0.1])
        cls._roi = pyCellAnalyst.RegionsOfInterest(inputImage=cls._im, regions_of_interest=cls._regions["reference"][0])
        cls._smooth_images = []
        for i in cls._roi.images:
            f = pyCellAnalyst.CurvatureAnisotropicDiffusion(inputImage=i, iterations=50)
            f.execute()
            cls._smooth_images.append(f.outputImage)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._imageRootDir)

    def test_threshold_methods(self):
        segs = []
        titles = []
        for c in pyCellAnalyst.Threshold.__subclasses__():
            thres = c(inputImage=self._smooth_images[0])
            thres.execute()
            titles.append(c.__name__)
            segs.append(thres.outputImage)

        sv = pyCellAnalyst.SliceViewer(inputImages=segs, titles=titles)
        sv.view()

    def test_threshold_two_pass(self):
        otsu = pyCellAnalyst.Otsu(inputImage=self._smooth_images[0], objectID=3)
        otsu.execute()

        otsu2 = pyCellAnalyst.Otsu(inputImage=self._smooth_images[0], mask=otsu.outputImage)
        otsu2.execute()
        sv = pyCellAnalyst.SliceViewer(inputImages=[self._roi.images[0], otsu.outputImage, otsu2.outputImage], titles=['Original', 'Otsu 1st Pass', 'Otsu Second Pass'])
        sv.view()

    def test_geodesic_active_contour(self):
        s1 = pyCellAnalyst.GeodesicActiveContour(inputImage=self._smooth_images[0], propagationScaling=20.0,
                                                 curvatureScaling=1.0, advectionScaling=1.0, maximumRMSError=0.005)
        s1.execute()


        # test using a thresholded image as seed. SimpleITK used directly to eliminate dependence on other tested features.
        seed = pyCellAnalyst.EightBitImage(sitk.OtsuThreshold(self._smooth_images[0].image, 0, 1),
                                           spacing=self._smooth_images[0].spacing)
        s2 = pyCellAnalyst.GeodesicActiveContour(inputImage=self._smooth_images[0], seed=seed,
                                                 propagationScaling=20.0, curvatureScaling=1.0,
                                                 advectionScaling=1.0, maximumRMSError=0.005)
        s2.execute()

        sv = pyCellAnalyst.SliceViewer(inputImages=[self._roi.images[0], s1.edgePotential, s1.outputImage, s2.outputImage], titles=['Original', 'Edge Potential', 'No seed', 'Otsu Seed 1'])
        sv.view()

    def test_visualize_isosurfaces(self):
        pass

class DeformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_surface_from_file(self):
        pass

    def test_surfaces_from_filelist(self):
        pass

    def test_surfaces_from_directory(self):
        pass

    def test_surfaces_from_memory(self):
        pass

    def test_ellipsoidal_method(self):
        pass

    def test_affine_transformation_method(self):
        pass

    def test_diffeomorphic_demons_method(self):
        pass

    def test_visualize_displacements(self):
        pass

if __name__ == '__main__':
    unittest.main()
