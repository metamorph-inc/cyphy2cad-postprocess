import os.path

import unittest
import numpy as np
import numpy.testing as nptest

from cyphy2cad_postprocess.reader import CyPhy2CADReader


class TestCyPhy2CADReader(unittest.TestCase):

    def test_output_dir_parse(self):
        cad_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        cad_reader = CyPhy2CADReader(cyphy2cad_output_dir=cad_output_dir, parse=True)
        cad_data = cad_reader.cad_data
        self.assertNotEqual(cad_data, None)
        self._check_cad_data(cad_data.data)

    def test_output_dir_noparse(self):
        cad_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        cad_reader = CyPhy2CADReader(cyphy2cad_output_dir=cad_output_dir)
        cad_data = cad_reader.cad_data
        self.assertEqual(cad_data, None)

    def test_nooutput_dir_parse(self):
        if os.getcwd() == os.path.join(os.path.dirname(os.path.abspath(__file__))):  # called from test dir
            cad_reader = CyPhy2CADReader(parse=True)
            cad_data = cad_reader.cad_data
            self.assertNotEqual(cad_data, None)
            self._check_cad_data(cad_data.data)
        else:
            with self.assertRaises(IOError):
                cad_reader = CyPhy2CADReader(parse=True)

    def test_nooutput_dir_noparse(self):
        cad_reader = CyPhy2CADReader()
        cad_data = cad_reader.cad_data
        self.assertEqual(cad_data, None)

    def test_parse_method(self):
        cad_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        cad_reader = CyPhy2CADReader()
        cad_data = cad_reader.parse(cad_output_dir)
        self._check_cad_data(cad_data.data)

    def _check_cad_data(self, data):

        def check_dict_kev_value(d, key, expected_val):
            self.assertIn(key, d.keys())
            self.assertEqual(d[key], expected_val)

        self.assertIn("components", data.keys())
        components = data["components"]
        expected_component_ids = ['28c38de6-a26e-4c44-926d-a63173498dace8ed074f-44f8-4354-931c-23c060d06b5a',
                                  '534f409d-2648-4d43-8a8f-88449313af9e041ce927-77c2-4099-b0bc-7ac113d3a0f3',
                                  '534f409d-2648-4d43-8a8f-88449313af9ebc6a5ce0-65ba-4fee-8a21-de0cf629ec69',
                                  '5c964f2c-cd43-458b-90c2-b27754253920',
                                  '28c38de6-a26e-4c44-926d-a63173498dacce7ec15c-0bda-4d5c-81ba-d149e6d990b8',
                                  '28c38de6-a26e-4c44-926d-a63173498dac59d3d05e-251f-4cab-bf72-ef784f039bf0',
                                  '0293b7a1-af8e-4522-8707-6fbf786585db',
                                  '534f409d-2648-4d43-8a8f-88449313af9ece7ec15c-0bda-4d5c-81ba-d149e6d990b8',
                                  '{272a6594-975d-4c4f-8cbe-b4f6e4d2b8f0}|1',
                                  '534f409d-2648-4d43-8a8f-88449313af9e59d3d05e-251f-4cab-bf72-ef784f039bf0',
                                  '76cddc75-c978-44ff-b83b-d3825ad83564', 'a316bf0e-b929-4b55-a142-489eaad74273',
                                  '28c38de6-a26e-4c44-926d-a63173498dac041ce927-77c2-4099-b0bc-7ac113d3a0f3',
                                  '28c38de6-a26e-4c44-926d-a63173498dacbc6a5ce0-65ba-4fee-8a21-de0cf629ec69',
                                  '534f409d-2648-4d43-8a8f-88449313af9ee8ed074f-44f8-4354-931c-23c060d06b5a']
        self.assertItemsEqual(components.keys(), expected_component_ids)

        top_asm_id = "{272a6594-975d-4c4f-8cbe-b4f6e4d2b8f0}|1"
        self.assertIn(top_asm_id, components.keys())
        top_asm = components[top_asm_id]
        self.assertIn("bounding_box", top_asm.keys())
        bbox = top_asm["bounding_box"]
        self.assertIn("bounding_box", bbox)
        nptest.assert_allclose(bbox["bounding_box"],
                               np.array([400.0, 486.2793604000001, 345.14999999999975]), rtol=1e-03)
        check_dict_kev_value(top_asm, "cad_filename_generated", "TestModel_1")
        check_dict_kev_value(top_asm, "cad_filename_original", "TestModel_1")
        check_dict_kev_value(top_asm, "cad_type", "ASSEMBLY")
        self.assertIn("center_of_gravity", top_asm.keys())
        nptest.assert_allclose(top_asm["center_of_gravity"],
                               np.array([135.18288805848897, 0.0, -115.75515182040105]), rtol=1e-03)
        check_dict_kev_value(top_asm, "component_name", "TestModel_1")
        check_dict_kev_value(top_asm, "coordinate_system", "DEFAULT")
        self.assertIn("inertia", top_asm.keys())
        inertia = top_asm["inertia"]
        self.assertIn("inertia_tensor_at_center_of_gravity", inertia.keys())
        nptest.assert_allclose(inertia["inertia_tensor_at_center_of_gravity"],
                               np.array([[3282608.9665136877, -148.6455123363897, 45517.73339260346],
                                         [-148.6455123363897, 914498.5206791221, 0.0],
                                         [45517.73339260346, 0.0, 2545592.3700071727]]), rtol=1e-03)
        self.assertIn("inertia_tensor_at_default_csys", inertia.keys())
        nptest.assert_allclose(inertia["inertia_tensor_at_default_csys"],
                               np.array([[3561133.291812647, -148.6455123363897, 370788.11744318693],
                                         [-148.6455123363897,1572884.9015699967,0.0],
                                         [370788.11744318693,0.0,2925454.425599088]]), rtol=1e-03)
        self.assertIn("principle_moments_of_inertia", inertia.keys())
        principle_moments_of_inertia = inertia["principle_moments_of_inertia"]
        self.assertIn("principle_moments", principle_moments_of_inertia.keys())
        nptest.assert_allclose(principle_moments_of_inertia["principle_moments"],
                               np.array([[914498.5113436849],
                                         [2542791.8616092685],
                                         [3285409.484247029]]), rtol=1e-03)
        self.assertIn("rotation_matrix", principle_moments_of_inertia.keys())
        nptest.assert_allclose(principle_moments_of_inertia["rotation_matrix"],
                               np.array([[6.280335717246306e-05, -0.06140953557236019, 0.9981126514559994],
                                         [0.9999999980263333, 5.6060242021740395e-06, -6.257719869068375e-05],
                                         [0.0, 0.9981126534161159, 0.06140953580323537]]), rtol=1e-03)
        self.assertIn("mass", top_asm.keys())
        self.assertAlmostEqual(float(top_asm['mass']), 20.786552812349075, 2)
        check_dict_kev_value(top_asm, "metric_id", "1")
        check_dict_kev_value(top_asm, "rotation", None)
        self.assertIn("surface_area", top_asm.keys())
        self.assertAlmostEqual(float(top_asm['surface_area']), 786649.1976811609, 2)
        check_dict_kev_value(top_asm, "translation", None)
        self.assertIn("units", top_asm.keys())
        units = top_asm["units"]
        check_dict_kev_value(units, "distance", "millimeter")
        check_dict_kev_value(units, "force", "kg mm/sec2")
        check_dict_kev_value(units, "mass", "kilogram")
        check_dict_kev_value(units, "temperature", "centigrade")
        check_dict_kev_value(units, "time", "second")
        self.assertIn("volume", top_asm.keys())
        self.assertAlmostEqual(float(top_asm['volume']), 12674211.599861905, 2)


if __name__ == '__main__':
    unittest.main()
