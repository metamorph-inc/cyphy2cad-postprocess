import os
import xml.etree.ElementTree as et
import json
from collections import defaultdict

import numpy as np

from json_encoders import NumpyEncoder
from json_utils import json_reformat_lists


class CyPhy2CADReader(object):
    def __init__(self, cyphy2cad_output_dir=None, parse=False):

        self._cad_data = None

        if not cyphy2cad_output_dir:
            cyphy2cad_output_dir = os.getcwd()
        self.cyphy2cad_output_dir = cyphy2cad_output_dir

        if parse:
            self.parse(cyphy2cad_output_dir)

    @property
    def cad_data(self):
        return self._cad_data

    def parse(self, cyphy2cad_output_dir=None):

        if not cyphy2cad_output_dir:
            cyphy2cad_output_dir = self.cyphy2cad_output_dir

        self._cad_data = CyPhy2CADData()

        self._parse_cadassembly_xml(cyphy2cad_output_dir)
        self._parse_cadassembly_metrics_xml(cyphy2cad_output_dir)
        self._parse_computed_values_xml(cyphy2cad_output_dir)

        return self._cad_data

    def _parse_cadassembly_xml(self, path):
        elem_tree = et.parse(os.path.join(path, "CADAssembly.xml"))
        root_elem = elem_tree.getroot()
        self._cad_data.cadassembly = root_elem

        comp_to_component_display_name = {}   # ComponentID to Component Display Name
        comp_to_cad_file_name_original = {}   # ComponentID to Original CAD File Name
        comp_to_type = {}                     # ComponentID to CAD Component Type (ASSEMBLY/PART)
        comp_to_constraints = {}              # ComponentID to CAD Assembly Constraint

        # mine CAD component data
        for cad_comp in root_elem.iter('CADComponent'):
            comp = cad_comp.get('ComponentID')
            comp_to_component_display_name[comp] = cad_comp.get('DisplayName')
            comp_to_cad_file_name_original[comp] = cad_comp.get('Name')
            comp_to_type[comp] = cad_comp.get('Type')
            constraints = []
            for constraint in cad_comp.findall('Constraint'):
                pairs = []
                # TODO
            if len(constraints) > 0:
                comp_to_constraints[comp] = constraints

        # record CAD component data
        for comp, met in comp_to_component_display_name.items():  # FIXME: items() instead of iteritems() for Python3 compatibility
            component_data = {
                'component_name': comp_to_component_display_name[comp],
                'cad_filename_original': comp_to_cad_file_name_original[comp],
                'cad_type': comp_to_type[comp]
                # TODO: 'constraints': comp_to_constraints.get(comp, None)
            }
            self._cad_data.components[comp].update(component_data)

    def _parse_cadassembly_metrics_xml(self, path=""):
        elem_tree = et.parse(os.path.join(path, "CADAssembly_metrics.xml"))
        root_elem = elem_tree.getroot()
        self._cad_data.cadassembly_metrics = root_elem

        comp_to_metric = {}                          # ComponentID to MetricID
        comp_to_rotation = {}                        # ComponentID to Rotation
        comp_to_translation = {}                     # ComponentID to Translation

        metric_to_csys = {}                          # MetricID to CoordinateSystem
        comp_to_cad_file_name_generated = {}         # MetricID to Generated CAD File Name
        cadname_to_metric = {}                       # CAD Name to MetricID

        metric_to_boundingbox_def_csys = {}          # MetricID to Bounding Box relative to CAD ASSEMBLY/PART Default CSYS
        metric_to_cog_vector_def_csys = {}           # MetricID to C.O.G. relative to CAD ASSEMBLY/PART Default CSYS
        metric_to_inertia_tensor_def_csys = {}       # MetricID to Inertia Tensor at CAD ASSEMBLY/PART Default CSYS
        metric_to_inertia_tensor_cog = {}            # MetricID to Inertia Tensor at CAD ASSEMBLY/PART C.O.G.
        metric_to_principle_moments_of_inertia = {}  # MetricID to Principle Moments of Inertia

        metric_to_surfacearea = {}                   # MetricID to Surface Area
        metric_to_volume = {}                        # MetricID to Volume
        metric_to_mass = {}                          # MetricID to Mass

        metric_to_units = {}                         # MetricID to Units

        def extract_matrix(element, dim=None):
            if not dim:
                dim = [3,3]  # default is 3x3 matrix
            matrix = [[0 for col in xrange(dim[1])] for row in xrange(dim[0])]
            for row_index, row in enumerate(element.findall("./Rows/Row")):
                for col_index, col in enumerate(row.findall('Column')):
                    matrix[row_index][col_index] = float(col.get('Value'))
            return matrix

        def extract_xyz_vector(element):
            return [float(element.get("X")), float(element.get("Y")), float(element.get("Z"))]

        def extract_xyz_point(element):
            return extract_xyz_vector(element)

        # mine CAD component data
        for cad_comp in root_elem.iter('CADComponent'):
            comp_to_metric[cad_comp.get('ComponentInstanceID')] = cad_comp.get('MetricID')

        for child in root_elem.findall("./MetricComponents/MetricComponent[@MetricID='1']/Children/ChildMetric"):
            comp = child.get('ComponentInstanceID')
            comp_to_rotation[comp] = np.array(extract_matrix(child.find('RotationMatrix')))
            comp_to_translation[comp] = np.array(extract_xyz_vector(child.find('Translation')))

        for met_comp in root_elem.iter('MetricComponent'):
            met = met_comp.get('MetricID')
            metric_to_csys[met] = met_comp.get('CoordinateSystem')
            comp_to_cad_file_name_generated[met] = met_comp.get('Name')
            cadname_to_metric[met_comp.get('Name')] = met
            bbox = met_comp.find('BoundingBox')
            bbox_pts = []
            for pt in bbox.findall('./OutlinePoints/Point'):
                bbox_pts.append(extract_xyz_point(pt))
            metric_to_boundingbox_def_csys[met] = {
                'bounding_box': np.array(extract_xyz_vector(bbox)),
                'outline_points': np.array(bbox_pts)
            }

            cog = met_comp.find('CG')
            if cog is not None:
                metric_to_cog_vector_def_csys[met] = np.array(extract_xyz_point(cog))

            inertia_def_csys = met_comp.find("./InertiaTensor[@At='DEFAULT_CSYS']")
            if inertia_def_csys is not None:
                metric_to_inertia_tensor_def_csys[met] = extract_matrix(inertia_def_csys)

            inertia_cog = met_comp.find("./InertiaTensor[@At='CENTER_OF_GRAVITY']")
            if inertia_cog is not None:
                metric_to_inertia_tensor_cog[met] = extract_matrix(inertia_cog)

            p_moments_inertia = met_comp.find('PrincipleMomentsOfInertia')
            if p_moments_inertia is not None:
                rotation_matrix = p_moments_inertia.find('RotationMatrix')
                metric_to_principle_moments_of_inertia[met] = {
                    'rotation_matrix': extract_matrix(rotation_matrix),
                    'principle_moments': extract_matrix(p_moments_inertia, dim=[3,1])
                }

            def get_scalar(metric_component, name):
                scalar = metric_component.find("./Scalars/Scalar[@Name='{}']".format(name))
                if scalar is not None:
                    return float(scalar.get('Value'))
                else:
                    return None

            metric_to_surfacearea[met] = get_scalar(met_comp, 'SurfaceArea')
            metric_to_volume[met] = get_scalar(met_comp, 'Volume')
            metric_to_mass[met] = get_scalar(met_comp, 'Mass')

            units = met_comp.find('Units')
            metric_to_units[met] = {
                'distance': units.get('Distance'),
                'force': units.get('Force'),
                'mass': units.get('Mass'),
                'temperature': units.get('Temperature'),
                'time': units.get('Time')
            }

        # record CAD component data
        for comp, met in comp_to_metric.items():  # FIXME: items() instead of iteritems() for Python3 compatibility
            component_data = {
                'metric_id': met,
                'rotation': comp_to_rotation.get(comp, None),
                'translation': comp_to_translation.get(comp, None),
                'coordinate_system': metric_to_csys[met],
                'cad_filename_generated': comp_to_cad_file_name_generated[met],
                'bounding_box': metric_to_boundingbox_def_csys[met],
                'center_of_gravity': metric_to_cog_vector_def_csys.get(met, None),
                'inertia': {
                    'inertia_tensor_at_default_csys': metric_to_inertia_tensor_def_csys.get(met, None),
                    'inertia_tensor_at_center_of_gravity': metric_to_inertia_tensor_cog.get(met, None),
                    'principle_moments_of_inertia': metric_to_principle_moments_of_inertia.get(met, None)
                },
                'surface_area': metric_to_surfacearea[met],
                'volume': metric_to_volume[met],
                'mass': metric_to_mass[met],
                'units': metric_to_units[met]
            }
            self._cad_data.components[comp].update(component_data)

    def _parse_computed_values_xml(self, path):
        elem_tree = et.parse(os.path.join(path, "ComputedValues.xml"))
        root_elem = elem_tree.getroot()
        self._cad_data.computed_values = root_elem

        comp_to_points = {}  # ComponentID to LinkPoints

        # mine CAD component data
        for cad_comp in root_elem.iter('Component'):
            comp = cad_comp.get('ComponentInstanceID')
            points = {}
            for metric in cad_comp.iter('Metric'):
                if ":" in metric.get('MetricID') and metric.get('ArrayValue') is not None:
                    pt_name = metric.get('MetricID').split(":")[-1]
                    pt_arrayvalue = [float(x) for x in metric.get('ArrayValue').split(";")]
                    points[pt_name] = np.array(pt_arrayvalue)
            if len(points.keys()) > 0:
                comp_to_points[comp] = points

        # record CAD component data
        for comp, v in comp_to_points.items():  # FIXME: items() instead of iteritems() for Python3 compatibility
            component_data = {
                'points': comp_to_points[comp]
            }
            self._cad_data.components[comp].update(component_data)


class CyPhy2CADData(object):
    def __init__(self):

        # Raw CyPhy2CAD Data
        self._cadassembly = None
        self._cadassembly_metrics = None
        self._computed_values = None

        self._data = {}
        self._cad_components = defaultdict(dict)

    def dump(self):
        json_str = json.dumps(self.data, indent=4, separators=(',', ': '), sort_keys=True, cls=NumpyEncoder)
        return json_reformat_lists(json_str)

    def write(self, filename):
        with open(filename, 'w') as f_out:
            f_out.write(self.dump())

    @property
    def cadassembly(self):
        return self._cadassembly

    @cadassembly.setter
    def cadassembly(self, cadassembly_xml):
        self._cadassembly = cadassembly_xml

    @cadassembly.deleter
    def cadassembly(self):
        self._cadassembly = None

    @property
    def cadassembly_metrics(self):
        return self._cadassembly_metrics

    @cadassembly_metrics.setter
    def cadassembly_metrics(self, cadassembly_metrics_xml):
        self._cadassembly_metrics = cadassembly_metrics_xml

    @cadassembly_metrics.deleter
    def cadassembly_metrics(self):
        self._cadassembly_metrics = None

    @property
    def computed_values(self):
        return self._computed_values

    @computed_values.setter
    def computed_values(self, computed_values_xml):
        self._computed_values = computed_values_xml

    @computed_values.deleter
    def computed_values(self):
        self._computed_values = None

    @property
    def components(self):
        return self._cad_components

    @components.setter
    def components(self, components):
        self._cad_components = components

    @components.deleter
    def components(self):
        self._cad_components = None

    @property
    def data(self):
        data = {
            'components': self._cad_components
        }
        return data
