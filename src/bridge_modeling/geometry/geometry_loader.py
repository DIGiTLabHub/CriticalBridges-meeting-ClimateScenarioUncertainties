"""
Geometry data loader - loads all bridge geometry data from JSON files.
"""
import json
from pathlib import Path


class GeometryLoader:
    """Centralized loader for all geometry data"""

    def __init__(self, data_dir: str = None):
        """
        Initialize geometry loader.

        Parameters
        ----------
        data_dir : str, optional
            Path to data/geometry directory. If None, uses default.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "geometry"
        else:
            self.data_dir = Path(data_dir)

    def load_nodes(self) -> list:
        """Load nodes as list of tuples for OpenSees"""
        with open(self.data_dir / "nodes.json", 'r') as f:
            data = json.load(f)

        nodes_dict = data['nodes']
        node_list = [(int(nid), coords['x'], coords['y'], coords['z'])
                    for nid, coords in nodes_dict.items()]
        return sorted(node_list, key=lambda x: x[0])

    def load_elements(self) -> list:
        """Load elements as list of tuples"""
        with open(self.data_dir / "elements.json", 'r') as f:
            data = json.load(f)

        elements_list = []
        for elem in data['elements']:
            if elem['format'] == 'full_element':
                elem_tuple = (
                    elem['type'], elem['id'], elem['node_i'], elem['node_j'],
                    elem['A_or_pts'], elem['E_or_secTag'], elem['I'],
                    elem['GJ'], elem['alphaY'], elem['alphaZ'], elem['transf_id']
                )
            else:
                elem_tuple = (elem['type'], elem['id'], elem['node_i'], elem['node_j'],
                            elem['A'], elem['E'], elem['I'])
            elements_list.append(elem_tuple)

        return elements_list

    def load_restraints(self) -> list:
        """Load restraints as list of tuples"""
        with open(self.data_dir / "restraints.json", 'r') as f:
            data = json.load(f)

        restraints_list = []
        for rest in data['data']:
            if isinstance(rest, dict):
                rest_tuple = (rest['node_id'], rest['dx'], rest['dy'], rest['dz'],
                            rest['rx'], rest['ry'], rest['rz'])
            else:
                rest_tuple = tuple(rest)
            restraints_list.append(rest_tuple)

        return sorted(restraints_list, key=lambda x: x[0])

    def load_constraints(self) -> list:
        """Load constraints as list of tuples"""
        with open(self.data_dir / "constraints.json", 'r') as f:
            data = json.load(f)

        constraints_list = []
        for cons in data['constraints']:
            cons_tuple = (cons['retained_node'], cons['constrained_node'], cons['dofs'])
            constraints_list.append(cons_tuple)

        return constraints_list

    def load_masses(self) -> list:
        """Load masses as list of tuples"""
        with open(self.data_dir / "masses.json", 'r') as f:
            data = json.load(f)

        masses_list = []
        for mass in data['masses']:
            mass_tuple = (mass['node_id'], mass['mX'], mass['mY'], mass['mZ'],
                        mass['mRX'], mass['mRY'], mass['mRZ'])
            masses_list.append(mass_tuple)

        return sorted(masses_list, key=lambda x: x[0])

    def load_all(self) -> dict:
        """Load all geometry data at once"""
        return {
            'nodes': self.load_nodes(),
            'elements': self.load_elements(),
            'restraints': self.load_restraints(),
            'constraints': self.load_constraints(),
            'masses': self.load_masses()
        }
