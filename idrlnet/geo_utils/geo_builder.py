""" A simple factory for constructing Geometric Objects"""

from .geo import Geometry
from .geo_obj import Line1D, Line, Tube2D, Rectangle, Circle, Plane, Tube3D, Box, Sphere, Cylinder, CircularTube, \
    Triangle, Heart

__all__ = ['GeometryBuilder']


class GeometryBuilder:
    GEOMAP = {'Line1D': Line1D,
              'Line': Line,
              'Rectangle': Rectangle,
              'Circle': Circle,
              'Channel2D': Tube2D,
              'Plane': Plane,
              'Sphere': Sphere,
              'Box': Box,
              'Channel': Tube3D,
              'Channel3D': Tube3D,
              'Cylinder': Cylinder,
              'CircularTube': CircularTube,
              'Triangle': Triangle,
              'Heart': Heart,
              }

    @staticmethod
    def get_geometry(geo: str, **kwargs) -> Geometry:
        """Simple factory method for constructing geometry object.
        :param geo: Specified a string for geometry, which should be in GeometryBuilder.GEOMAP
        :rtype geo: str
        :param kwargs:
        :return: A geometry object with given kwargs.
        :rtype: Geometry
        """
        assert geo in GeometryBuilder.GEOMAP.keys(), f'The geometry {geo} not implemented!'
        return GeometryBuilder.GEOMAP[geo](**kwargs)
