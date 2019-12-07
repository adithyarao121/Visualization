import numpy as np
from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor
from volume.volume import GradientVolume, Volume
from collections.abc import ValuesView
import math
from numba import jit
import random
import pandas as pd


# TODO: Implement trilinear interpolation

@jit(nopython=True)
def slicer_jit(volume: np.array, vec1: np.array, image_size: int, voxels: np.array, step: int=1, trilinear: bool=False):
    image_center = image_size // 2
    for i in range(0, image_size, step):
        for j in range(0, image_size, step):
            x_real = vec1[0, 0] * (i - image_center) \
                   + vec1[1, 0] * (j - image_center) \
                   + vec1[2, 0] * 1

            y_real = vec1[0, 1] * (i - image_center) \
                   + vec1[1, 1] * (j - image_center) \
                   + vec1[2, 1] * 1

            z_real = vec1[0, 2] * (i - image_center) \
                   + vec1[1, 2] * (j - image_center) \
                   + vec1[2, 2] * 1

            # get_voxel() is now integrated in numba
            if x_real < 0 or y_real < 0 or z_real < 0 or x_real >= 256 or y_real >= 256 or z_real >= 162:
                voxel_value = 0

            else:
                if trilinear:
                    # TODO Input your trilinear interpolation here
                    voxel_value = 0
                    pass
                else:
                    x = np.int(x_real)
                    y = np.int(y_real)
                    z = np.int(z_real)
                    voxel_value = volume[x, y, z]

            # Stores all voxel values
            voxels[(i * image_size + j)] = voxel_value

    return voxels

@jit(nopython=True)
def mip_jit(volume: np.array, vec1: np.array, image_size: int, voxels: np.array, step: int=1):
    image_center = image_size // 2
    for i in range(0, image_size, step):
        for j in range(0, image_size, step):
            voxel_value = 0
            for k in range(0, image_size, step):
                x_real = np.int(vec1[0, 0] * (i - image_center) \
                       + vec1[1, 0] * (j - image_center) \
                       + vec1[2, 0] * (k) \
                       + vec1[3, 0] * 1)

                y_real = np.int(vec1[0, 1] * (i - image_center) \
                       + vec1[1, 1] * (j - image_center) \
                       + vec1[2, 1] * (k) \
                       + vec1[3, 1] * 1)

                z_real = np.int(vec1[0, 2] * (i - image_center) \
                       + vec1[1, 2] * (j - image_center) \
                       + vec1[2, 2] * (k) \
                       + vec1[3, 2] * 1)

                # get_voxel() is now integrated in numba
                if x_real < 0 or y_real < 0 or z_real < 0 or x_real >= 256 or y_real >= 256 or z_real >= 162:
                    prod = 0
                else:
                    prod = volume[x_real, y_real, z_real]
                if voxel_value < prod:
                    voxel_value = prod

            voxels[i * image_size + j] = voxel_value

    return voxels



def get_trilinear(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
       return 0

    xf = int(math.floor(x))
    yf = int(math.floor(y))
    zf = int(math.floor(z))
    xc = int(math.ceil(x))
    yc = int(math.ceil(y))
    zc = int(math.ceil(z))

    if xc >= volume.dim_x:
        xc = xc-1
    if yc >= volume.dim_y:
        yc = yc-1
    if zc >= volume.dim_z:
        zc = zc-1

    p0 = volume.data[xf, yf, zf]
    p1 = volume.data[xc, yf, zf]
    p2 = volume.data[xf, yc, zf]
    p3 = volume.data[xc, yc, zf]
    p4 = volume.data[xf, yf, zc]
    p5 = volume.data[xc, yf, zc]
    p6 = volume.data[xf, yc, zc]
    p7 = volume.data[xc, yc, zc]

    # alpha = 0.5
    # beta = 0.5
    # gamma = 0.5

    prod = 0.125 * p0 + 0.125 * p1 + 0.125 * p2 + 0.125 * p3 + 0.125 * p4 + 0.125 * p5 + 0.125 * p6 + 0.125 * p7

    return prod


def get_voxel(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
        return 0

    x = int(math.floor(x))
    y = int(math.floor(y))
    z = int(math.floor(z))
    return volume.data[x, y, z]


class RaycastRendererImplementation(RaycastRenderer):
    """
    Class to be implemented.
    """

    def clear_image(self):
        """Clears the image data"""
        self.image.fill(0)

    # TODO: Implement trilinear interpolation


    def render_slicer(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        u_vector = view_matrix[0:3]
        v_vector = view_matrix[4:7]
        view_vector = view_matrix[8:11]
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Combine all vectors to create one matrix.
        vec1 = np.vstack([u_vector,
                          v_vector,
                          volume_center])

        # Predefine the voxel array, this is required for Numba, also note that np.float16 are not accepted by Numba.
        voxels = np.zeros((image_size * image_size), dtype=np.float32)

        # This returns all voxel values as a long array.
        values = slicer_jit(volume=volume.data, vec1=vec1, image_size=image_size,
                            voxels=voxels, step=1, trilinear=False)

        # Reshape to the image dimensions
        values = np.reshape(values, newshape=(image_size, image_size))

        # Normalize data,
        value_normalized = np.clip(np.floor(values / volume_maximum * 255), a_min=0, a_max=255)

        # Add an extra dimension for stacking the RGBA values a bit later
        value_normalized = np.expand_dims(value_normalized, axis=0)
        red = value_normalized
        green = red
        blue = red

        # Calculate alpha
        alpha = np.clip(value_normalized * 255, a_min=0, a_max=255)

        # Create image by stacking the layers, transposing (since rgba are now at the beginning), unravel (flatten)
        rgba = np.vstack([red, green, blue, alpha])
        image[:] = np.transpose(rgba, axes=[2, 1, 0]).ravel()
        print("u_vector=", u_vector)
        print("v_vector= ", v_vector)
        print("View vector= ", view_vector)

    def render_mip_normal(self, view_matrix: np.ndarray, volume: Volume, energy_volumes: dict, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]
        print("++++++++", u_vector)
        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]
        print("-------", v_vector)

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]
        print("*******", view_vector)
        print(view_matrix)
        # Center of the image. Image is squared
        image_center = image_size / 2
        # print("Image size=", image_size, "Image center=", image_center)
        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()
        print("___")
        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            print(i, " of", image_size)
            for j in range(0, image_size, step):
                value = 0
                for k in range(0, image_size, step):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         volume_center[0] + view_vector[0] * k
                    # print("voxel_coordinate_x=", voxel_coordinate_x)

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         volume_center[1] + view_vector[1] * k
                    # print("voxel_coordinate_y=", voxel_coordinate_y)

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         volume_center[2] + view_vector[2] * k
                    # print("voxel_coordinate_z=", voxel_coordinate_z)

                    trial = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                    if value < trial:
                        value = trial

                # Get voxel value
                # value = get_trilinear(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                # Normalize value to be between 0 and 1
                # print("$$$", value)
                red = value / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # print("before flooring red=", red, ", blue", blue, ", green", green)
                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255
                # print("after flooring red=", red, ", blue", blue, ", green", green)
                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

        gradient_volume = GradientVolume(volume)
        gradient_volume.compute()

        for key in energy_volumes:
            energy_color = TFColor(random.random(), random.random(), random.random(), 0.5)
            gradient_volume_energy = GradientVolume(energy_volumes[key])
            gradient_volume_energy.compute()
            self.render_energies(view_matrix, energy_volumes[key], image_size, image, gradient_volume_energy,
                                 energy_color)

    def render_mip(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        u_vector = view_matrix[0:3]
        v_vector = view_matrix[4:7]
        view_vector = view_matrix[8:11]
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Combine all vectors to create one matrix.
        vec1 = np.vstack([u_vector,
                          v_vector,
                          view_vector,
                          volume_center])

        # Predefine the voxel array, this is required for Numba, also note that np.float16 are not accepted by Numba.
        voxels = np.zeros((image_size * image_size), dtype=np.float32)

        # This returns all voxel values as a long array.
        values = mip_jit(volume=volume.data, vec1=vec1, image_size=image_size, voxels=voxels, step=1)

        # Reshape to the image dimensions
        values = np.reshape(values, newshape=(image_size, image_size))

        # Normalize data,
        value_normalized = np.clip(np.floor(values / volume_maximum * 255), a_min=0, a_max=255)

        # Add an extra dimension for stacking the RGBA values a bit later
        value_normalized = np.expand_dims(value_normalized, axis=0)
        red = value_normalized
        green = red
        blue = red

        # Calculate alpha
        alpha = np.clip(value_normalized * (2 ** 8), a_min=0, a_max=255)

        # Create image by stacking the layers, transposing (since rgba are now at the beginning), unravel (flatten)
        rgba = np.vstack([red, green, blue, alpha])
        image[:] = np.transpose(rgba, axes=[2, 1, 0]).ravel()
        print("u_vector=", u_vector)
        print("v_vector= ", v_vector)
        print("View vector= ", view_vector)

    # TODO: Implement Compositing function. TFColor is already imported. self.tfunc is the current transfer function.

    def render_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]
        print("++++++++", u_vector)
        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]
        print("-------", v_vector)

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]
        print("*******", view_vector)
        print(view_matrix)
        # Center of the image. Image is squared
        image_center = image_size / 2
        # print("Image size=", image_size, "Image center=", image_center)
        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()
        print("___")
        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(71, image_size - 70, step):
            print(i, " of", image_size)

            for j in range(71, image_size - 70, step):
                value = 0
                colour = TFColor()
                for k in range(0, 81, step):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                        volume_center[0] + view_vector[0] * k
                    # print("voxel_coordinate_x=", voxel_coordinate_x)

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                        volume_center[1] + view_vector[1] * k
                    # print("voxel_coordinate_y=", voxel_coordinate_y)

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                        volume_center[2] + view_vector[2] * k
                    # print("voxel_coordinate_z=", voxel_coordinate_z)

                    value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                # Get voxel value

                # Normalize value to be between 0 and 1
                    colour_of_voxel = self.tfunc.get_color(value)
                    colour.r = colour_of_voxel.r * colour_of_voxel.a + (1 - colour_of_voxel.a) * colour.r
                    colour.b = colour_of_voxel.b * colour_of_voxel.a + (1 - colour_of_voxel.a) * colour.b
                    colour.g = colour_of_voxel.g * colour_of_voxel.a + (1 - colour_of_voxel.a) * colour.g
                    colour.a = colour_of_voxel.a * colour_of_voxel.a + (1 - colour_of_voxel.a) * colour.a
                    red = colour.r
                    green = colour.g
                    blue = colour.b
                    alpha = colour.a

                    # print("before flooring red=", red, ", blue", blue, ", green", green)
                    # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255
                # print("after flooring red=", red, ", blue", blue, ", green", green)
                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    # TODO: Mouse Brain TF
    def mousebrain_TF(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray, gradient_volume: GradientVolume):
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):  # keep the starting point of i and j loop at 71 and end at image_size-70
            for j in range(0, image_size, step):

                Ctotal = 0
                AlphaTotal = 0
                for k in range(0, image_size, step):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         volume_center[0] + view_vector[0] * (k)

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         volume_center[1] + view_vector[1] * (k)

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         volume_center[2] + view_vector[2] * (k)

                    # Get voxel value
                    voxel_value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                    if (voxel_value == -1):
                        voxel_value = volume_maximum/2

                    red = voxel_value / volume_maximum

                    alpha = gradient_volume.get_gradient(int(voxel_coordinate_x), int(voxel_coordinate_y),
                                                         int(voxel_coordinate_z))

                    Ctotal = Ctotal * (1 - alpha) + red * alpha
                    AlphaTotal = AlphaTotal + (1 - AlphaTotal) * alpha


                print("Ctotal red: ", i, " ", j, " Color: ", Ctotal, "Alpha: ", AlphaTotal)
                #red = Ctotal.r * 255 if Ctotal.r < 255 else 255
                #green = Ctotal.g * 255 if Ctotal.g < 255 else 255
                #blue = Ctotal.b * 255 if Ctotal.b < 255 else 255
                #alpha = Ctotal.a * 255 if Ctotal.a < 255 else 255

                Ctotal = Ctotal * 255 if Ctotal < 255 else 255
                AlphaTotal = AlphaTotal * 255 if AlphaTotal < 255 else 255

                image[(j * image_size + i) * 4] = Ctotal
                image[(j * image_size + i) * 4 + 1] = 0
                image[(j * image_size + i) * 4 + 2] = 0
                image[(j * image_size + i) * 4 + 3] = AlphaTotal

    # TODO: Implement function to render multiple energy volumes and annotation volume as a silhouette.
    # TODO: Mouse Brain TF
    def render_annotation(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray,
                          gradient_volume: GradientVolume, energy_volumes: dict, color: TFColor, LUT: dict):

        # TODO: How do I zoom a little more?
        # TODO: How do you suggest I have a base color for an energy, and make deeper or lighter colors on top of that?

        # Tweak Factor: Vary to get optimum results
        r = 1
        # Tweak Factor: Vary to get optimum results

        # TODO: How should I initialize this
        fv = volume.get_maximum()  # fv should probably be choosen to a value that occurs the most.

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 70 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):

                Ctotal = TFColor(0, 0, 0, 0)
                alpha = 0.8
                for k in range(0, image_size, step):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         volume_center[0] + view_vector[0] * k

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         volume_center[1] + view_vector[1] * k

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         volume_center[2] + view_vector[2] * k

                    # Get voxel value

                    voxel_value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                    if voxel_value > 0 and voxel_value in LUT:
                        hex_color = LUT[voxel_value]
                        color.r, color.g, color.b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
                        Ctotal.r = Ctotal.r * (1 - alpha) + color.r * alpha
                        Ctotal.g = Ctotal.g * (1 - alpha) + color.g * alpha
                        Ctotal.b = Ctotal.b * (1 - alpha) + color.b * alpha
                        Ctotal.a = Ctotal.a * (1 - alpha) + color.a * alpha
                    else:
                        Ctotal.r = Ctotal.r
                        Ctotal.g = Ctotal.g
                        Ctotal.b = Ctotal.b
                        Ctotal.a = Ctotal.a

                    # TODO: How do you suggest I handle the color?
                    # red = voxel_value / volume_maximum

                    # gradient = gradient_volume.get_gradient(voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)

                    # if gradient.magnitude == 0 and voxel_value == fv:
                    #    alpha = 1
                    # elif voxel_value - r * gradient.magnitude < fv < voxel_value + r * gradient.magnitude:

                    #    alpha = 1 - (fv - voxel_value) / (r * gradient.magnitude)

                    # else:
                    #    alpha = 0

                Ctotal.r = (Ctotal.r + Ctotal.g + Ctotal.b)/3
                Ctotal.b = Ctotal.r
                Ctotal.g = Ctotal.r
                print("Ctotal red: ", i, " ", j, " Color: ", Ctotal, "Alpha: ")
                Ctotal.r = Ctotal.r * 255 if Ctotal.r < 255 else 255
                Ctotal.g = Ctotal.g * 255 if Ctotal.g < 255 else 255
                Ctotal.b = Ctotal.b * 255 if Ctotal.b < 255 else 255
                Ctotal.a = Ctotal.a * 255 if Ctotal.a < 255 else 255


                image[(j * image_size + i) * 4] = Ctotal.r
                image[(j * image_size + i) * 4 + 1] = Ctotal.g
                image[(j * image_size + i) * 4 + 2] = Ctotal.b
                image[(j * image_size + i) * 4 + 3] = 255


        gradient_volume = GradientVolume(volume)
        gradient_volume.compute()

        for key in energy_volumes:
            energy_color = TFColor(random.random(), random.random(), random.random(), 0.5)
            gradient_volume_energy = GradientVolume(energy_volumes[key])
            gradient_volume_energy.compute()
            self.render_energies(view_matrix, energy_volumes[key], image_size, image, gradient_volume_energy,energy_color)

    def render_energies(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray,
                        gradient_volume: GradientVolume, color: TFColor):

        # Tweak Factor: Vary to get optimum results
        r = 1
        # Tweak Factor: Vary to get optimum results

        # TODO: How should I initialize this
        fv = volume.get_maximum()  # fv should probably be choosen to a value that occurs the most.

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 70 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):

                Ctotal = TFColor(0, 0, 0, 0)
                # alpha = 0.8
                for k in range(0, volume.dim_z, step):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         volume_center[0] + view_vector[0] * k

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         volume_center[1] + view_vector[1] * k

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         volume_center[2] + view_vector[2] * k

                    # Get voxel value
                    voxel_value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                    gradient = gradient_volume.get_gradient(int(voxel_coordinate_x),
                                                            int(voxel_coordinate_y),
                                                            int(voxel_coordinate_z))

                    if voxel_value > 0:

                        if gradient.magnitude == 0 and voxel_value == fv:
                            alpha = 1
                        elif voxel_value - r * gradient.magnitude < fv < voxel_value + r * gradient.magnitude:

                            alpha = 1 - (fv - voxel_value) / (r * gradient.magnitude)

                        else:
                            alpha = 0

                        Ctotal.r = Ctotal.r * (1 - alpha) + color.r * alpha
                        Ctotal.g = Ctotal.g * (1 - alpha) + color.g * alpha
                        Ctotal.b = Ctotal.b * (1 - alpha) + color.b * alpha
                        Ctotal.a = Ctotal.a * (1 - alpha) + color.a * alpha

                print("Ctotal red: ", i, " ", j, " Color: ", Ctotal, "Alpha: ")
                Ctotal.r = Ctotal.r * 255 if Ctotal.r < 255 else 255
                Ctotal.g = Ctotal.g * 255 if Ctotal.g < 255 else 255
                Ctotal.b = Ctotal.b * 255 if Ctotal.b < 255 else 255
                Ctotal.a = Ctotal.a * 255 if Ctotal.a < 255 else 255

                if Ctotal.r != 0:
                    image[(j * image_size + i) * 4] = Ctotal.r
                    image[(j * image_size + i) * 4 + 1] = Ctotal.g
                    image[(j * image_size + i) * 4 + 2] = Ctotal.b
                    image[(j * image_size + i) * 4 + 3] = 255




    def render_mouse_brain(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict,
                           image_size: int, image: np.ndarray):

        # self.render_mip_normal(view_matrix, annotation_volume, energy_volumes, image_size, image)


        # annotation 8,3,2 are super small. No use visualizing
        # 8 is just dots
        # Do all visualization practices with annotation_7
        image.fill(0)
        #
        gradient_volume = GradientVolume(annotation_volume)
        gradient_volume.compute()
        df = pd.read_csv("structures.csv")
        LUT = pd.Series(df.color.values, index=df.database_id).to_dict()
        #
        annotation_color = TFColor(0, 0, 0, 0)

        self.render_annotation(view_matrix, annotation_volume, image_size, image, gradient_volume, energy_volumes,
                               annotation_color, LUT)
        # gradient_volume = GradientVolume(annotation_volume)
        # gradient_volume.compute()

class GradientVolumeImpl(GradientVolume):
    # TODO: Implement gradient compute function. See parent class to check available attributes.
    def compute(self):
        pass
