from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from filmscope.util import display_with_points


# take a bunch of lines and condense them to a set of vertical and horizontal lines
def condense_lines(lines, angle_thresh=0.05, r_thresh=80):
    if lines is None:
        return {"horizontal": [], "vertical": []}
    for i in range(lines.shape[0]):
        if abs(lines[i, :, 1] - np.pi) < angle_thresh:
            lines[i, :, 1] = lines[i, :, 1] - np.pi
            lines[i, :, 0] = -lines[i, :, 0]

    angle_dict = {"horizontal": 0, "vertical": np.pi / 2}

    lines_dict = {}

    for direction, angle in angle_dict.items():
        line_group = np.asarray(
            lines[np.where(abs(lines[:, :, 1] - angle) < angle_thresh)]
        )
        # sort the line group according to r (first index)
        line_group = line_group[line_group[:, 0].argsort()]

        new_line_group = []
        for entry in line_group:
            # make sure we didn't already handle a close value
            if (
                len(new_line_group) > 0
                and len(
                    np.where(
                        abs(np.asarray(new_line_group)[:, 0] - entry[0]) < r_thresh
                    )[0]
                )
                > 0
            ):
                continue

            close_entries = line_group[
                np.where(abs(line_group[:, 0] - entry[0]) < r_thresh)
            ]
            entry = np.mean(close_entries, axis=0)
            new_line_group.append([entry.tolist()])
        lines_dict[direction] = np.asarray(new_line_group).squeeze()

    # this looks like something we should read more later:
    # https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv

    return lines_dict


def make_binary_image(image, threshold_values):
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        threshold_values["adaptive_threshold_range"],
        2,
    )
    blurred = cv2.medianBlur(binary, threshold_values["blur_range"])

    return blurred


def get_line_intersect(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1  # [0]
    rho2, theta2 = line2  # [0]
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    # x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def find_approx_points(lines_dict):
    all_points = []
    for hline in lines_dict["horizontal"]:
        for vline in lines_dict["vertical"]:
            point = get_line_intersect(hline, vline)
            all_points.append(point)
    return np.asarray(all_points)

# this will return True if the area around center_point is determined to have
# a line along the "axis" direction.
# both of which are in display coordinates
def determine_line_validity(
    image,
    crop_distance,
    center_point,
    esf,
    lsf,
    show=False,
    show_if_failed=False,
):
    is_line = True

    # Step 1: assess standard deviation of surrounding image
    start0 = max(int(center_point[1] - crop_distance), 0)
    end0 = min(int(center_point[1] + crop_distance), image.shape[0])
    start1 = max(int(center_point[0] - crop_distance), 0)
    end1 = min(int(center_point[0] + crop_distance), image.shape[1])

    subimage = image[start0:end0, start1:end1]

    deviation = np.std(subimage)
    deviation_thresh = 1
    if deviation < deviation_thresh:
        is_line = False

    # Step 2: Check whether there is a clear low point in the ESF
    esf = esf - np.mean(subimage)
    esf = esf / np.max(np.abs(subimage))
    m2 = np.min(esf)
    # important note: this was -0.04 for a long time
    m2_thresh = -0.04
    if m2 > m2_thresh:
        is_line = False

    # Step 3: Then check whether LSF takes correct shape right before and after low point in ESF
    lsf = lsf / np.max(np.abs(subimage))
    min_avg_number = 5
    avg_number = 20
    min_index = np.where(esf == np.min(esf))[0][0]
    min_index2 = np.where(esf == np.min(esf))[0][-1]
    # TODO get rid of this
    if min_index == 0:
        min_index = 1

    if min_index < min_avg_number or min_index2 > len(esf) - min_avg_number:
        is_line = False

    start = max(min_index - avg_number, 0)
    sum_below = np.sum(lsf[start:min_index])
    if sum_below > m2 / 3:  # maybe use something else?
        is_line = False
    end = min(min_index2 + 1 + avg_number, len(lsf))
    sum_above = np.sum(lsf[min_index2:end])
    if sum_above < -m2 / 3:
        is_line = False

    if show or (show_if_failed and not is_line):
        plt.figure()
        plt.plot(esf, label="esf")
        plt.axhline(y=0)
        plt.axvline(x=min_index, linestyle="--", color="black")
        plt.axvline(x=min_index2, linestyle="--", color="black")
        plt.plot(np.arange(avg_number) + min_index + 1, np.ones(avg_number) * sum_above)
        plt.plot(
            np.arange(avg_number) + min_index - avg_number,
            np.ones(avg_number) * sum_below,
        )
        plt.plot(lsf, label="lsf")
        plt.legend()
        plt.show()

        plt.figure()
        plt.imshow(subimage)
        plt.colorbar()
        plt.title(f"is this a line? {is_line}. m2: {m2}. deviation: {deviation}")
        plt.show()

    return is_line


# this is a little more of a brute force approach
def _get_surrounding_indices(line_values, point, ignore_warning=False):
    for i in range(len(line_values) - 1):
        p1 = line_values[i]
        p2 = line_values[i + 1]

        if p1 <= point and p2 >= point:
            return i, i + 1
        if p1 >= point and p2 <= point:
            return i, i + 1

    if not ignore_warning:  
        print("may be adjusting center at non-line area")
        print("failing.")
    return -1, -1


def get_surrounding_indices(line_values, point, ignore_warning=False):
    idx0 = np.argmin(np.abs(line_values - point))
    if line_values[idx0] < point:
        if (idx0 + 1) < len(line_values) and line_values[idx0 + 1] > point:
            return idx0, idx0 + 1
        elif line_values[idx0 - 1] > point:
            return idx0 - 1, idx0
        else:
            return _get_surrounding_indices(line_values, point, ignore_warning)
    else:
        if (idx0 + 1) < len(line_values) and line_values[idx0 + 1] < point:
            return idx0, idx0 + 1
        elif line_values[idx0 - 1] < point:
            return idx0 - 1, idx0
        else:
            return _get_surrounding_indices(line_values, point, ignore_warning)


# find the point where a line between two points (left_point and right_point)
# would cross the "intersect" line
def get_line_crossing(left_point, right_point, intersect):
    m1 = (right_point[1] - left_point[1]) / (right_point[0] - left_point[0])
    b1 = left_point[1] - (m1 * left_point[0])

    m2 = 0
    b2 = intersect

    xi = (b1 - b2) / (m2 - m1)
    yi = m1 * xi + b1

    return xi, yi


def get_lsf_esf(
    image,
    center_point,
    axis=0,
    avg_num=101,
    plot_range=75,
    show=True,
    return_indices=False,
    ignore_warning=False,
):
    # might eventually want to interpolate more

    def fail():
        if return_indices:
            return (-1, -1, -1, -1, -1, -1, -1)
        else:
            return -1, -1, -1

    # what if we determine that the line is too close to the edge of the image?
    min_distance = 10
    if center_point[1] < min_distance or center_point[0] < min_distance:
        return fail()
    if (image.shape[1] - center_point[0]) < min_distance:
        return fail()
    if (image.shape[0] - center_point[1]) < min_distance:
        return fail()

    if axis == 0:
        xstart = max(int(center_point[1] - avg_num / 2), 0)
        xend = min(int(center_point[1] + avg_num / 2), image.shape[0])
        ystart = max(int(center_point[0] - plot_range / 2), 0)
        yend = min(int(center_point[0] + plot_range / 2), image.shape[1])
    elif axis == 1:
        xstart = max(int(center_point[1] - plot_range / 2), 0)
        xend = min(int(center_point[1] + plot_range / 2), image.shape[0])
        ystart = max(int(center_point[0] - avg_num / 2), 0)
        yend = min(int(center_point[0] + avg_num / 2), image.shape[1])
    else:
        raise ValueError(f"axis must be 0 or 1, not {axis}")
    subimage = image[xstart:xend, ystart:yend]

    def fail():
        if return_indices:
            return (-1, -1, -1, -1, -1, -1, subimage)
        else:
            return -1, -1, subimage

    if show:
        plt.figure()
        plt.imshow(subimage)
        plt.show()

    esf = np.mean(subimage, axis=axis)
    
    # 2023/02/09 trying new way of calculating lsf
    lsf = np.zeros(len(esf))
    lsf[1:-1] = esf[2:] / 2 - esf[:-2] / 2

    # find first intersection point for lower half max
    half_max = np.min(lsf) / 2
    min_index = np.where(lsf == np.min(lsf))[0][0]
    idx0, idx1 = get_surrounding_indices(lsf[: min_index + 1], half_max, ignore_warning)
    if idx0 == -1:
        return fail()
    intersect0 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max)[0]

    # find second intersection point for lower half max
    idx0, idx1 = get_surrounding_indices(lsf[min_index:], half_max, ignore_warning)
    if idx0 == -1:
        return fail()
    idx0 = idx0 + min_index
    idx1 = idx1 + min_index
    intersect1 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max)[0]

    # find first intersection point for higher half max
    half_max = np.max(lsf) / 2
    max_index = np.where(lsf == np.max(lsf))[0][0]
    idx0, idx1 = get_surrounding_indices(lsf[: max_index + 1], half_max, ignore_warning)
    if idx0 == -1:
        return fail()
    intersect2 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max)[0]

    # find second intersection point for higher half max
    idx0, idx1 = get_surrounding_indices(lsf[max_index:], half_max, ignore_warning)
    if idx0 == -1:
        return fail()
    idx0 = idx0 + max_index
    idx1 = idx1 + max_index
    intersect3 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max)[0]

    if show:
        plt.figure()
        plt.plot(
            esf - np.mean(esf),
            ".-",
            label="ESF",
        )
        plt.plot(
            lsf,
            ".-",
            label="LSF",
        )
        plt.legend()
        plt.title("ESF and LSF")
        plt.xlabel("Pixel Index")
        plt.axhline(y=np.max(lsf) / 2, color="r", linestyle="--")
        plt.axhline(y=np.min(lsf) / 2, color="r", linestyle="--")
        plt.axvline(x=intersect0, color="b", linestyle="--")
        plt.axvline(x=intersect1, color="b", linestyle="--")
        plt.axvline(x=intersect2, color="b", linestyle="--")
        plt.axvline(x=intersect3, color="b", linestyle="--")

        plt.show()

    if return_indices:
        return lsf, esf, xstart, xend, ystart, yend, subimage
    return lsf, esf, subimage


# take information about a point in an image
# and re-center it around a detected line
# this is an improvement on a previous function
# which we might want to go back and change
# TODO: don't always use default lsf_range
def adjust_center(image, binary_image, center_point, axis, show=False, lsf_range=150):
    lsf, esf, xstart, xend, ystart, yend, sub_image = get_lsf_esf(
        image,
        center_point,
        axis=axis,
        return_indices=True,
        show=False,
        plot_range=lsf_range,
        avg_num=lsf_range,  # TODO: think about this more?
        ignore_warning=True,
    )
    if type(esf) == int and esf == -1:
        return (-1, -1)

    # assess whether there is actually a line here
    # TODO: switch "show" back to False
    lsf2, esf2, xstart2, xend2, ystart2, yend2, sub_image = get_lsf_esf(
        binary_image,
        center_point,
        axis=axis,
        return_indices=True,
        show=False,
        plot_range=lsf_range,
        ignore_warning=True,
    )

    if type(esf2) == int and esf2 == -1:
        return (-1, -1)

    # not totally sure if this is best choice for crop_distance
    valid = determine_line_validity(
        binary_image,
        crop_distance=int(lsf_range / 2),
        center_point=center_point,
        esf=esf2,
        lsf=lsf2,
        show=False,
        show_if_failed=False,
    )
    if not valid:
        # determine_line_validity(binary_image, center_point, axis, esf2, lsf2, show=True)
        return (-1, -1)

    mid_index = np.where(esf == np.min(esf))[0][0]

    indexR = mid_index - 1
    while lsf[indexR] < 0:
        indexR = indexR + 1
    indexL = indexR - 1
    midpoint = get_line_crossing((indexR, lsf[indexR]), (indexL, lsf[indexL]), 0)

    if show:
        plt.figure()
        plt.plot(lsf, ".-", label="lsf")
        plt.plot(np.zeros(len(lsf)))
        plt.plot(esf - np.mean(esf), ".-", label="esf")
        plt.axvline(x=midpoint[0])
        plt.legend()
        plt.title(f"Adjusting point: {center_point}")

    if axis == 1:
        return (center_point[0], xstart + midpoint[0])
    else:
        return (ystart + midpoint[0], center_point[1])


# take a bunch of approximate points in an image
# and return the location of graph vertices
def get_all_vertices(
    image,
    approx_points,
    expected_spacing,
    binary_threshold_values,
    expected_spacing_thresh=40,
    show=False,
    lsf_range=75,
    display_title="Image with vertices",
    display_downsample=1,
):
    binary_image = make_binary_image(image, binary_threshold_values)

    # adjust center in both directions
    vertices = []
    for point in approx_points:
        dir1_adj = adjust_center(
            image, binary_image, point, axis=1, lsf_range=lsf_range
        )
        dir0_adj = adjust_center(
            image, binary_image, point, axis=0, lsf_range=lsf_range
        )

        if -1 in dir0_adj or -1 in dir1_adj:
            continue

        vertices.append([dir0_adj[0], dir1_adj[1]])

    if show:
        radius = int(image.shape[0] * 0.003 * display_downsample)
        display_with_points(
            image,
            np.asarray(vertices, dtype=int),
            radius=radius,
            thickness=math.ceil(radius / 2),
            title=display_title,
            display_downsample=display_downsample,
        )

    return vertices