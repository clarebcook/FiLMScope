import numpy as np
import math


# find the vertices which are matched with vertices from a different image
def align_vertices(vertices0, vertices1, expected_shift, shift_limit):
    # Find vertices which are closest to the expected shift away from each other
    # The shift must be within "thresh" of the expected shift
    # those units are in pixels
    new_vertices0 = []
    new_vertices1 = []
    for point in vertices0:
        # find the closest point in vertices2
        diff = vertices1 - point
        distance_sq = np.power(diff[:, 0] - expected_shift[0], 2) + np.power(
            diff[:, 1] - expected_shift[1], 2
        )
        index = np.where(distance_sq == np.min(distance_sq))

        if (
            abs(diff[index[0][0], 0] - expected_shift[0]) > shift_limit
            or abs(diff[index[0][0], 1] - expected_shift[1]) > shift_limit
        ):  # distance_sq[index[0][0]] > thresh**2:
            continue
        match_point = vertices1[index[0][0]]  # [0]

        # check if this point already matched with a different point
        # this is not the best way to do it
        entered = False
        for entered_point in new_vertices1:
            if (
                match_point[0] == entered_point[0]
                and match_point[1] == entered_point[1]
            ):
                entered = True
                break
        if entered:
            continue

        new_vertices0.append(point)
        new_vertices1.append(match_point)

    new_vertices0 = np.asarray(new_vertices0)
    new_vertices1 = np.asarray(new_vertices1)
    return new_vertices0, new_vertices1


def organize_by_axis(vertices, axis, expected_spacing):
    # can revisit the 0.3
    thresh = 0.3 * expected_spacing
    rows = []
    row_locs = []
    for point in vertices:
        location = point[axis]

        if len(np.where(np.abs(np.asarray(row_locs) - location) < thresh)[0]) > 0:
            continue
        points = vertices[np.where(np.abs(vertices[:, axis] - location) < thresh)]
        rows.append(points)
        row_locs.append(location)
    rows = np.asarray(rows, dtype=object)

    # then order the rows correctly
    disordered = np.zeros(len(rows))
    for i, pointset in enumerate(rows):
        disordered[i] = pointset[0, axis]
    indices = np.argsort(disordered)
    rows = rows[indices]

    return rows


def get_vertices_distances(vertices, expected_spacing):
    # find the average x and y ditsances between all the vertices
    # for each point, we'll look at the points to its right and below it, if they exist
    # and add those to the average distances
    def _distance(point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    # note that horizontal and vertical ditsances are according to the graph itself
    # not how it appears when displayed using matplotlib.imshow with default settings
    horizontal_distances = []
    vertical_distances = []
    for i, point in enumerate(vertices[:-1]):
        # could revisit the 0.25
        thresh = 0.25 * expected_spacing
        # points are ordered in columns, so if there's a point below it will be the next point
        next_point = vertices[i + 1]
        # a distance in 0 direction greater than thresh would suggest we've gone onto
        # the next column
        if abs(next_point[0] - point[0]) < thresh and abs(next_point[1] - point[1]) > thresh:
            vertical_distances.append(_distance(point, next_point))

        # then look for a point to the right
        possible_points = vertices[i + 1 :]
        row_points = possible_points[
            np.where(np.abs(possible_points[:, 1] - point[1]) < thresh)
        ]
        if len(row_points) == 0:
            continue

        right_point = row_points[0]
        distance = _distance(point, right_point)
        if distance > thresh:
            horizontal_distances.append(distance)

    return vertical_distances, horizontal_distances

# find a precise average spacing
# given an approximate spacing
def get_average_spacing(vertices, expected_spacing):
    # find the average spacing
    v_distances, h_distances = get_vertices_distances(vertices, expected_spacing)
    v_distances = np.asarray(v_distances)
    h_distances = np.asarray(h_distances)

    # Can revisit the 1.3
    thresh = 1.3 * expected_spacing
    v_distances = v_distances[np.where(v_distances < thresh)]
    h_distances = h_distances[np.where(h_distances < thresh)]

    v_spacing = np.mean(v_distances)
    h_spacing = np.mean(h_distances)

    return h_spacing, v_spacing

# what is display vs array coordinates? 
# "vertices" are in display coordinates
# matrix_shape is in array coordinates
def get_organized_vertices_matrix(
    vertices, expected_spacing, matrix_shape=None, ref_index=None, ref_point=None
):
    low0 = np.min(vertices[:, 0])
    high0 = np.max(vertices[:, 0])
    low1 = np.min(vertices[:, 1])
    high1 = np.max(vertices[:, 1])
    h_spacing, v_spacing = get_average_spacing(vertices, expected_spacing)

    # decide how big the matrix will be
    length0 = int((high0 - low0) / h_spacing + 4)
    length1 = int((high1 - low1) / v_spacing + 4)

    # actual vertices will be organized by array coordinates
    # but vertices themselves will be kept in display coordinates
    if matrix_shape is None:
        actual_vertices = np.zeros((length1, length0, 2))
    else:
        actual_vertices = np.zeros((matrix_shape[0], matrix_shape[1], 2))
    actual_vertices[:, :, :] = np.nan

    # Vertices are still in display coordinates
    # and specified axis is in display coordinates
    # so axis=1 will give display rows 
    # i.e. points with the same y display coordinate
    # and same x array coordinate
    organized_vertices = organize_by_axis(
        vertices.copy(), axis=1, expected_spacing=expected_spacing
    )

    # find the point closest to the ref_point
    if ref_index is None:
        # loop through organized until we find our leftmost point
        for row, pointset in enumerate(organized_vertices):
            if low0 in pointset:
                ref_index = [row, np.where(pointset == low0)[0][0]]
                ref_point = pointset[np.where(pointset == low0)[0][0]]
                break
    else:
        smallest_distance = np.inf
        for row, pointset in enumerate(organized_vertices):
            distances_squared = np.power(pointset[:, 0] - ref_point[0], 2) + np.power(
                pointset[:, 1] - ref_point[1], 2
            )
            distances = np.sqrt(distances_squared.astype(float))
            min_distance = np.min(distances)
            if min_distance < smallest_distance:
                index = np.where(distances == min_distance)[0][0]
                adjusted_ref_point = pointset[index]
                smallest_distance = min_distance
        ref_point = adjusted_ref_point

    # Can revisit the 0.2
    thresh = 0.2 * expected_spacing
    
    # get the order in which we'll go through the display rows in "organized_vertices"
    numbers = np.zeros(actual_vertices.shape[0], dtype=np.uint8)
    numbers[0] = ref_index[0] # we'll start by trying to find the ref point's row
    numbers[1:ref_index[0] + 1] = np.flip(np.arange(ref_index[0])) # then work backwards to row 0
    numbers[ref_index[0] + 1:] = np.arange(ref_index[0] + 1, actual_vertices.shape[0]) # then forwards to the end

    # this is where we'll fill in average display y information for the display rows
    # as we fill them in
    row_locs = np.ones(len(numbers)) * np.nan
    row_locs[ref_index[0]] = ref_point[1] 
    
    mean_ys = np.zeros(len(organized_vertices))
    for i in range(len(mean_ys)):
        mean_ys[i] = np.mean(organized_vertices[i][:, 1])
    
    for row_index in numbers:
        # identify the row with an entry in "row_locs" that is closest to this row index 
        avail_rows_indices = np.where(~np.isnan(row_locs))[0]
        closest_row_index = avail_rows_indices[np.argmin(np.abs(avail_rows_indices - row_index))] 
        closest_row = row_locs[closest_row_index]
        expected_row_loc = closest_row + v_spacing * (row_index - closest_row_index)
        
        # Find the row in our organized vertices that is nearest to the expected row location
        org_row_arg = np.argmin(np.abs(mean_ys - expected_row_loc)) 
        diff = abs(mean_ys[org_row_arg] - expected_row_loc)
        if diff > thresh:
            continue 
        pointset = organized_vertices[org_row_arg]

        # now fill these points into the columns
        # start from the first column and work up until we find a point 
        # check if there's anything in that closest row
        close_row_points = actual_vertices[closest_row_index, :, 0]

        # find all the non-nan column locs
        filled_indices = np.where(~np.isnan(close_row_points))[0]

        if len(filled_indices) < 1:
            ref_col_loc = ref_point[0]
            ref_col_index = ref_index[1]
        else:
            ref_col_index = filled_indices[0]
            ref_col_loc = close_row_points[ref_col_index]
        
        numbers2 = np.zeros(actual_vertices.shape[1], dtype=np.uint8)
        numbers2[0] = ref_col_index # we'll start by trying to find the ref point's row
        numbers2[1:ref_col_index + 1] = np.flip(np.arange(ref_col_index)) # then work backwards to row 0
        numbers2[ref_col_index + 1:] = np.arange(ref_col_index + 1, actual_vertices.shape[1]) # then forwards to the end
        
        col_locs = np.ones(len(numbers2)) * np.nan
        col_locs[ref_col_index] = ref_col_loc
        
        # then fill in the column
        # this probably isn't the best way to do it

        # I've decided if I skip more than 2 points, I should just go on to the next row
        # to avoid errors ?
        skipped = 0
        for col_index in numbers2:
            # find the existing found point that's closest
            avail_col_indices = np.where(~np.isnan(col_locs))[0]
            closest_col_index = avail_col_indices[np.argmin(np.abs(avail_col_indices - col_index))] 
            closest_col = col_locs[closest_col_index]
            exp_loc = closest_col + h_spacing * (col_index - closest_col_index)

            # find the index in the pointset that should go here 
            closest_ind = np.argmin(np.abs(pointset[:, 0] - exp_loc)) 
            closest_loc = pointset[closest_ind, 0] 
            if abs(closest_loc - exp_loc) > thresh:
                skipped = skipped + 1
                if skipped > 2:
                    break
                continue 
 
            actual_vertices[row_index, col_index] = pointset[closest_ind]
            col_locs[col_index] = pointset[closest_ind, 0]
            
        # and get the average value to put in mean_ys
        row_y_vals = actual_vertices[row_index, :, 1]
        mean_y_loc = np.mean(row_y_vals[~np.isnan(row_y_vals)])
        row_locs[row_index] = mean_y_loc
    return actual_vertices

def pick_reference_point(vertices_matrix, next_vertices, threshold):
    # we want to find a point and an index in the vertices matrix
    # that has a matching point in next_vertices (i.e. within some threshold)

    def valid_reference_index(index):
        point = vertices_matrix[index]
        distances_squared = np.power(next_vertices[:, 0] - point[0], 2) + np.power(
            next_vertices[:, 1] - point[1], 2
        )
        distances = np.sqrt(distances_squared.astype(float))
        # print(np.sort(distances))
        min_distance = np.min(distances)


        # get rid of this
        #if min_distance < threshold:
        #    print(point, min_distance, next_vertices[np.argmin(distances)])

        return min_distance < threshold

    # iterate until we find a valid index
    ivals = np.arange(vertices_matrix.shape[0])
    split = int(vertices_matrix.shape[0] / 2)
    ivals = np.roll(ivals, split)

    jvals = np.arange(vertices_matrix.shape[1])
    split = int(vertices_matrix.shape[1] / 2)
    jvals = np.roll(jvals, split)
    for i in ivals:
        for j in jvals:
            valid_index = valid_reference_index((i, j))
            if valid_index:
                index = (i, j)
                break
        if valid_index:
            break

    return index


def get_vertices_matrices(all_vertices, expected_spacing):
    ref_index = None
    ref_point = None
    # we need to make sure we go through the planes in order
    # even if the dictionary stored them out of order 
    plane_numbers = np.sort(np.asarray([x for x in all_vertices.keys()]))

    # find the plane with the MOST vertices, to base our sizes off of
    most_vertices = 0
    best_plane = -1
    for plane_number in plane_numbers:#, vertices in all_vertices.items():
        vertices = all_vertices[plane_number]
        num_vertices = len(vertices)
        if num_vertices > most_vertices:
            most_vertices = num_vertices
            best_plane = plane_number
    matrix = get_organized_vertices_matrix(all_vertices[best_plane], expected_spacing)
    full_shape = np.concatenate(([len(all_vertices)], matrix.shape))
    matrix_shape = matrix.shape
    vertices_matrices = np.zeros(full_shape)
    vertices_matrices[:, :, :, :] = np.nan

    for plane_number in plane_numbers:#, vertices in all_vertices.items():
        # print(f"plane {plane_number}, {ref_point}, {ref_index}")
        vertices = all_vertices[plane_number]
        if len(vertices) == 0:
            continue
        matrix = get_organized_vertices_matrix(
            vertices, expected_spacing, matrix_shape, ref_index, ref_point
        )

        plane_index = np.where(plane_numbers == plane_number)[0][0]
        vertices_matrices[plane_index] = matrix

        if plane_number < len(all_vertices) - 1:
            next_plane = plane_number + 1
            next_vertices = None
            while next_plane < len(all_vertices):
                next_vertices = all_vertices[next_plane]
                if next_vertices is not None:
                    break
                next_plane = next_plane + 1
            if next_vertices is not None:
                # could revisit the 0.3
                ref_index = pick_reference_point(
                    matrix, next_vertices, threshold=0.2 * expected_spacing
                )
                ref_point = matrix[ref_index]

    return vertices_matrices