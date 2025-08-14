# Calculate Mean Radiant Temperature (MRT)

# Read the csv file in data folder and add new column for MRT and save it
import pandas as pd
import os
import folium
from math import radians, sin, cos, sqrt, asin

file_location = '../data/'
results_location = '../results/'

# 6 direction radiation flux densities (K, L) W/m^2

#Back
back_shortwave = "BckSW"
back_longwave = "BckLWCo" 

#Front
front_shortwave = "FntSW"
front_longwave = "FntLWCo" 

#Left
left_shortwave = "LftSW"
left_longwave = "LftLWCo" 

#Right
right_shortwave = "RtSW"
right_longwave = "RtLWCo" 

#Up
up_shortwave = "UpSW"
up_longwave = "UpLWCo" 

#Down
down_shortwave = "DnSW"
down_longwave = "DnLWCo" 

# Absorption coefficients (ak, al)
absorption_coefficient_shortwave = 0.7
absorption_coefficient_longwave = 0.97

# stefan-boltzmann constant (σ)
stefan_boltzmann_constant = 5.670374419e-8  # W/m^2/K^4

# angular coefficients (W)
angular_coefficient_back = 0.22
angular_coefficient_front = 0.22
angular_coefficient_left = 0.22
angular_coefficient_right = 0.22
angular_coefficient_up = 0.06
angular_coefficient_down = 0.06


# route followed
reference_points = [
    (27.7129249, -97.3260006), #1
    (27.7127061, -97.3260539), #2
    (27.7122306, -97.3251876), #3
    (27.7121665, -97.3248929), #4
    (27.7119789, -97.3246257), #5
    (27.7122368, -97.3238686), #6
    (27.71259385702372, -97.32426121800366), #7
    (27.712579016168924, -97.32446774809237), #8
    (27.7127601, -97.3245294), #9
    (27.71308301047632, -97.3246447738855), #10
    (27.713303578977833, -97.32452105765262), #11
    (27.7132567, -97.3241851), #12
    (27.7130994, -97.3238197), #13
    (27.7133671, -97.3232363), #14
    (27.7136927, -97.3230998), #15
    (27.7138313, -97.3236192), #16
    (27.7140723, -97.3240604), #17
    (27.7140427, -97.3241620), #18
    (27.7144033, -97.3239575), #19
    (27.714307733244528, -97.32375736803056), #(27.7143424, -97.3237791), #20
    (27.71443654975673, -97.3237097588214), # (27.7144181, -97.3237087), #21
    (27.71439084068967, -97.32359509439443), #22
    (27.7150817660244, -97.32362061669737), # (27.7151139, -97.3235997), #23
    (27.7152931, -97.3235400), #24
    (27.7149058, -97.3241214), #25
    (27.7146980, -97.3242367), #26
    (27.7142181, -97.3247055), #27
    (27.7140186, -97.3255118), #28
    (27.7137411, -97.3256684), #29
    (27.7129249, -97.3260006), #30
]


def extract_folder_name(file_path):
    """
    Extracts the folder name from the file path and creates a results directory.
    
    This function takes a file path containing 'Marty_' in the name, extracts the portion
    after 'Marty_' and before the file extension to create a unique folder name for storing
    results. If the folder doesn't exist, it creates it.
    
    Args:
        file_path (str): Path to the input file containing 'Marty_' in the name
        
    Returns:
        str: The extracted folder name for storing results
    """
    result_folder_name = file_path.split("Marty_")[1].split('.')[0]
    result_folder_path = os.path.join(results_location, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    print("Result folder : ", result_folder_path)
    return result_folder_name

def calculate_tmrt(data):
    """
    Calculates the Mean Radiant Temperature (MRT) for a single data point.
    
    This function implements the MRT calculation formula using 6-directional radiation
    flux densities (shortwave and longwave), absorption coefficients, angular coefficients,
    and the Stefan-Boltzmann constant. The formula calculates the weighted sum of all
    directional radiation fluxes and converts the result from Kelvin to Celsius.
    
    Args:
        data (pandas.Series): A row of data containing radiation flux values for all 6 directions
        
    Returns:
        float: Calculated MRT in Celsius, or NaN if any required values are missing
    """
    # Skip if any required column contains NaN
    required_columns = [
        back_shortwave, back_longwave,
        front_shortwave, front_longwave,
        left_shortwave, left_longwave,
        right_shortwave, right_longwave,
        up_shortwave, up_longwave,
        down_shortwave, down_longwave
    ]
    
    if any(pd.isnull(data[col]) for col in required_columns):
        # Return NaN to preserve NaN values
        return float('nan')

    # if string cannot be converted to float, raise an error
    try:
        return (
            (
                (
                    (
                        angular_coefficient_back * (absorption_coefficient_shortwave * float(data[back_shortwave]) + absorption_coefficient_longwave * float(data[back_longwave])) 
                        + angular_coefficient_front * (absorption_coefficient_shortwave * float(data[front_shortwave]) + absorption_coefficient_longwave * float(data[front_longwave]))
                        + angular_coefficient_left * (absorption_coefficient_shortwave * float(data[left_shortwave]) + absorption_coefficient_longwave * float(data[left_longwave]))
                        + angular_coefficient_right * (absorption_coefficient_shortwave * float(data[right_shortwave]) + absorption_coefficient_longwave * float(data[right_longwave]))
                        + angular_coefficient_up * (absorption_coefficient_shortwave * float(data[up_shortwave]) + absorption_coefficient_longwave * float(data[up_longwave]))
                        + angular_coefficient_down * (absorption_coefficient_shortwave * float(data[down_shortwave]) + absorption_coefficient_longwave * float(data[down_longwave]))
                    ) / (absorption_coefficient_longwave * stefan_boltzmann_constant)
                ) ** 0.25
            ) - 273.15
        )
    except Exception as e:
        # Return NaN for consistency
        return float('nan')

def check(data):
    """
    Validates calculated MRT values against existing Tmrt values in the dataset.
    
    This function compares the newly calculated 'MRT' column with an existing 'Tmrt'
    column (if present) to verify the accuracy of the calculation. It identifies any
    mismatched values and reports their indices for debugging purposes.
    
    Args:
        data (pandas.DataFrame): DataFrame containing both 'MRT' and 'Tmrt' columns
    """
    #compare the MRT values with Tmrt values
    if 'Tmrt' in data.columns:
        mrt_values = data['MRT']
        tmrt_values = data['Tmrt']
        
        try:
            pd.testing.assert_series_equal(mrt_values, tmrt_values, check_names=False)
            print("MRT and Tmrt values match perfectly!")
        except AssertionError:
            print("MRT and Tmrt values do not match.")
            # print the values that do not match
            mismatched_indices = mrt_values[mrt_values != tmrt_values].index
            print(f"Mismatched indices: {mismatched_indices.tolist()}")

def split_paths(data):
    """
    Splits the dataset into separate paths based on NaN values.
    
    This function identifies rows containing NaN values in any column and uses them
    as natural breakpoints to split the continuous data into separate path segments.
    Each segment represents a distinct walking/measurement path.
    
    Args:
        data (pandas.DataFrame): Input DataFrame with potential NaN rows
        
    Returns:
        tuple: (paths, nan_rows) where paths is a list of [start_idx, end_idx, segment_df]
               and nan_rows is a list of indices containing NaN values
    """
    # Split on rows with NaN in coordinates or any column
    nan_rows = data[data.isnull().any(axis=1)].index.tolist()
    paths = []
    split_points = [0] + nan_rows + [len(data)]
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        segment = data.iloc[start:end].copy()
        if len(segment) > 1:
            segment = segment.reset_index(drop=True)
            paths.append([start, end, segment])
            print(f"Created path with {len(segment)} points")
    return paths, nan_rows

def create_path_map(path_idx, path_df, stop_locations):
    """
    Creates an interactive Folium map visualization for a specific path with stop locations.
    
    This function generates a detailed map showing the walking path as a blue line,
    reference points as numbered yellow markers, start/end points with colored icons,
    and stop measurement locations with purple markers. It handles overlapping
    locations by using extension lines and smart positioning to avoid visual clutter.
    
    Args:
        path_idx (int): Index number of the path being visualized
        path_df (pandas.DataFrame): DataFrame containing the path coordinates and timestamps
        stop_locations (dict): Dictionary containing stop point information
        
    Returns:
        folium.Map: Interactive map object ready for saving or display
    """
    
    map_center = {'lat': 27.71357483718376, 'lon': -97.32477530414555}
    zoom_level = 18

    # Create map
    m = folium.Map(location=[map_center['lat'], map_center['lon']], zoom_start=zoom_level)

    # Add enhanced legend in top left corner to avoid overlapping with layer control
    path_legend_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 10px; width: 200px; min-height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; color: #333;
                padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        <h4 style="margin: 0 0 10px 0; text-align: center; font-size: 16px; font-weight: bold; color: #2c3e50;">Path {path_idx + 1}</h4>
        <hr style="margin: 5px 0; border: 1px solid #bdc3c7;">
        
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <div style="width: 20px; height: 3px; background-color: blue; margin-right: 8px;"></div>
            <span>Walking Path</span>
        </div>
        
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <div style="width: 12px; height: 12px; background-color: green; border-radius: 50%; margin-right: 8px;"></div>
            <span>Start Point</span>
        </div>
        
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <div style="width: 12px; height: 12px; background-color: red; border-radius: 50%; margin-right: 8px;"></div>
            <span>End Point</span>
        </div>
        
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <div style="width: 12px; height: 12px; background-color: purple; border-radius: 50%; border: 1px solid white; margin-right: 8px; color: white; font-size: 8px; text-align: center; line-height: 10px;">1</div>
            <span>Stop Locations</span>
        </div>
        
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <div style="width: 12px; height: 12px; background-color: yellow; border: 1px solid red; border-radius: 50%; margin-right: 8px; color: black; font-size: 8px; text-align: center; line-height: 10px;">R</div>
            <span>Reference Points</span>
        </div>
        
        <div style="margin-top: 10px; font-size: 10px; color: #7f8c8d; text-align: center;">
            Toggle layers using the control panel
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(path_legend_html))

    # Create a feature group for reference points (as a toggleable layer, unselected by default)
    reference_layer = folium.FeatureGroup(name="Reference Points", show=False)
    
    # Add reference points to the layer
    for i, (lat, lon) in enumerate(reference_points):
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 10px; font-weight: bold; color: black; background-color: yellow; border: 2px solid red; border-radius: 50%; width: 18px; height: 18px; text-align: center; line-height: 14px;">{i+1}</div>',
                icon_size=(18, 18),
                icon_anchor=(9, 9)
            ),
            popup=folium.Popup(f"Reference Point {i+1}", max_width=300)
        ).add_to(reference_layer)
    
    # Add the reference layer to the map
    reference_layer.add_to(m)

    path = path_df.copy()
    path.dropna(subset=['Full_DecLatitude', 'Full_DecLongitude', 'TIMESTAMP'], inplace=True)

    # Create path line
    coordinates = [[float(row['Full_DecLatitude']), float(row['Full_DecLongitude'])] for _, row in path.iterrows()]

    folium.PolyLine(
        coordinates,
        color='blue',
        weight=3,
        opacity=0.8,
        popup=folium.Popup(f'Path {path_idx + 1}', max_width=300)
    ).add_to(m)
    
    # Add start marker
    folium.Marker(
        [path.iloc[0]['Full_DecLatitude'], path.iloc[0]['Full_DecLongitude']],
        popup=folium.Popup(f'Path {path_idx + 1} Start', max_width=300),
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    # Add end marker
    folium.Marker(
        [path.iloc[-1]['Full_DecLatitude'], path.iloc[-1]['Full_DecLongitude']],
        popup=folium.Popup(f'Path {path_idx + 1} End', max_width=300),
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # Convert stop_locations dict to list for compatibility
    if isinstance(stop_locations, dict):
        stop_locations_list = list(stop_locations.values())
    else:
        stop_locations_list = stop_locations
    
    # Check if we have any stop locations to display
    if not stop_locations_list:
        print(f"No stop locations found for path {path_idx + 1}")
        folium.LayerControl().add_to(m)
        return m
    
    # Improved visualization for stop points
    def calculate_smart_offset(base_lat, base_lon, used_positions, offset_distance=0.00012):
        """Calculate smart offset position that avoids path and other markers (longer line equivalent)"""
        import math
        
        # Try different angles to find non-overlapping position
        angles = [45, 315, 135, 225, 90, 270, 180, 0]  # NE, NW, SE, SW, N, S, W, E
        
        for angle in angles:
            angle_rad = math.radians(angle)
            lat_offset = offset_distance * math.cos(angle_rad)
            lon_offset = offset_distance * math.sin(angle_rad)
            new_lat = base_lat + lat_offset
            new_lon = base_lon + lon_offset
            
            # Check if this position is too close to any used position
            too_close = False
            for used_lat, used_lon in used_positions:
                if abs(new_lat - used_lat) < 0.00005 and abs(new_lon - used_lon) < 0.00005:
                    too_close = True
                    break
            
            if not too_close:
                return new_lat, new_lon
        
        # If all positions are taken, use a larger offset
        angle_rad = math.radians(45)
        lat_offset = offset_distance * 1.5 * math.cos(angle_rad)
        lon_offset = offset_distance * 1.5 * math.sin(angle_rad)
        return base_lat + lat_offset, base_lon + lon_offset

    # Group nearby locations for better visualization
    proximity_threshold = 0.00005  # ~5 meters
    processed_locations = set()
    used_positions = []
    
    for i, location in enumerate(stop_locations_list):
        if i in processed_locations:
            continue
            
        # Find all nearby locations
        nearby_group = [{'idx': i, 'location': location}]
        processed_locations.add(i)
        
        for j, other_location in enumerate(stop_locations_list):
            if j <= i or j in processed_locations:
                continue
                
            lat_diff = abs(location['Full_DecLatitude'] - other_location['Full_DecLatitude'])
            lon_diff = abs(location['Full_DecLongitude'] - other_location['Full_DecLongitude'])
            
            if lat_diff < proximity_threshold and lon_diff < proximity_threshold:
                nearby_group.append({'idx': j, 'location': other_location})
                processed_locations.add(j)
        
        # Add markers for the group
        if len(nearby_group) == 1:
            # Single location - place numbered marker directly on path
            loc = nearby_group[0]['location']
            
            # Numbered marker directly on the path point
            folium.Marker(
                [loc['Full_DecLatitude'], loc['Full_DecLongitude']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12px; font-weight: bold; color: white; background-color: purple; border: 2px solid white; border-radius: 50%; width: 20px; height: 20px; text-align: center; line-height: 16px;">{loc["location_number"]}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                ),
                popup=folium.Popup(f"Path {path_idx + 1} - Location {loc['location_number']}<br>"
                        f"Duration: {loc.get('duration', 45):.1f} seconds<br>"
                        f"From: {loc.get('start_time', 'N/A')}<br>"
                        f"To: {loc.get('end_time', 'N/A')}", max_width=400)
            ).add_to(m)
            
        else:
            # Multiple overlapping locations - use bigger circle markers on path with bigger extension lines
            base_lat = nearby_group[0]['location']['Full_DecLatitude']
            base_lon = nearby_group[0]['location']['Full_DecLongitude']
            
            # Bigger circle marker at the actual overlapping location on path
            folium.CircleMarker(
                [base_lat, base_lon],
                radius=6,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.9,
                weight=3,
                popup=folium.Popup(f"Overlapping location with {len(nearby_group)} points", max_width=400)
            ).add_to(m)
            
            for idx, item in enumerate(nearby_group):
                loc = item['location']
                
                # Calculate smart offset position for each overlapping point (longer line)
                label_lat, label_lon = calculate_smart_offset(base_lat, base_lon, used_positions, offset_distance=0.00012)
                used_positions.append((label_lat, label_lon))
                
                # Longer extension line
                folium.PolyLine(
                    [[base_lat, base_lon], [label_lat, label_lon]],
                    color='purple',
                    weight=4,
                    opacity=0.9
                ).add_to(m)
                
                # Number label at end of extension line
                folium.Marker(
                    [label_lat, label_lon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12px; font-weight: bold; color: white; background-color: purple; border: 2px solid white; border-radius: 50%; width: 20px; height: 20px; text-align: center; line-height: 16px;">{loc["location_number"]}</div>',
                        icon_size=(20, 20),
                        icon_anchor=(10, 10)
                    ),
                    popup=folium.Popup(f"Path {path_idx + 1} - Location {loc['location_number']}<br>"
                            f"Duration: {loc.get('duration', 45):.1f} seconds<br>"
                            f"From: {loc.get('start_time', 'N/A')}<br>"
                            f"To: {loc.get('end_time', 'N/A')}", max_width=400)
                ).add_to(m)

    # Add layer control to toggle reference points on/off
    folium.LayerControl().add_to(m)

    return m


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * asin(sqrt(a))


def find_stop_locations(path, window_size=10, reference_points=reference_points):
    """
    Identifies stop measurement locations along a path by matching them to reference points.
    
    This function uses an optimal assignment algorithm to match actual GPS coordinates from
    the walking path to predefined reference points. It maintains chronological order while
    prioritizing closest distance matches. Special handling is applied to the start and end
    points (reference points 1 and 30) to ensure proper path continuity.
    
    Args:
        path (list): Path information as [start_idx, end_idx, path_dataframe]
        window_size (int): Number of points before/after each location for duration calculation
        reference_points (list): List of (latitude, longitude) tuples for reference locations
        
    Returns:
        tuple: (locations, stop_indices) where locations is a list of location dictionaries
               and stop_indices maps original dataframe indices to location numbers
    """

    start, end, path_df = path[0], path[1], path[2].copy()
    path_df['TIMESTAMP'] = pd.to_datetime(path_df['TIMESTAMP'], errors='coerce')
    path_df['Full_DecLatitude'] = pd.to_numeric(path_df['Full_DecLatitude'], errors='coerce')
    path_df['Full_DecLongitude'] = pd.to_numeric(path_df['Full_DecLongitude'], errors='coerce')
    
    # Sort path by timestamp to ensure chronological order
    path_df = path_df.sort_values('TIMESTAMP').reset_index(drop=True)
    
    n_stops = len(reference_points)
    
    # Create all reference-point pairs with their distances
    ref_point_pairs = []
    for ref_idx, (ref_lat, ref_lon) in enumerate(reference_points):
        for path_idx, row in path_df.iterrows():
            if pd.isnull(row['Full_DecLatitude']) or pd.isnull(row['Full_DecLongitude']):
                continue
            distance = haversine(ref_lat, ref_lon, row['Full_DecLatitude'], row['Full_DecLongitude'])
            
            # Special handling for reference points 1 and 30 (same location)
            if ref_idx == 0:  # Reference point 1 - strongly prefer early path points
                distance += path_idx * 0.01  # Larger penalty for later points
            elif ref_idx == 29:  # Reference point 30 - strongly prefer late path points  
                distance += (len(path_df) - path_idx) * 0.01  # Larger penalty for earlier points
            
            ref_point_pairs.append((ref_idx, path_idx, distance))
    
    # Sort by distance to prioritize best matches
    ref_point_pairs.sort(key=lambda x: x[2])
    
    # Pre-assign reference points 1 and 30 to ensure they get proper positions
    assignments = {}  # ref_idx -> path_idx
    used_path_indices = set()
    
    # Find best assignment for reference point 1 (prefer beginning of path)
    ref_1_lat, ref_1_lon = reference_points[0]
    best_start_idx = None
    best_start_distance = float('inf')
    for path_idx in range(min(len(path_df) // 3, 500)):  # Search first third or 500 points
        row = path_df.iloc[path_idx]
        if pd.isnull(row['Full_DecLatitude']) or pd.isnull(row['Full_DecLongitude']):
            continue
        distance = haversine(ref_1_lat, ref_1_lon, row['Full_DecLatitude'], row['Full_DecLongitude'])
        if distance < best_start_distance:
            best_start_distance = distance
            best_start_idx = path_idx
    
    if best_start_idx is not None:
        assignments[0] = best_start_idx
        used_path_indices.add(best_start_idx)
        print(f"Pre-assigned reference 1 to path index {best_start_idx}, distance: {best_start_distance:.1f}m")
    
    # Find best assignment for reference point 30 (prefer end of path)
    ref_30_lat, ref_30_lon = reference_points[29]
    best_end_idx = None
    best_end_distance = float('inf')
    start_search = max(len(path_df) * 2 // 3, len(path_df) - 500)  # Search last third or 500 points
    for path_idx in range(start_search, len(path_df)):
        if path_idx in used_path_indices:
            continue
        row = path_df.iloc[path_idx]
        if pd.isnull(row['Full_DecLatitude']) or pd.isnull(row['Full_DecLongitude']):
            continue
        distance = haversine(ref_30_lat, ref_30_lon, row['Full_DecLatitude'], row['Full_DecLongitude'])
        if distance < best_end_distance:
            best_end_distance = distance
            best_end_idx = path_idx
    
    if best_end_idx is not None:
        assignments[29] = best_end_idx
        used_path_indices.add(best_end_idx)
        print(f"Pre-assigned reference 30 to path index {best_end_idx}, distance: {best_end_distance:.1f}m")
    
    # Now assign the remaining reference points (2-29)
    for ref_idx, path_idx, distance in ref_point_pairs:
        # Skip if already assigned or if this is reference point 1 or 30 (already handled)
        if ref_idx in assignments or path_idx in used_path_indices or ref_idx in [0, 29]:
            continue
        
        # Check chronological constraints with already assigned points
        valid = True
        for assigned_ref, assigned_path in assignments.items():
            # If this reference comes before an assigned one, its path index must be smaller
            if ref_idx < assigned_ref and path_idx >= assigned_path:
                valid = False
                break
            # If this reference comes after an assigned one, its path index must be larger
            if ref_idx > assigned_ref and path_idx <= assigned_path:
                valid = False
                break
        
        if valid:
            assignments[ref_idx] = path_idx
            used_path_indices.add(path_idx)
            print(f"Assigned reference {ref_idx + 1} to path index {path_idx}, distance: {distance:.1f}m")
    
    # Fill in any missing assignments with closest available points
    for ref_idx in range(n_stops):
        if ref_idx in assignments:
            continue
        
        # Find closest available point that maintains order
        best_path_idx = None
        best_distance = float('inf')
        
        ref_lat, ref_lon = reference_points[ref_idx]
        for path_idx, row in path_df.iterrows():
            if path_idx in used_path_indices:
                continue
            if pd.isnull(row['Full_DecLatitude']) or pd.isnull(row['Full_DecLongitude']):
                continue
            
            # Check order constraints
            valid = True
            for assigned_ref, assigned_path in assignments.items():
                if ref_idx < assigned_ref and path_idx >= assigned_path:
                    valid = False
                    break
                if ref_idx > assigned_ref and path_idx <= assigned_path:
                    valid = False
                    break
            
            if valid:
                distance = haversine(ref_lat, ref_lon, row['Full_DecLatitude'], row['Full_DecLongitude'])
                if distance < best_distance:
                    best_distance = distance
                    best_path_idx = path_idx
        
        if best_path_idx is not None:
            assignments[ref_idx] = best_path_idx
            used_path_indices.add(best_path_idx)
            print(f"Fallback: Assigned reference {ref_idx + 1} to path index {best_path_idx}, distance: {best_distance:.1f}m")

    # Sort assignments by path index to get chronological order, then assign sequential location numbers
    path_ref_pairs = [(assignments[ref_idx], ref_idx) for ref_idx in assignments.keys()]
    path_ref_pairs.sort(key=lambda x: x[0])  # Sort by path index
    
    locations = []
    stop_indices = {}
    
    for location_number, (path_idx, ref_idx) in enumerate(path_ref_pairs, 1):
        row = path_df.iloc[path_idx]
        
        # Calculate duration
        start_time = row['TIMESTAMP']
        end_time = row['TIMESTAMP']
        if path_idx >= window_size and path_idx < len(path_df) - window_size:
            start_time = path_df.iloc[path_idx - window_size]['TIMESTAMP']
            end_time = path_df.iloc[path_idx + window_size]['TIMESTAMP']
        
        # Calculate distance for this assignment
        ref_lat, ref_lon = reference_points[ref_idx]
        distance = haversine(ref_lat, ref_lon, row['Full_DecLatitude'], row['Full_DecLongitude'])
        
        locations.append({
            'Full_DecLatitude': row['Full_DecLatitude'],
            'Full_DecLongitude': row['Full_DecLongitude'],
            'duration': (end_time - start_time).total_seconds() if pd.notna(start_time) and pd.notna(end_time) else 45,
            'start_time': start_time,
            'end_time': end_time,
            'index': path_idx,
            'location_number': location_number,  # Sequential: 1, 2, 3, ...
            'reference_distance': distance,
            'reference_point': ref_idx + 1       # Original reference point number
        })
        
        # Map back to original dataframe indices
        original_idx = start + path_idx
        stop_indices[original_idx] = location_number
        
        print(f"Final: Location {location_number} assigned to reference point {ref_idx + 1} at path index {path_idx}")
    
    print(f"Successfully assigned {len(path_ref_pairs)} out of {n_stops} reference points")
    return locations, stop_indices

def add_stops(file_location, timestamps_file_name, data, tolerance=1):
    """
    Assigns stop numbers to data points based on exact timestamps from a timestamps file.
    
    This function loads a timestamps file containing exact stop times and matches them
    to corresponding data points in the main dataset. It identifies periods of zero
    speed around each timestamp and assigns the same stop number to all points in
    that stationary period.
    
    Args:
        file_location (str): Directory path containing the timestamps file
        timestamps_file_name (str): Name of the Excel file with stop timestamps
        data (pandas.DataFrame): Main dataset to add stop assignments to
        
    Returns:
        None: Modifies the data DataFrame in place by adding 'Stop' column values
    """
    # Load the timestamps data
    timestamps_data = pd.read_excel(os.path.join(file_location, timestamps_file_name))
    path_stops = {}
    
    print("Original timestamps data:")
    print(timestamps_data.head())
    
    # Create TIMESTAMP column from Date and Time columns
    timestamps_data["TIMESTAMP"] = pd.to_datetime(
        timestamps_data['Date'].astype(str) + ' ' + timestamps_data['Time'].astype(str), 
        errors='coerce'
    )

    timestamps_data["TIMESTAMP"] = timestamps_data["TIMESTAMP"] + pd.to_timedelta(timestamps_data['TIMESTAMP'].dt.second % 2, unit='s')

    # Initialize Stop column if it doesn't exist
    if 'Stop' not in data.columns:
        data['Stop'] = None
    print(data.columns)
    # Process each stop timestamp
    stop_indices = []
    for idx, row in timestamps_data.iterrows():
        cur_stop_number = row['Stop']
        target_timestamp = row['TIMESTAMP']
        
        print(f"Processing Stop {cur_stop_number} at {target_timestamp}")
        
        target_idx = data[data['TIMESTAMP'] == target_timestamp].index
        if(target_idx.empty):
            #selecting the nth timestamp
            offset = pd.to_datetime(target_timestamp).second // 2
            target_timestamp = pd.to_datetime(target_timestamp).replace(second=0, microsecond=0)
            target_idx = data[data['TIMESTAMP'] == target_timestamp].index + offset

        data.at[target_idx[0], 'Stop'] = cur_stop_number
        r = target_idx[0] + 1
        cnt = 1
        while r < len(data) and cnt < 22 and data.at[r, 'Speed'] == 0:
            # Assign stop number to all points with zero speed after the target timestamp
            data.at[r, 'Stop'] = cur_stop_number
            r += 1
            cnt += 1
        l = target_idx[0] - 1
        while l >= 0 and cnt < 22 and data.at[l, 'Speed'] == 0:
            # Assign stop number to all points with zero speed before the target timestamp
            data.at[l, 'Stop'] = cur_stop_number
            l -= 1
            cnt += 1
        selected_stop = (l + r) // 2 # in the center
        path_stops[data.at[target_idx[0], 'PathNumber']] = path_stops.get(data.at[target_idx[0], 'PathNumber'], {})
        path_stops[data.at[target_idx[0], 'PathNumber']][row['Stop']] = {
            'Full_DecLatitude': data.at[selected_stop, 'Full_DecLatitude'],
            'Full_DecLongitude': data.at[selected_stop, 'Full_DecLongitude'],
            'duration': 45,
            'start_time': data.at[l + 1, 'TIMESTAMP'],
            'end_time': data.at[r - 1, 'TIMESTAMP'],
            'index': data.at[target_idx[0], 'PathNumber'],
            'location_number': row['Stop'],
        }
        stop_indices.append(selected_stop)

        print(f"Assigned Stop {cur_stop_number} to indices from {target_idx[0] + 1} to {r - 1} = {r - (target_idx[0] + 1)} (zero-speed points).")
    return path_stops, stop_indices

def calculate_average_mrt(data, stop_indices, result_folder_path, file_name):
    """
    Calculates average MRT values for stop points and saves results to Excel files.
    
    This function performs two types of averaging:
    1. Local averaging: For each stop point, calculates average MRT using 3 points
       before and after the stop location
    2. Global averaging: Calculates overall average MRT for each of the 30 reference
       point locations across all measurements
    
    The function saves two Excel files:
    - AVG_[filename]: Contains all data points with local averages for stop locations
    - AVG_STOP_[filename]: Contains one row per reference point with global averages
    
    Args:
        data (pandas.DataFrame): Complete dataset with MRT values and location assignments
        stop_indices_all_paths (list): List of all stop point indices across paths
        result_folder_path (str): Directory path where results should be saved
        file_name (str): Original filename to use as base for output filenames
    """
    #for every stop point, calculate the average MRT by selecting 3 rows before and after the stop point and 
    # calculate the average MRT for each stop point and save it to a new column 'Average_MRT'
    if 'MRT' not in data.columns:
        print("MRT column not found in data, cannot calculate average MRT.")
        return
        
    #selecting specific columns for saving
    columns_to_save = [
        'TIMESTAMP', 'Full_DecLatitude', 'Full_DecLongitude', 'MRT',
        'PathNumber', 'Stop'
    ]
    data = data[columns_to_save].copy()

    if 'Average_MRT' in data.columns:
        data.drop(columns=['Average_MRT'], inplace=True)
    data['Average_MRT'] = None

    # place average MRT column at the 4th column position 
    cols = data.columns.tolist()
    cols.remove('Average_MRT')
    cols.insert(3, 'Average_MRT')
    data = data[cols]

    for idx in stop_indices:
        print(idx)
        # Get the range of indices to consider for averaging
        start_idx = max(0, idx - 3)
        end_idx = min(len(data), idx + 4)

        # Select the rows in the specified range
        selected_rows = data.iloc[start_idx:end_idx]
        # Calculate the average MRT for these rows
        average_mrt = selected_rows['MRT'].mean()
        # Update the Average_MRT column for the stop point
        data.at[idx, 'Average_MRT'] = average_mrt
    data['Stop'] = data['Stop'].replace('', pd.NA)
    data.dropna(subset=['Stop', 'Average_MRT'], inplace=True)

    # Save the updated DataFrame with Average MRT to a new Excel file
    output_file = os.path.join(result_folder_path, str("AVG_" + file_name))
    data.to_excel(output_file, index=False)
    print(f"Average MRT values saved to {output_file}")


    #calculate the average MRT for each stop point and save it to a new column 'Stationary_Average_MRT'
    result = pd.DataFrame()
    result['Full_DecLatitude'] = None
    result['Full_DecLongitude'] = None
    result['Stop'] = None
    result['Stationary_Average_MRT'] = None

    for i in range(1, 31):
        stop_rows = data[data['Stop'] == i]
        if not stop_rows.empty:
            average_mrt = stop_rows['Average_MRT'].mean()
            result.loc[i, 'Stop'] = i
            result.loc[i, 'Full_DecLatitude'] = reference_points[i-1][0]
            result.loc[i, 'Full_DecLongitude'] = reference_points[i-1][1]
            result.loc[i, 'Stationary_Average_MRT'] = average_mrt
        else:
            print(f"No data found for Stop {i}, skipping.")
    # Save the updated DataFrame with  Average Stationary point MRT to a new Excel file
    output_file = os.path.join(result_folder_path, str("AVG_STOP_" + file_name))
    result.to_excel(output_file, index=False)
    print(f"Average Stationary MRT values saved to {output_file}")

def convert_gmt_to_cst(data):
    """
    Converts timestamp data from GMT (Greenwich Mean Time) to CST (Central Standard Time).
    
    This function handles timezone conversion for the TIMESTAMP column, converting from
    UTC/GMT to US/Central timezone and removing timezone information for cleaner display.
    It uses pandas datetime parsing with error handling for malformed timestamps.
    
    Args:
        data (pandas.DataFrame): DataFrame containing a 'TIMESTAMP' column in GMT format
        
    Returns:
        pandas.DataFrame: DataFrame with TIMESTAMP column converted to CST
    """
    """Convert TIMESTAMP column from GMT to US/Central timezone"""
    #7/23/2025 13:56
    if 'TIMESTAMP' in data.columns:
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], errors='coerce', utc=True, format='%m/%d/%Y %H:%M')
        data['TIMESTAMP'] = data['TIMESTAMP'].dt.tz_convert('US/Central').dt.tz_localize(None)  
    return data
def main():
    """
    Main processing function that orchestrates the complete MRT analysis workflow.
    
    This function serves as the primary entry point and coordinates all analysis steps:
    1. Processes all Excel files in the data directory
    2. Converts timestamps from GMT to CST
    3. Calculates MRT values for each data point
    4. Splits data into separate paths based on NaN values
    5. Identifies stop measurement locations along each path
    6. Creates interactive maps for visualization
    7. Calculates and saves average MRT values
    8. Saves processed data with path and location assignments
    
    The function handles multiple files automatically and creates organized output
    directories for each processed dataset.
    """
    """Main function to process all Excel files and calculate MRT with stop locations."""
    for file in os.listdir(file_location):
        if file.endswith('.xlsx') and file.startswith('Marty_'):
            # print(f"Processing file: {file}")
            timestamps_file_name = f"Timestamps_{file.split('Marty_')[1].split('.')[0]}.xlsx"
            # print(f"Generated timestamps file name: {timestamps_file_name}")
            # break
            result_folder_name = extract_folder_name(file)
            result_folder_path = os.path.join(results_location, result_folder_name)
            data = pd.read_excel(os.path.join(file_location, file))
            

            #################################################################################################
            # Convert timestamps from GMT to CST
            data = convert_gmt_to_cst(data)
            #################################################################################################


            #################################################################################################
            # Check if all required columns exist in the DataFrame
            required_columns = [
                back_shortwave, back_longwave,
                front_shortwave, front_longwave,
                left_shortwave, left_longwave,
                right_shortwave, right_longwave,
                up_shortwave, up_longwave,
                down_shortwave, down_longwave
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"Missing columns in {file}: {missing_columns}")
                continue
            
            # Apply TMRT calculation to each row
            data['MRT'] = data.apply(calculate_tmrt, axis=1)
            print("MRT calculated and added to the dataframe.")

            # Reorder columns to put MRT at position 1 (2nd position)
            cols = data.columns.tolist()
            cols.remove('MRT')
            cols.insert(2, 'MRT')
            data = data[cols]

            check(data)
            #################################################################################################


            #################################################################################################
            paths, nan_rows = split_paths(data)
            print(f"Split data into {len(paths)} paths.")

            # Add columns for path and stop point
            data['PathNumber'] = ''
            data['Stop'] = ''

            # Initialize all PathNumber values first
            current_path_number = 1
            for i, path in enumerate(paths):
                start_idx, end_idx, _ = path
                for j in range(start_idx, end_idx):
                    if j < len(data):
                        data.at[j, 'PathNumber'] = current_path_number
                current_path_number += 1

            # Process each path for stop locations
            # stop_locations, stop_indices = 
            path_stops, stop_indices = add_stops(file_location, timestamps_file_name, data)
            print("Stop points added to the data based on timestamps.")
            # print(path_stops)
            # break
            for i, path in enumerate(paths):
                # Get stop locations for this specific path
                current_path_stops = path_stops.get(i + 1, {})
                
                path_map = create_path_map(i, path[2], current_path_stops)
                if path_map is not None:
                    path_map.save(os.path.join(result_folder_path, f"path_{i + 1}.html"))
                else:
                    print("No paths to display on the map.")


            # Save the updated dataframe with new columns
            data.to_excel(os.path.join(result_folder_path, file), index=False, na_rep='NaN')
            print(f"Processed {file} and saved to {result_folder_path}")
            #################################################################################################

            # break
            #################################################################################################
            calculate_average_mrt(data, stop_indices, result_folder_path, file)
            print(f"Average MRT calculated and saved to {result_folder_path}")
            #################################################################################################

        else:
            print(f"Skipping {file}, not a .xlsx file.")

if __name__ == "__main__":
    main()