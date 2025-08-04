# Calculate Mean Radiant Temperature (MRT)

# Read the csv file in data folder and add new column for MRT and save it
import pandas as pd
import os
import pytz
from datetime import timedelta
import folium
import math

file_location = '../dummy/'
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
    # (27.712924292548223, -97.3260302257682), #1
    # (27.71270039170841, -97.32611024223762), #2
    # (27.712229584887485, -97.32528231358658), #3
    # (27.7121665, -97.3248929), #4
    # (27.711947351059774, -97.32467364628596), #5
    # (27.712263215364796, -97.32384499306387), #6
    # (27.71259385702372, -97.32426121800366), #7
    # (27.712579016168924, -97.32446774809237), #8
    # (27.71286752202876, -97.32456564872071), #9
    # (27.71308301047632, -97.3246447738855), #10
    # (27.713303578977833, -97.32452105765262), #11
    # (27.713268221860357, -97.32426318280378), #12
    # (27.713106256296676, -97.32384489976478), #13
    # (27.71332133659946, -97.32330385574681), #14
    # (27.713624681211083, -97.32310872504199), #15
    # (27.713800395122746, -97.3236384613245), #16
    # (27.71403606502429, -97.32407163807902), #17
    # (27.714005196452497, -97.32419569024428), #18
    # (27.714411993287246, -97.3240014969394), #19
    # (27.714369523113312, -97.32383713673069), #20
    # (27.714417606685785, -97.32372247229237), #21
    # (27.714380801977548, -97.32359372626304), #22
    # (27.715097786238747, -97.32361339258881), #23
    # (27.71529071273654, -97.32352085637925), #24
    # (27.714882649788645, -97.32415034502864), #25
    # (27.71469330264857, -97.3242438745645), #26
    # (27.71421246717124, -97.32471728445017), #27
    # (27.71398095303776, -97.32555279255413), #28
    # (27.713700760916392, -97.3257003140584), #29
    # (27.712924292548223, -97.3260302257682),  #30

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

stationary_locations = dict()

def extract_folder_name(file_path):
    result_folder_name = file_path.split("Marty_")[1].split('.')[0]
    result_folder_path = os.path.join(results_location, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    print("Result folder : ", result_folder_path)
    return result_folder_name

def calculate_tmrt(data):
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

def create_path_map(path_idx, path_df, stationary_locations):
    """Create a folium map with multiple paths and location markers with improved visualization for overlapping points"""
    map_center = {'lat': 27.71357483718376, 'lon': -97.32477530414555}
    zoom_level = 18

    # Create map
    m = folium.Map(location=[map_center['lat'], map_center['lon']], zoom_start=zoom_level)

    # Create a feature group for reference points (as a toggleable layer)
    reference_layer = folium.FeatureGroup(name="Reference Points")
    
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
    
    # Improved visualization for stationary points
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
    
    for i, location in enumerate(stationary_locations):
        if i in processed_locations:
            continue
            
        # Find all nearby locations
        nearby_group = [{'idx': i, 'location': location}]
        processed_locations.add(i)
        
        for j, other_location in enumerate(stationary_locations):
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
                        f"Reference Point: {loc.get('reference_point', 'N/A')}<br>"
                        f"Duration: {loc['duration']:.1f} seconds<br>"
                        f"Distance to ref: {loc['reference_distance']:.1f}m<br>"
                        f"From: {loc['start_time']}<br>"
                        f"To: {loc['end_time']}", max_width=400)
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
                            f"Reference Point: {loc.get('reference_point', 'N/A')}<br>"
                            f"Duration: {loc['duration']:.1f} seconds<br>"
                            f"Distance to ref: {loc['reference_distance']:.1f}m<br>"
                            f"From: {loc['start_time']}<br>"
                            f"To: {loc['end_time']}", max_width=400)
                ).add_to(m)

    # Add layer control to toggle reference points on/off
    folium.LayerControl().add_to(m)

    return m


# def find_stationary_locations(path, window_size=10, reference_points=reference_points):
#     """
#     Detect stationary points in the path by finding nearest points to reference points.
#     Assign location numbers based on chronological walking sequence.
#     Returns: (locations, stationary_indices_dict)
#     """
#     start, end, path_df = path[0], path[1], path[2].copy()
#     locations = []
#     path_df['TIMESTAMP'] = pd.to_datetime(path_df['TIMESTAMP'], errors='coerce')

#     # Ensure numeric types for calculations
#     path_df['Full_DecLatitude'] = pd.to_numeric(path_df['Full_DecLatitude'], errors='coerce')
#     path_df['Full_DecLongitude'] = pd.to_numeric(path_df['Full_DecLongitude'], errors='coerce')

#     def haversine(lat1, lon1, lat2, lon2):
#         from math import radians, sin, cos, sqrt, asin
#         R = 6371000
#         phi1, phi2 = radians(lat1), radians(lat2)
#         dphi = radians(lat2 - lat1)
#         dlambda = radians(lon2 - lon1)
#         a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
#         return 2 * R * asin(sqrt(a))

#     # Find all potential matches with distances for each reference point
#     all_matches = []
    
#     # Handle reference points 1 and 30 specially since they're at the same location
#     ref_1_coords = reference_points[0]  # Reference point 1
#     ref_30_coords = reference_points[29]  # Reference point 30
    
#     # Check if ref 1 and ref 30 are at the same location (within 10 meters)
#     same_location = haversine(ref_1_coords[0], ref_1_coords[1], ref_30_coords[0], ref_30_coords[1]) < 10
    
#     for ref_idx, (ref_lat, ref_lon) in enumerate(reference_points):
#         min_distance = float('inf')
#         nearest_idx = None
        
#         # Special handling for reference points 1 and 30 if they're at the same location
#         if same_location and ref_idx in [0, 29]:  # Reference points 1 and 30
#             # For ref point 1, only look in the first half of the path
#             # For ref point 30, only look in the second half of the path
#             if ref_idx == 0:  # Reference point 1
#                 search_range = range(0, len(path_df) // 2)
#             else:  # Reference point 30
#                 search_range = range(len(path_df) // 2, len(path_df))
#         else:
#             search_range = range(len(path_df))
        
#         for i in search_range:
#             row = path_df.iloc[i]
#             if pd.isnull(row['Full_DecLatitude']) or pd.isnull(row['Full_DecLongitude']):
#                 continue
                
#             distance = haversine(ref_lat, ref_lon, row['Full_DecLatitude'], row['Full_DecLongitude'])
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_idx = i
        
#         # Only add if we found a point within reasonable distance (e.g., 50 meters)
#         if nearest_idx is not None and min_distance < 50:
#             all_matches.append({
#                 'ref_idx': ref_idx,
#                 'path_idx': nearest_idx,
#                 'distance': min_distance,
#                 'ref_lat': ref_lat,
#                 'ref_lon': ref_lon
#             })
    
#     # Sort by path index to maintain chronological order
#     all_matches.sort(key=lambda x: x['path_idx'])
    
#     stationary_indices = dict()
#     used_indices = set()
    
#     # Process matches in chronological order but use reference point numbers as location numbers
#     for match in all_matches:
#         path_idx = match['path_idx']
        
#         if path_idx in used_indices:
#             continue
            
#         row = path_df.iloc[path_idx]
        
#         # Calculate duration by looking at surrounding points for time window
#         start_time = row['TIMESTAMP']
#         end_time = row['TIMESTAMP']
#         if path_idx >= window_size and path_idx < len(path_df) - window_size:
#             start_time = path_df.iloc[path_idx - window_size]['TIMESTAMP']
#             end_time = path_df.iloc[path_idx + window_size]['TIMESTAMP']
        
#         # Use reference point number + 1 as location number (so it matches the reference point)
#         location_number = match['ref_idx'] + 1
        
#         locations.append({
#             'Full_DecLatitude': row['Full_DecLatitude'],
#             'Full_DecLongitude': row['Full_DecLongitude'],
#             'duration': (end_time - start_time).total_seconds(),
#             'start_time': start_time,
#             'end_time': end_time,
#             'index': path_idx,
#             'location_number': location_number,
#             'reference_distance': match['distance'],
#             'reference_point': match['ref_idx'] + 1
#         })
#         stationary_indices[path_idx] = location_number
#         used_indices.add(path_idx)
#         print(f"Assigned stationary location {location_number} to reference point {match['ref_idx']+1} at path index {path_idx}, distance: {match['distance']:.1f}m")

#     return locations, stationary_indices

# def find_stationary_locations(path, window_size=10, reference_points=reference_points, max_distance=50):
#     """
#     Find near-path points for reference points, following the path order and prioritizing proximity.
#     Ensures each assigned path point is between previous and next reference, and never reused.
#     """
#     start, end, path_df = path[0], path[1], path[2].copy()
#     locations = []
#     path_df['TIMESTAMP'] = pd.to_datetime(path_df['TIMESTAMP'], errors='coerce')
#     path_df['Full_DecLatitude'] = pd.to_numeric(path_df['Full_DecLatitude'], errors='coerce')
#     path_df['Full_DecLongitude'] = pd.to_numeric(path_df['Full_DecLongitude'], errors='coerce')

#     matches = []
#     used_path_indices = set()

#     def haversine(lat1, lon1, lat2, lon2):
#         from math import radians, sin, cos, sqrt, asin
#         R = 6371000
#         phi1, phi2 = radians(lat1), radians(lat2)
#         dphi = radians(lat2 - lat1)
#         dlambda = radians(lon2 - lon1)
#         a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
#         return 2 * R * asin(sqrt(a))

#     for ref_idx, (ref_lat, ref_lon) in enumerate(reference_points):
#         point_distances = []
#         for i, row in path_df.iterrows():
#             if pd.isnull(row['Full_DecLatitude']) or pd.isnull(row['Full_DecLongitude']):
#                 continue
#             distance = haversine(ref_lat, ref_lon, row['Full_DecLatitude'], row['Full_DecLongitude'])
#             if distance <= max_distance:
#                 point_distances.append((i, distance))
#         # Sort by distance (closest first)
#         point_distances.sort(key=lambda x: x[1])

#         selected_idx = None
#         for idx, distance in point_distances:
#             if idx in used_path_indices:
#                 continue
#             # First reference point: no previous constraint
#             if ref_idx == 0:
#                 selected_idx = idx
#                 break
#             # Others: maintain strict path order
#             prev_selected = next((m['path_idx'] for m in matches if m['ref_idx'] == ref_idx - 1), None)
#             if prev_selected is None or idx > prev_selected:
#                 selected_idx = idx
#                 break

#         if selected_idx is not None:
#             matches.append({
#                 'ref_idx': ref_idx,
#                 'path_idx': selected_idx,
#                 'distance': next(d for (i, d) in point_distances if i == selected_idx),
#             })
#             used_path_indices.add(selected_idx)

#     # Prepare stationary locations output:
#     stationary_indices = dict()
#     for match in matches:
#         path_idx = match['path_idx']
#         row = path_df.loc[path_idx]
#         start_time = row['TIMESTAMP']
#         end_time = row['TIMESTAMP']
#         if path_idx >= window_size and path_idx < len(path_df) - window_size:
#             start_time = path_df.iloc[path_idx - window_size]['TIMESTAMP']
#             end_time = path_df.iloc[path_idx + window_size]['TIMESTAMP']
#         location_number = match['ref_idx'] + 1
#         locations.append({
#             'Full_DecLatitude': row['Full_DecLatitude'],
#             'Full_DecLongitude': row['Full_DecLongitude'],
#             'duration': (end_time - start_time).total_seconds(),
#             'start_time': start_time,
#             'end_time': end_time,
#             'index': path_idx,
#             'location_number': location_number,
#             'reference_distance': match['distance'],
#             'reference_point': match['ref_idx'] + 1
#         })
#         stationary_indices[start + path_idx] = location_number
#         print(
#             f"Assigned stationary location {location_number} to reference point {location_number} "
#             f"at path index {start + path_idx}, distance: {match['distance']:.1f}m"
#         )
#     return locations, stationary_indices

def find_stationary_locations(path, window_size=10, reference_points=reference_points):
    """
    Find stationary points using optimal assignment that prioritizes closest matches
    while maintaining chronological order.
    """
    from math import radians, sin, cos, sqrt, asin

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
        return 2 * R * asin(sqrt(a))

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
    stationary_indices = {}
    
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
        stationary_indices[original_idx] = location_number
        
        print(f"Final: Location {location_number} assigned to reference point {ref_idx + 1} at path index {path_idx}")
    
    print(f"Successfully assigned {len(path_ref_pairs)} out of {n_stops} reference points")
    return locations, stationary_indices

def convert_gmt_to_cst(data):
    """Convert TIMESTAMP column from GMT to US/Central timezone"""
    #7/23/2025 13:56
    if 'TIMESTAMP' in data.columns:
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], errors='coerce', utc=True, format='%m/%d/%Y %H:%M')
        data['TIMESTAMP'] = data['TIMESTAMP'].dt.tz_convert('US/Central').dt.tz_localize(None)  
    return data

def main():
    """Main function to process all Excel files and calculate MRT with stationary locations."""
    for file in os.listdir(file_location):
        if file.endswith('.xlsx'):
            result_folder_name = extract_folder_name(file)
            result_folder_path = os.path.join(results_location, result_folder_name)
            data = pd.read_excel(os.path.join(file_location, file))
            
            # Convert timestamps from GMT to CST
            data = convert_gmt_to_cst(data)
            
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
            
            paths, nan_rows = split_paths(data)
            print(f"Split data into {len(paths)} paths.")

            # Add columns for path and stationary point
            data['PathNumber'] = ''
            data['StationaryNumber'] = ''

            # Initialize all PathNumber values first
            current_path_number = 1
            for i, path in enumerate(paths):
                start_idx, end_idx, _ = path
                for j in range(start_idx, end_idx):
                    if j < len(data):
                        data.at[j, 'PathNumber'] = current_path_number
                current_path_number += 1

            # Process each path for stationary locations
            for i, path in enumerate(paths):
                stationary_locations, stationary_indices = find_stationary_locations(path)

                print(f"Path {i + 1} has {len(stationary_locations)} stationary locations.")
                print(f"Stationary indices: {stationary_indices}")

                # Update the data with stationary numbers
                start_idx, end_idx, _ = path
                for path_idx, stationary_num in stationary_indices.items():
                    # Convert path-relative index to actual data index
                    data.at[path_idx, 'StationaryNumber'] = stationary_num
                    print(f"Set StationaryNumber {stationary_num} at data index {path_idx} (path {i+1}, path_idx {path_idx})")

                path_map = create_path_map(i, path[2], stationary_locations)
                if path_map is not None:
                    path_map.save(os.path.join(result_folder_path, f"path_{i + 1}.html"))
                else:
                    print("No paths to display on the map.")

            # Save the updated dataframe with new columns
            data.to_excel(os.path.join(result_folder_path, file), index=False, na_rep='NaN')
            print(f"Processed {file} and saved to {result_folder_path}")

            calculate_average_mrt(data, result_folder_path)
            print(f"Average MRT calculated and saved to {result_folder_path}")

        else:
            print(f"Skipping {file}, not a .xlsx file.")

if __name__ == "__main__":
    main()