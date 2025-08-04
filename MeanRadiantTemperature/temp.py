import pandas as pd
import os
import math
import folium

file_location = '../dummy/'
results_location = '../results/'

# Define reference coordinates (replace by your accurate 30 point sequence):
reference_points = [
    (27.7129249, -97.3260006), (27.7127061, -97.3260539), (27.7122306, -97.3251876), (27.7121665, -97.3248929),
    (27.7119789, -97.3246257), (27.7122368, -97.3238686), (27.71259385702372, -97.32426121800366), (27.712579016168924, -97.32446774809237),
    (27.7127601, -97.3245294), (27.71308301047632, -97.3246447738855), (27.713303578977833, -97.32452105765262), (27.7132567, -97.3241851),
    (27.7130994, -97.3238197), (27.7133671, -97.3232363), (27.7136927, -97.3230998), (27.7138313, -97.3236192),
    (27.7140723, -97.3240604), (27.7140427, -97.3241620), (27.7144033, -97.3239575), (27.7143424, -97.3237791),
    (27.714307733244528, -97.32375736803056), (27.71443654975673, -97.3237097588214), (27.71439084068967, -97.32359509439443),
    (27.7152931, -97.3235400), (27.7149058, -97.3241214), (27.7146980, -97.3242367), (27.7142181, -97.3247055),
    (27.7140186, -97.3255118), (27.7137411, -97.3256684), (27.7129249, -97.3260006)
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def extract_folder_name(file_path):
    result_folder_name = file_path.split("Marty_")[1].split('.')[0]
    result_folder_path = os.path.join(results_location, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    print("Result folder : ", result_folder_path)
    return result_folder_name

def calculate_tmrt(data):
    # ...your original calculation...
    return float('nan')  # stub for brevity—insert your real calculation!

def split_paths(data):
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

def find_stationary_locations(path, reference_points):
    start, end, path_df = path[0], path[1], path[2].copy()
    path_df['Full_DecLatitude'] = pd.to_numeric(path_df['Full_DecLatitude'], errors='coerce')
    path_df['Full_DecLongitude'] = pd.to_numeric(path_df['Full_DecLongitude'], errors='coerce')
    n_stops = len(reference_points)
    used_path_indices = set()
    assigned_indices = [None] * n_stops
    # Assign each reference the nearest available path point
    for ref_idx, (ref_lat, ref_lon) in enumerate(reference_points):
        candidates = []
        for i, row in path_df.iterrows():
            if pd.isnull(row['Full_DecLatitude']) or pd.isnull(row['Full_DecLongitude']):
                continue
            dist = haversine(ref_lat, ref_lon, row['Full_DecLatitude'], row['Full_DecLongitude'])
            candidates.append((i, dist))
        candidates.sort(key=lambda x: x[1])
        for i, dist in candidates:
            if i not in used_path_indices:
                assigned_indices[ref_idx] = i
                used_path_indices.add(i)
                break
    # Enforce path order
    order_pairs = sorted(zip(assigned_indices, range(n_stops)))
    ordered_indices = [p[0] for p in order_pairs]
    ordered_ref_indices = [p[1] for p in order_pairs]
    locations = []
    stationary_indices = dict()
    for new_order, path_idx in enumerate(ordered_indices):
        ref_idx = ordered_ref_indices[new_order]
        row = path_df.loc[path_idx]
        locations.append({
            'Full_DecLatitude': row['Full_DecLatitude'],
            'Full_DecLongitude': row['Full_DecLongitude'],
            'index': path_idx,
            'location_number': ref_idx + 1,
            'reference_point': ref_idx + 1
        })
        stationary_indices[path_idx] = ref_idx + 1
    print(f"Assigned all {n_stops} stationary points.")
    return locations, stationary_indices

def create_path_map(path_idx, path_df, stationary_locations):
    map_center = {'lat': 27.71357483718376, 'lon': -97.32477530414555}
    zoom_level = 18
    m = folium.Map(location=[map_center['lat'], map_center['lon']], zoom_start=zoom_level)
    reference_layer = folium.FeatureGroup(name="Reference Points")
    for i, (lat, lon) in enumerate(reference_points):
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 10px; font-weight: bold; color: black; background-color: yellow; border: 2px solid red; border-radius: 50%; width: 18px; height: 18px; text-align: center; line-height: 14px;">{i+1}</div>',
                icon_size=(18, 18), icon_anchor=(9, 9)
            ),
            popup=folium.Popup(f"Reference Point {i+1}", max_width=300)
        ).add_to(reference_layer)
    reference_layer.add_to(m)
    coordinates = [[float(row['Full_DecLatitude']), float(row['Full_DecLongitude'])] for _, row in path_df.iterrows()]
    folium.PolyLine(
        coordinates,
        color='blue',
        weight=3,
        opacity=0.8,
        popup=folium.Popup(f'Path {path_idx + 1}', max_width=300)
    ).add_to(m)
    for loc in stationary_locations:
        folium.Marker(
            [loc['Full_DecLatitude'], loc['Full_DecLongitude']],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12px; font-weight: bold; color: white; background-color: purple; border: 2px solid white; border-radius: 50%; width: 24px; height: 24px; text-align: center; line-height: 20px;">{loc["location_number"]}</div>',
                icon_size=(24, 24), icon_anchor=(12, 12)
            ),
            popup=folium.Popup(f"Stationary Location {loc['location_number']}", max_width=300)
        ).add_to(m)
    folium.LayerControl().add_to(m)
    return m

for file in os.listdir(file_location):
    if file.endswith('.xlsx'):
        result_folder_name = extract_folder_name(file)
        result_folder_path = os.path.join(results_location, result_folder_name)
        data = pd.read_excel(os.path.join(file_location, file))
        data['MRT'] = data.apply(calculate_tmrt, axis=1)
        print("MRT calculated and added to the dataframe.")
        paths, nan_rows = split_paths(data)
        print(f"Split data into {len(paths)} paths.")
        data['PathNumber'] = ''
        data['StationaryNumber'] = ''
        current_path_number = 1
        for i, path in enumerate(paths):
            start_idx, end_idx, _ = path
            for j in range(start_idx, end_idx):
                if j < len(data):
                    data.at[j, 'PathNumber'] = current_path_number
            current_path_number += 1
        for i, path in enumerate(paths):
            stationary_locations, stationary_indices = find_stationary_locations(path, reference_points)
            print(f"Path {i + 1} has {len(stationary_locations)} stationary locations.")
            print(f"Stationary indices: {stationary_indices}")
            start_idx, end_idx, _ = path
            for path_idx, stationary_num in stationary_indices.items():
                actual_data_idx = start_idx + path_idx
                if actual_data_idx < len(data):
                    data.at[actual_data_idx, 'StationaryNumber'] = stationary_num
                    print(f"Set StationaryNumber {stationary_num} at data index {actual_data_idx} (path {i + 1}, path_idx {path_idx})")
            path_map = create_path_map(i, path[2], stationary_locations)
            if path_map is not None:
                path_map.save(os.path.join(result_folder_path, f"path_{i + 1}.html"))
        data.to_excel(os.path.join(result_folder_path, file), index=False, na_rep='NaN')
        print(f"Processed {file} and saved to {result_folder_path}")
    else:
        print(f"Skipping {file}, not a .xlsx file.")
