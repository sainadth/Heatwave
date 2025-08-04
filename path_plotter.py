# c:\Users\spagladala1\Documents\PathPlotter\path_plotter.py
import pandas as pd
import folium
from folium import plugins
import numpy as np

# ...existing code...

def load_and_process_data(csv_file):
    """Load CSV data and split into paths at NAN values"""
    df = pd.read_csv(csv_file)
    
    # Remove rows where latitude or longitude is NaN
    df_clean = df.dropna(subset=['latitude', 'longitude'])
    
    # Find indices where NaN values occurred in the original data
    nan_indices = df[df[['latitude', 'longitude']].isna().any(axis=1)].index.tolist()
    
    # Split data into paths
    paths = []
    start_idx = 0
    
    for nan_idx in nan_indices:
        if start_idx < len(df_clean):
            # Get data between NaN breaks
            path_data = df_clean.iloc[start_idx:nan_idx].copy()
            if len(path_data) > 0:
                paths.append(path_data)
            start_idx = nan_idx + 1
    
    # Add remaining data as final path
    if start_idx < len(df_clean):
        final_path = df_clean.iloc[start_idx:].copy()
        if len(final_path) > 0:
            paths.append(final_path)
    
    return paths

def find_stationary_locations(path_df, min_duration=40):
    """Find locations where the entity stayed for >= min_duration seconds"""
    locations = []
    location_counter = 1
    
    if len(path_df) < 2:
        return locations, path_df
    
    path_df = path_df.copy()
    path_df['timestamp'] = pd.to_datetime(path_df['timestamp'])
    
    i = 0
    while i < len(path_df):
        current_lat = path_df.iloc[i]['latitude']
        current_lon = path_df.iloc[i]['longitude']
        start_time = path_df.iloc[i]['timestamp']
        
        # Find consecutive points at same location
        j = i + 1
        while j < len(path_df):
            if (abs(path_df.iloc[j]['latitude'] - current_lat) < 0.0001 and 
                abs(path_df.iloc[j]['longitude'] - current_lon) < 0.0001):
                j += 1
            else:
                break
        
        # Check duration
        if j > i + 1:  # Multiple points at same location
            end_time = path_df.iloc[j-1]['timestamp']
            duration = (end_time - start_time).total_seconds()
            
            if duration >= min_duration:
                locations.append({
                    'latitude': current_lat,
                    'longitude': current_lon,
                    'location_number': location_counter,
                    'duration': duration,
                    'start_time': start_time,
                    'end_time': end_time
                })
                location_counter += 1
        
        i = j if j > i + 1 else i + 1
    
    return locations, path_df

def create_path_map(paths):
    """Create a folium map with multiple paths and location markers"""
    if not paths:
        return None
    
    # Calculate center point for map
    all_lats = []
    all_lons = []
    for path in paths:
        all_lats.extend(path['latitude'].tolist())
        all_lons.extend(path['longitude'].tolist())
    
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
              'gray', 'black', 'lightgray']
    
    for path_idx, path_df in enumerate(paths):
        color = colors[path_idx % len(colors)]
        
        # Find stationary locations for this path
        stationary_locations, _ = find_stationary_locations(path_df)
        
        # Create path line
        coordinates = [[row['latitude'], row['longitude']] for _, row in path_df.iterrows()]
        folium.PolyLine(
            coordinates,
            color=color,
            weight=3,
            opacity=0.8,
            popup=f'Path {path_idx + 1}'
        ).add_to(m)
        
        # Add start marker
        folium.Marker(
            [path_df.iloc[0]['latitude'], path_df.iloc[0]['longitude']],
            popup=f'Path {path_idx + 1} Start',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # Add end marker
        folium.Marker(
            [path_df.iloc[-1]['latitude'], path_df.iloc[-1]['longitude']],
            popup=f'Path {path_idx + 1} End',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
        
        # Add stationary location markers
        for location in stationary_locations:
            folium.Marker(
                [location['latitude'], location['longitude']],
                popup=f"Path {path_idx + 1} - Location {location['location_number']}<br>"
                      f"Duration: {location['duration']:.1f} seconds<br>"
                      f"From: {location['start_time']}<br>"
                      f"To: {location['end_time']}",
                icon=folium.Icon(
                    color='blue', 
                    icon='info-sign',
                    prefix='fa'
                )
            ).add_to(m)
            
            # Add location number as text
            folium.Marker(
                [location['latitude'], location['longitude']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 14px; font-weight: bold; color: white; background-color: blue; border-radius: 50%; width: 25px; height: 25px; text-align: center; line-height: 25px;">{location["location_number"]}</div>',
                    icon_size=(25, 25),
                    icon_anchor=(12, 12)
                )
            ).add_to(m)
    
    return m

def main():
    csv_file = "data.csv"  # Update with your CSV file path
    
    try:
        # Load and process data
        paths = load_and_process_data(csv_file)
        
        if not paths:
            print("No valid paths found in the data.")
            return
        
        print(f"Found {len(paths)} paths in the data.")
        
        # Create map
        map_obj = create_path_map(paths)
        
        if map_obj:
            # Save map
            output_file = "path_map_with_locations.html"
            map_obj.save(output_file)
            print(f"Map saved as {output_file}")
            
            # Print summary
            for i, path in enumerate(paths):
                locations, _ = find_stationary_locations(path)
                print(f"Path {i + 1}: {len(path)} points, {len(locations)} stationary locations")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()