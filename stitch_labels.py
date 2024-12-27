"""
    通过子图对应的json标签映射为原图json标签,
    子文件夹中包含子图的json标签，根据不同命名格式对不同位置的标签进行处理，top和left的offset为0，另外两条边的offset记录在文件名,

    输入：主文件夹，包含子文件夹，每一个子文件夹代表一个大图对应的小图的所有json标签,
    输出：文件夹包含大图的json标签
"""

import json
import os


def adjust_geojson_labels(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each subfolder in the input folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Initialize the new geojson data
            all_features = []

            # Process each geojson file in the subfolder
            for geojson_file in os.listdir(subfolder_path):
                if geojson_file.endswith('.geojson'):
                    geojson_path = os.path.join(subfolder_path, geojson_file)

                    with open(geojson_path, 'r') as f:
                        data = json.load(f)

                    # Extract the part of the name (top, left, right, or bottom)
                    part_name = geojson_file.split('_')[1]  # top, left, right, or bottom
                    extra_value = int(geojson_file.split('_')[-1].split('.')[0])

                    # Adjust the coordinates in the geojson data
                    for feature in data['features']:
                        coordinates = feature['geometry']['coordinates']
                        if part_name == 'top' or part_name == 'left':
                            # Coordinates remain the same for 'top' and 'left' parts
                            adjusted_coordinates = coordinates
                        elif part_name == 'right':
                            # For 'right', add the extra_value to the x-coordinates
                            adjusted_coordinates = [[x + extra_value, y] for x, y in coordinates]
                        elif part_name == 'bottom':
                            # For 'bottom', add the extra_value to the y-coordinates
                            adjusted_coordinates = [[x, y + extra_value] for x, y in coordinates]

                        # Add the adjusted feature to the list
                        feature['geometry']['coordinates'] = adjusted_coordinates
                        all_features.append(feature)

            # Create the final geojson for this subfolder
            final_geojson = {
                "type": "FeatureCollection",
                "features": all_features
            }

            # Save the final geojson with the original image's name
            output_geojson_path = os.path.join(output_folder, f"{subfolder}.geojson")
            with open(output_geojson_path, 'w') as f:
                # Use json.dump with 'indent' to create a formatted output
                json.dump(final_geojson, f, indent=4)

            print(f"Processed {subfolder}")

# Example usage
input_folder = r"D:\gold\mini_label"
output_folder = r"D:\gold\10_label"
adjust_geojson_labels(input_folder, output_folder)